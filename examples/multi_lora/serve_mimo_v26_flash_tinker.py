"""Serve MiMo-V2.6-Flash-RL through Miles' Tinker-compatible multi-LoRA gateway on B300.

The SGLang engines serve the official checkpoint (FP8 attention, MXFP4 experts) with the
SGLang cookbook kernels. The trainer loads its BF16 conversion (tools/convert_mimo_v2_to_bf16.py,
passed as ``--ref-load``). Adapters train the attention projections only: the experts and the
router stay frozen, and the engine applies each adapter without re-quantizing base weights.

One node: 4 trainer GPUs (EP4) and one TP4/EP4 engine. Two nodes: 8 trainer GPUs (EP8) and two
engines; start the Ray cluster first and set ``MILES_SCRIPT_EXTERNAL_RAY=1`` on its head.

python examples/multi_lora/serve_mimo_v26_flash_tinker.py serve --hf-checkpoint <official> --ref-load <bf16>
"""

from dataclasses import dataclass, field

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

# SGLang slices the official fused qkv_proj into 4 kv-head shards, so engine attention TP is 4.
_ENGINE_GPUS = 4


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = field(default_factory=U.create_run_id)
    base_model: str = "XiaomiMiMo/MiMo-V2.6-Flash-RL"
    hf_checkpoint: str = "/data/model-cache/mimo-v26/MiMo-V2.6-Flash-RL"
    ref_load: str = "/data/model-cache/mimo-v26/MiMo-V2.6-Flash-RL-bf16"
    # The 4-layer partial (mimo26-p4-bf16 + mimo26-p4-native) uses the same flags with fewer layers.
    megatron_model_type: str = "mimo-v2.6-flash"
    save_dir: str = "/checkpoints"
    megatron_path: str = "/root/Megatron-LM"

    num_gpus_per_node: int = 8
    actor_num_nodes: int = 1
    actor_num_gpus_per_node: int = 4
    rollout_num_gpus: int = 4
    tp: int = 2

    lora_rank: int = 32
    lora_alpha: int = 32
    n_adapters: int = 4
    target_modules: str = "attn"

    tinker_port: int = 10613
    context_length: int = 131072
    max_tokens_per_gpu: int = 65536
    sglang_mem_fraction_static: float = 0.8
    sglang_max_running_requests: int = 64
    # R3 (MiMo-V2.6 section 6.4): the trainer replays the experts the engine routed each sampled token to.
    routing_replay: bool = True
    extra_args: str = ""

    def __post_init__(self) -> None:
        if self.rollout_num_gpus % _ENGINE_GPUS:
            raise ValueError(f"rollout GPUs must be a multiple of the {_ENGINE_GPUS}-GPU engine")
        if self.tp > 4:
            raise ValueError("trainer TP must not exceed the 4 global-attention KV heads")
        if self.target_modules != "attn":
            raise ValueError("MXFP4 engine experts cannot take LoRA; train attention adapters only")


def _serve(args: ScriptArgs) -> None:
    trainer_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
    checkpoint_args = (
        f"--hf-checkpoint {args.hf_checkpoint} --ref-load {args.ref_load} --load {args.ref_load} "
        "--megatron-to-hf-mode bridge "
    )
    lora_args = (
        f"--lora-rank {args.lora_rank} --lora-alpha {args.lora_alpha} --lora-dropout 0.0 "
        f"--multi-lora-n-adapters {args.n_adapters} --target-modules {args.target_modules} "
        "--no-gradient-accumulation-fusion "
    )
    tinker_args = (
        f"--tinker-base-model {args.base_model} --tinker-server-port {args.tinker_port} "
        f"--tinker-checkpoint-root {args.save_dir}/{args.run_id} "
    )
    topology_args = (
        f"--actor-num-nodes {args.actor_num_nodes} --actor-num-gpus-per-node {args.actor_num_gpus_per_node} "
        f"--rollout-num-gpus {args.rollout_num_gpus} --num-gpus-per-node {args.num_gpus_per_node} "
    )
    parallel_args = (
        f"--tensor-model-parallel-size {args.tp} {'--sequence-parallel ' if args.tp > 1 else ''}"
        "--pipeline-model-parallel-size 1 --context-parallel-size 1 "
        f"--expert-model-parallel-size {trainer_gpus} --expert-tensor-parallel-size 1 "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
    )
    batching_args = (
        f"--seq-length {args.context_length} --rollout-max-context-len {args.context_length} "
        f"--max-tokens-per-gpu {args.max_tokens_per_gpu} --micro-batch-size 1 --use-dynamic-batch-size "
        "--qkv-format thd "
    )
    # The SGLang cookbook launch for MiMo-V2.6 on B300 (sglang#40448), as in the Miles MXFP4 engine path.
    sglang_args = (
        f"--rollout-num-gpus-per-engine {_ENGINE_GPUS} --sglang-ep-size {_ENGINE_GPUS} "
        "--sglang-moe-runner-backend deep_gemm --sglang-moe-a2a-backend deepep --sglang-deepep-mode auto "
        "--sglang-attention-backend fa4 --sglang-page-size 1 --sglang-moe-dense-tp-size 1 "
        "--sglang-enable-dp-lm-head --sglang-swa-full-tokens-ratio 0.1 --sglang-chunked-prefill-size 32768 "
        f"--sglang-max-running-requests {args.sglang_max_running_requests} "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} --sglang-lora-backend triton "
    )
    model_args = (
        "--attention-dropout 0.0 --hidden-dropout 0.0 --attention-softmax-in-fp32 --attention-backend fused "
        "--accumulate-allreduce-grads-in-fp32 --optimizer adam --lr 1e-6 "
    )
    replay_args = "--use-rollout-routing-replay " if args.routing_replay else ""
    train_args = (
        f"{checkpoint_args}{lora_args}{tinker_args}{topology_args}{parallel_args}{batching_args}"
        f"{sglang_args}{model_args}{replay_args}{args.extra_args}"
    )
    args.create_backend().execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        train_script="serve_tinker.py",
        megatron_path=args.megatron_path,
        job_lifetime="launcher",
        extra_env_vars={"OMP_NUM_THREADS": "4", "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false"},
    )


@app.command()
@U.dataclass_cli
def serve(args: ScriptArgs) -> None:
    """Start the Tinker gateway, the trainer and the SGLang engines."""
    _serve(args)


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
