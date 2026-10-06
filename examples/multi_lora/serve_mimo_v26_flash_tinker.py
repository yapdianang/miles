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
    hf_checkpoint: str = (
        "/data/model-cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.6-Flash-RL/snapshots/"
        "5711b268169967567844e1e560e8a3966da959b1"
    )
    ref_load: str = "/data/model-cache/mimo-v26/MiMo-V2.6-Flash-RL-bf16"
    # The 4-layer partial (mimo26-p4-bf16 + mimo26-p4-native) uses the same flags with fewer layers.
    megatron_model_type: str = "mimo-v2.6-flash"
    save_dir: str = "/checkpoints"
    megatron_path: str = "/root/Megatron-LM"

    num_gpus_per_node: int = 8
    actor_num_nodes: int = 1
    actor_num_gpus_per_node: int = 4
    rollout_num_gpus: int = 4
    tp: int = 4

    lora_rank: int = 32
    lora_alpha: int = 32
    n_adapters: int = 4
    target_modules: str = "attn"

    tinker_port: int = 10613
    # The client caps a trajectory's context at 128K (Tau3 MiMo configs); a 250K-token datum runs the TP4
    # trainer out of memory in the unfused fp32 cross entropy.
    context_length: int = 131072
    max_tokens_per_gpu: int = 131072
    # DFlash speculative decoding with the drafter shipped in the checkpoint's dflash/ directory.
    dflash: bool = True
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
    # Xiaomi's verified B300 Flash launch from the SGLang MiMo-V2.6 cookbook (sglang 983e6438, PR #40448),
    # flag for flag. SGLang itself turns FA4 page size 1 into 128 and drops the DP LM head at dp 1.
    # Miles adds LoRA, routed-expert capture (R3), and its own host/port/seed arguments.
    sglang_args = (
        f"--rollout-num-gpus-per-engine {_ENGINE_GPUS} --sglang-ep-size {_ENGINE_GPUS} "
        "--sglang-dp-size 1 --sglang-pp-size 1 "
        "--sglang-moe-runner-backend deep_gemm --sglang-moe-a2a-backend deepep --sglang-deepep-mode auto "
        "--sglang-moe-dense-tp-size 1 --sglang-enable-dp-lm-head "
        "--sglang-log-level-http warning --sglang-enable-cache-report "
        "--sglang-page-size 1 --sglang-cuda-graph-max-bs-decode 64 --sglang-max-running-requests 64 "
        "--sglang-mem-fraction-static 0.6 --sglang-swa-full-tokens-ratio 0.03 "
        "--sglang-chunked-prefill-size 49152 --sglang-max-prefill-tokens 65536 "
        "--sglang-reasoning-parser mimo --sglang-tool-call-parser mimo "
        "--sglang-attention-backend fa4 --sglang-context-length 1048576 "
        "--sglang-cuda-graph-backend-prefill disabled "
        "--sglang-mm-enable-dp-encoder --sglang-mm-attention-backend fa4 "
        "--sglang-lora-backend triton "
    )
    if args.dflash:
        sglang_args += (
            "--sglang-speculative-algorithm DFLASH "
            f"--sglang-speculative-draft-model-path {args.hf_checkpoint}/dflash "
            "--sglang-speculative-num-draft-tokens 8 "
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
