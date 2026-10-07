"""Serve MiMo-V2.6-Flash-RL through Miles' Tinker-compatible multi-LoRA gateway on B300.

The SGLang engines serve the official checkpoint with its FP8 linears dequantized to BF16
(tools/convert_mimo_v2_to_bf16.py --keep-quant --bf16-linears) and run the MXFP4 experts on BF16
activations: FP8 activations in the linears and DeepGEMM W4A8 experts raise the sampler/trainer
k3 KL on Tau3 turns from 0.0009 to 0.0060. The trainer loads the full BF16 conversion
(tools/convert_mimo_v2_to_bf16.py, passed as ``--ref-load``). Adapters train the attention
projections only: the experts and the router stay frozen, and the engine applies each adapter
without re-quantizing base weights.

One node: 4 trainer GPUs (EP4) and one TP4/EP4 engine. Two nodes: 8 trainer GPUs (EP8) and two
engines; start the Ray cluster first and set ``MILES_SCRIPT_EXTERNAL_RAY=1`` on its head.

``prepare`` builds whichever of the two checkpoints is missing (a conversion is finished once its
model.safetensors.index.json exists). It downloads the official checkpoint at ``--base-model-revision``
into ``--hf-hub-cache`` and converts it on the GPU:

  HF_HUB_OFFLINE=0 hf download XiaomiMiMo/MiMo-V2.6-Flash-RL --revision <revision> --cache-dir <hf hub cache>
  python tools/convert_mimo_v2_to_bf16.py --model-dir <snapshot> --save-dir <hf checkpoint> --device cuda \
    --keep-quant --bf16-linears
  python tools/convert_mimo_v2_to_bf16.py --model-dir <snapshot> --save-dir <ref load> --device cuda

where <snapshot> is <hf hub cache>/models--XiaomiMiMo--MiMo-V2.6-Flash-RL/snapshots/<revision>.

python examples/multi_lora/serve_mimo_v26_flash_tinker.py prepare
python examples/multi_lora/serve_mimo_v26_flash_tinker.py serve --hf-checkpoint <bf16 linears> --ref-load <bf16>

tools/mimo_dev.py runs this service on a dev pod and tools/mimo_gates.py measures it; see the README.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

# SGLang slices the official fused qkv_proj into 4 kv-head shards, so engine attention TP is 4.
_ENGINE_GPUS = 4
# Muown for LoRA (MiMo-V2.6 section 5.1): DoRA row magnitudes on Adam, Muon on the LoRA tangent space.
# Extra scale 0.1 is the paper's 0.5 times Muown's Adam-matching 0.2.
_MUOWN_ARGS = (
    "--lora-type dora --muon-momentum 0.95 --muon-nesterov --muon-num-ns-steps 10 --muon-coefficient-type simple "
    "--muon-extra-scale-factor 0.1 "
)


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = field(default_factory=U.create_run_id)
    base_model: str = "XiaomiMiMo/MiMo-V2.6-Flash-RL"
    # A commit hash, which also names the snapshot directory in the hub cache.
    base_model_revision: str = "5711b268169967567844e1e560e8a3966da959b1"
    hf_hub_cache: str = "/data/model-cache/huggingface/hub"
    hf_checkpoint: str = "/data/model-cache/mimo-v26/MiMo-V2.6-Flash-RL-w4a16-linear"
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
    # "muown": clients send section 5.1's lr 3e-6, betas 0.95/0.95, eps 1e-8, no weight decay, clip 1.0 as AdamParams.
    optimizer: str = "adam"

    tinker_port: int = 10613
    # 256K needs fused_loss: the logits path runs the TP4 trainer out of memory on a 250K-token datum.
    context_length: int = 262144
    max_tokens_per_gpu: int = 262144
    # Activation recompute: "full" (every layer), "selective" (core attention only) or "none".
    recompute: str = "full"
    # DFlash speculative decoding with the drafter shipped in the checkpoint's dflash/ directory.
    dflash: bool = True
    # flashinfer_mxfp4 (TRT-LLM on SM100) runs the MXFP4 experts on BF16 activations without an all-to-all;
    # the cookbook's deep_gemm + deepep quantizes expert activations to FP8. A BF16 checkpoint needs triton.
    moe_a2a_backend: str = "none"
    moe_runner_backend: str = "flashinfer_mxfp4"
    flashinfer_mxfp4_moe_precision: str = "bf16"
    sglang_mem_fraction_static: float = 0.6
    # R3 (MiMo-V2.6 section 6.4): the trainer replays the experts the engine routed each sampled token to.
    routing_replay: bool = True
    # Section 6.4: no routes for a cached prefix on later turns; the gateway rebuilds them from earlier samples.
    routed_expert_deltas: bool = True
    # Top-p candidate-set replay (section 6.4): the trainer renormalizes within each sampled token's support.
    # Clients sampling with top_p < 1 must also pass a top_k bound (at most --sglang-sampling-mask-max-tokens).
    sampling_support_replay: bool = True
    # Fused loss (section 6.4): the loss scores hidden states against the output weight in 4096-row chunks
    # that backward recomputes, so the trainer never holds [tokens, vocab/TP] logits.
    fused_loss: bool = True
    extra_args: str = ""

    def __post_init__(self) -> None:
        if self.rollout_num_gpus % _ENGINE_GPUS:
            raise ValueError(f"rollout GPUs must be a multiple of the {_ENGINE_GPUS}-GPU engine")
        if self.tp > 4:
            raise ValueError("trainer TP must not exceed the 4 global-attention KV heads")
        if self.recompute not in ("full", "selective", "none"):
            raise ValueError(f"recompute must be full, selective or none, not {self.recompute}")
        if self.target_modules != "attn":
            raise ValueError("MXFP4 engine experts cannot take LoRA; train attention adapters only")
        if self.optimizer not in ("adam", "muown"):
            raise ValueError(f"optimizer must be adam or muown, not {self.optimizer}")


def _is_converted(path: str) -> bool:
    # The converter writes the index last, so it marks a finished conversion.
    return (Path(path) / "model.safetensors.index.json").exists()


def _prepare(args: ScriptArgs) -> None:
    conversions = {args.hf_checkpoint: " --keep-quant --bf16-linears", args.ref_load: ""}
    pending = {path: flags for path, flags in conversions.items() if not _is_converted(path)}
    if not pending:
        return
    backend = args.create_backend()
    backend.exec_command_cpu(
        f"HF_HUB_OFFLINE=0 hf download {args.base_model} --revision {args.base_model_revision} "
        f"--cache-dir {args.hf_hub_cache}"
    )
    snapshot = f"{args.hf_hub_cache}/models--{args.base_model.replace('/', '--')}/snapshots/{args.base_model_revision}"
    for path, flags in pending.items():
        backend.exec_command_gpu(
            f"python {U.repo_base_dir}/tools/convert_mimo_v2_to_bf16.py "
            f"--model-dir {snapshot} --save-dir {path} --device cuda{flags}"
        )


def _serve(args: ScriptArgs) -> None:
    os.makedirs(f"{args.save_dir}/nccl_trace", exist_ok=True)
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
    )
    if args.recompute == "full":
        parallel_args += "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
    elif args.recompute == "selective":
        parallel_args += "--recompute-granularity selective "
    batching_args = (
        f"--seq-length {args.context_length} --rollout-max-context-len {args.context_length} "
        f"--max-tokens-per-gpu {args.max_tokens_per_gpu} --micro-batch-size 1 --use-dynamic-batch-size "
        "--qkv-format thd "
    )
    # Xiaomi's verified B300 Flash launch from the SGLang MiMo-V2.6 cookbook (sglang 983e6438, PR #40448),
    # except the MoE runner and all-to-all above. SGLang itself turns FA4 page size 1 into 128 and drops
    # the DP LM head at dp 1. Miles adds LoRA, routed-expert capture (R3), and its own host/port/seed arguments.
    sglang_args = (
        f"--rollout-num-gpus-per-engine {_ENGINE_GPUS} --sglang-ep-size {_ENGINE_GPUS} "
        "--sglang-dp-size 1 --sglang-pp-size 1 "
        f"--sglang-moe-runner-backend {args.moe_runner_backend} --sglang-moe-a2a-backend {args.moe_a2a_backend} "
        "--sglang-deepep-mode auto "
        "--sglang-moe-dense-tp-size 1 --sglang-enable-dp-lm-head "
        "--sglang-log-level-http warning --sglang-enable-cache-report "
        "--sglang-page-size 1 --sglang-cuda-graph-max-bs-decode 64 --sglang-max-running-requests 64 "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} --sglang-swa-full-tokens-ratio 0.03 "
        "--sglang-chunked-prefill-size 49152 --sglang-max-prefill-tokens 65536 "
        "--sglang-reasoning-parser mimo --sglang-tool-call-parser mimo "
        "--sglang-attention-backend fa4 --sglang-context-length 1048576 "
        "--sglang-cuda-graph-backend-prefill disabled "
        "--sglang-mm-enable-dp-encoder --sglang-mm-attention-backend fa4 "
        "--sglang-lora-backend triton "
    )
    if args.moe_runner_backend == "flashinfer_mxfp4":
        sglang_args += f"--sglang-flashinfer-mxfp4-moe-precision {args.flashinfer_mxfp4_moe_precision} "
    if args.dflash:
        sglang_args += (
            "--sglang-speculative-algorithm DFLASH "
            f"--sglang-speculative-draft-model-path {args.hf_checkpoint}/dflash "
            "--sglang-speculative-num-draft-tokens 8 "
        )
    model_args = (
        "--attention-dropout 0.0 --hidden-dropout 0.0 --attention-softmax-in-fp32 --attention-backend fused "
        f"--accumulate-allreduce-grads-in-fp32 --optimizer {args.optimizer} --lr 1e-6 "
    )
    if args.optimizer == "muown":
        model_args += _MUOWN_ARGS
    replay_args = "--use-rollout-routing-replay " if args.routing_replay else ""
    if args.routing_replay and args.routed_expert_deltas:
        replay_args += "--tinker-routed-expert-deltas "
    if args.sampling_support_replay:
        replay_args += "--tinker-sampling-support-replay "
    loss_args = "--tinker-fused-loss --log-probs-chunk-size 4096 " if args.fused_loss else ""
    train_args = (
        f"{checkpoint_args}{lora_args}{tinker_args}{topology_args}{parallel_args}{batching_args}"
        f"{sglang_args}{model_args}{replay_args}{loss_args}{args.extra_args}"
    )
    args.create_backend().execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        train_script="serve_tinker.py",
        megatron_path=args.megatron_path,
        job_lifetime="launcher",
        extra_env_vars={
            "OMP_NUM_THREADS": "4",
            "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false",
            # As in the GLM-5.3 launcher: one inductor compile thread per trainer actor.
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            # Keep the NCCL flight recorder so a collective timeout leaves per-rank traces.
            "TORCH_NCCL_TRACE_BUFFER_SIZE": "4096",
            "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
            "TORCH_NCCL_DEBUG_INFO_TEMP_FILE": f"{args.save_dir}/nccl_trace/rank_",
        },
    )


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs) -> None:
    """Build the engine and trainer checkpoints that are missing."""
    _prepare(args)


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
