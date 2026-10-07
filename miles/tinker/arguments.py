from miles.utils.lora.hf_lora_targets import LORA_TARGET_GROUPS, parse_lora_targets

_GLM5_NEXT_NATIVE_SMOKE_HF_TARGETS = {
    "model.language_model.layers.*.mlp.gate_proj",
    "model.language_model.layers.*.mlp.up_proj",
    "model.language_model.layers.*.mlp.down_proj",
    "model.language_model.layers.*.mlp.shared_experts.gate_proj",
    "model.language_model.layers.*.mlp.shared_experts.up_proj",
    "model.language_model.layers.*.mlp.shared_experts.down_proj",
}


def add_tinker_arguments(parser):
    group = parser.add_argument_group("Tinker")

    def add_argument(name, **kwargs):
        return group.add_argument(f"--tinker-{name}", **kwargs)

    add_argument("server-host", default="0.0.0.0")
    add_argument("server-port", type=int, default=10613)
    add_argument(
        "base-model",
        help="Model name advertised by the gateway (default: --hf-checkpoint)",
    )
    add_argument(
        "checkpoint-root",
        help="Directory for tinker:// checkpoints (default: <save>/tinker)",
    )
    add_argument(
        "routed-experts-cache-gb",
        type=float,
        default=64.0,
        help="Host memory for engine-routed experts kept for --use-rollout-routing-replay",
    )
    add_argument(
        "routed-expert-deltas",
        action="store_true",
        help=(
            "With --use-rollout-routing-replay, request routes only past the prompt prefix that an earlier "
            "sample of the same adapter recorded, and rebuild each datum's routes from that chain"
        ),
    )
    add_argument(
        "sampling-support-replay",
        action="store_true",
        help=(
            "Record the engine's top-k/top-p sampling support of each sampled token and renormalize "
            "training log-probs within it; sampled logprobs are returned renormalized the same way"
        ),
    )
    add_argument(
        "fused-loss",
        action="store_true",
        help=(
            "Compute losses from the output layer's input in --log-probs-chunk-size row chunks that backward "
            "recomputes, instead of storing [tokens, vocab/TP] logits"
        ),
    )
    add_argument(
        "engine-affinity",
        action="store_true",
        help=(
            "Send every turn of a rollout to the engine holding its context, and start each rollout on the "
            "engine with the most free request slots and KV; without it the router places every request"
        ),
    )
    add_argument(
        "sampler-record-cache-gb",
        type=float,
        default=64.0,
        help="Host memory for sampler records (supports and route deltas), evicted oldest chain first",
    )
    return parser


def configure_tinker_args(args):
    assert args.train_backend == "megatron", "Tinker requires the Megatron backend"
    assert args.exclude_modules is None, "Tinker selects complete training groups; --exclude-modules is not supported"
    if args.tinker_fused_loss:
        assert args.log_probs_chunk_size > 0, "--tinker-fused-loss bounds its logits by --log-probs-chunk-size"
        assert not args.true_on_policy_mode, "--tinker-fused-loss does not score the true-on-policy full vocabulary"
    groups = parse_lora_targets(args.target_modules)
    if groups is None:
        groups = list(LORA_TARGET_GROUPS)
    if "glm5_next" in (args.model_name or "").lower():
        if set(groups) == _GLM5_NEXT_NATIVE_SMOKE_HF_TARGETS:
            args.tinker_lora_groups = ["mlp"]
            args.target_modules = groups
            return
    assert set(groups) <= set(LORA_TARGET_GROUPS), "Tinker --target-modules accepts only attn,mlp,unembed groups"
    args.tinker_lora_groups = groups
    args.target_modules = groups
