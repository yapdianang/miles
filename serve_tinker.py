import asyncio
import logging
from contextlib import suppress

import uvicorn

from miles.backends.megatron_utils.megatron_config import compute_trainer_args
from miles.ray.placement_group import create_trainer_handles
from miles.ray.rollout.router_manager import resolve_router_addrs
from miles.ray.specs.inference import compute_router_providers, create_inference_controller_handle
from miles.ray.specs.train import ACTOR_ROLE, compute_trainer_configs
from miles.ray.wiring import get_backend_capability
from miles.tinker.arguments import add_tinker_arguments, configure_tinker_args
from miles.tinker.core.service import TinkerService
from miles.tinker.core.types import GatewayConfig
from miles.tinker.expert_load import moe_layers
from miles.tinker.runtime import MilesBackend, RoutedExpertsCache
from miles.tinker.sampler_records import SamplerRecordStore
from miles.tinker.server.app import build_app
from miles.utils.arguments import parse_args
from miles.utils.async_utils import Disposer, with_disposer
from miles.utils.hf_utils.config import load_hf_config
from miles.utils.http_utils import init_http_client
from miles.utils.orchestration_utils import init_orchestration_script

logger = logging.getLogger(__name__)


async def serve(args, *, disposer: Disposer):
    assert args.multi_lora, "serve_tinker requires --multi-lora-n-adapters > 0"
    # The trainer can load a BF16 conversion (--ref-load) of a quantized engine checkpoint.
    assert args.load in (args.hf_checkpoint, args.ref_load), (
        "Tinker trainers must load the engine's frozen HF base or its --ref-load conversion"
    )
    checkpoint_root = args.tinker_checkpoint_root or (args.save and f"{args.save}/tinker")
    assert checkpoint_root, "set --tinker-checkpoint-root (or --save to derive <save>/tinker)"
    hf_config = load_hf_config(args.hf_checkpoint).get_text_config()
    max_tokens_per_datum = hf_config.max_position_embeddings
    if args.max_tokens_per_gpu is not None:
        # The trainer pads each packed microbatch to this multiple.
        pad_size = args.tensor_model_parallel_size * args.data_pad_size_multiplier
        trainer_token_limit = args.max_tokens_per_gpu // pad_size * pad_size
        max_tokens_per_datum = min(max_tokens_per_datum, trainer_token_limit)
    assert max_tokens_per_datum > 0, "trainer token budget must fit at least one padding block"
    _worker_manager = init_orchestration_script(args, disposer=disposer)
    init_http_client(args)

    capability = get_backend_capability(args)
    await resolve_router_addrs(args, router_providers=compute_router_providers(args, capability=capability))
    inference_controller = create_inference_controller_handle(capability=capability)
    await inference_controller.init()
    disposer.add(inference_controller)

    trainer_configs = compute_trainer_configs(args)
    [actor_config] = [config for config in trainer_configs if config.role == ACTOR_ROLE]
    trainer = create_trainer_handles(args, trainer_configs=trainer_configs)[actor_config.trainer_id]
    await trainer.init(compute_trainer_args(args, actor_config))
    disposer.add(trainer)

    config = GatewayConfig(
        base_model=args.tinker_base_model or args.hf_checkpoint,
        n_slots=args.multi_lora_n_adapters,
        checkpoint_root=checkpoint_root,
        vocab_size=hf_config.vocab_size,
        max_tokens_per_datum=max_tokens_per_datum,
        lora_alpha=args.lora_alpha,
        max_lora_rank=args.lora_rank,
        trains_attn="attn" in args.tinker_lora_groups,
        trains_mlp="mlp" in args.tinker_lora_groups,
        trains_unembed="unembed" in args.tinker_lora_groups,
    )
    router_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    actor_world_size = args.actor_num_nodes * args.actor_num_gpus_per_node
    dp_size = actor_world_size // (
        args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size
    )
    route_deltas = args.use_rollout_routing_replay and args.tinker_routed_expert_deltas
    routed_experts = None
    if args.use_rollout_routing_replay and not route_deltas:
        routed_experts = RoutedExpertsCache(max_bytes=int(args.tinker_routed_experts_cache_gb * 2**30))
    sampler_records = None
    if args.tinker_sampling_support_replay or route_deltas:
        sampler_records = SamplerRecordStore(
            max_bytes=int(args.tinker_sampler_record_cache_gb * 2**30),
            supports=args.tinker_sampling_support_replay,
            routes=route_deltas,
        )
    backend = MilesBackend(
        trainer,
        router_url,
        dp_size=dp_size,
        inference_controller=inference_controller,
        routed_experts=routed_experts,
        num_layers=args.num_layers,
        sampler_records=sampler_records,
        moe_layers=moe_layers(args.moe_layer_freq, args.num_layers) if args.num_experts else None,
        num_experts=args.num_experts,
    )
    service = TinkerService(backend, config)

    server = uvicorn.Server(
        uvicorn.Config(
            build_app(service), host=args.tinker_server_host, port=args.tinker_server_port, log_level="info"
        )
    )
    logger.info(f"tinker gateway serving {config.base_model} on :{args.tinker_server_port}")
    # supervise both: a crashed dispatcher must take the HTTP server down with it,
    # not keep answering /healthz while every training future pends forever
    service_task = asyncio.create_task(service.run())
    server_task = asyncio.create_task(server.serve())
    try:
        done, _ = await asyncio.wait({service_task, server_task}, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    finally:
        for task in (service_task, server_task):
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


if __name__ == "__main__":
    args = parse_args(add_tinker_arguments, entry="serve", preprocess_args=configure_tinker_args)
    # commands ship one work unit at a time; its size is the batch size
    args.use_dynamic_global_batch_size = True
    args.delay_split_train_data_by_dp = True
    asyncio.run(with_disposer(serve, args))
