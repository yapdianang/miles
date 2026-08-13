from __future__ import annotations

from functools import partial

from miles.ray.specs.addressing import describe_how_the_run_reaches_this_deployment
from miles.ray.specs.entrypoint import compute_specs
from miles.utils.arguments import parse_args
from miles.utils.workers.backend_capability import factory
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.deployment_entrypoint import DeploymentWiring
from miles.utils.workers.ray_worker_manager import RayWorkerManager
from miles.utils.workers.serving.utils import override_argv
from miles.utils.workers.types import ClusterBackend, WorkerCommBackend


def compute_deployment_wiring(argv: list[str]) -> DeploymentWiring:
    with override_argv(argv):
        args = parse_args()
    return DeploymentWiring(
        launch_worker_manager=partial(launch_worker_manager, args),
        describe_reachability=partial(describe_how_the_run_reaches_this_deployment, args),
    )


def launch_worker_manager(args):
    match ClusterBackend(args.cluster_backend):
        case ClusterBackend.KUBERNETES:
            return None
        case ClusterBackend.RAY:
            return _launch_ray_worker_manager(args)


def get_backend_capability(args) -> BackendCapability:
    return factory.get_backend_capability(
        specs=compute_specs(args), cluster_backend=ClusterBackend(args.cluster_backend)
    )


def _launch_ray_worker_manager(args):
    from miles.ray.placement_group import create_placement_groups

    specs = compute_specs(args)
    # TODO: pass in specs instead of args
    pgs = create_placement_groups(args)
    return RayWorkerManager.launch(args, specs, pgs, comm_backend=WorkerCommBackend(args.worker_comm_backend))
