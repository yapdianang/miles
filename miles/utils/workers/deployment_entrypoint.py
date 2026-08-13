import argparse
import asyncio
import logging
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from miles.utils.arguments import parse_args
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.function_registry import load_function
from miles.utils.logging_utils import configure_logger
from miles.utils.workers.serving.utils import override_argv, split_worker_argv
from miles.utils.workers.types import DeploySelector

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeploymentWiring:
    launch_worker_manager: Callable[[], Any]
    describe_reachability: Callable[[], Awaitable[str]]


def main() -> None:
    own_argv, worker_argv = split_worker_argv(sys.argv[1:])
    own_args = _parse_own_args(own_argv)
    with override_argv(worker_argv):
        args = parse_args()
    asyncio.run(_serve_deployed_workers(args, own_args=own_args, worker_argv=worker_argv))


async def _serve_deployed_workers(args, *, own_args: argparse.Namespace, worker_argv: list[str]) -> None:
    configure_logger(args, source=SimpleProcessIdentity(component="main"))
    selector = DeploySelector.of(args)
    assert not selector.deploys_orchestration_script(), (
        f"this entrypoint installs the workers of a deployment that carries no orchestration script, and "
        f"--deploy-component {selector.value} carries one: launch it through its driver script instead"
    )

    wiring: DeploymentWiring = load_function(own_args.wiring)(worker_argv)
    specs = load_function(own_args.specs)(worker_argv)

    _worker_manager = wiring.launch_worker_manager()
    logger.info(
        f"Deployed the {selector.value} workers of this run: "
        f"{[spec.name for spec in specs]}. "
        f"{await wiring.describe_reachability()}"
    )
    logger.info(
        "This deployment carries no orchestration script, so it has no training to finish and stays up until it is "
        "torn down"
    )

    await asyncio.Event().wait()


def _parse_own_args(own_argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", required=True)
    parser.add_argument("--wiring", required=True)
    return parser.parse_args(own_argv)


if __name__ == "__main__":
    main()
