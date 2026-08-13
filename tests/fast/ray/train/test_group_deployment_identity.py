from __future__ import annotations

from argparse import Namespace
from unittest.mock import MagicMock

import pytest

from miles.ray.train.group import TrainerController
from miles.utils.workers.cell_operations.base import BaseCellOperations
from miles.utils.workers.types import DeploymentIdentity
from miles.utils.workers.worker_provider.base import BaseWorkerProvider

pytestmark = pytest.mark.asyncio


def _args(**overrides) -> Namespace:
    defaults = dict(
        run_uuid="0123456789abcdef",
        deploy_component="trainer",
        trainer_controller_addrs=None,
        api_server_port=0,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _controller(launch_args: Namespace) -> TrainerController:
    return TrainerController(
        launch_args=launch_args,
        cell_provider=MagicMock(spec=BaseWorkerProvider),
        cell_operations=MagicMock(spec=BaseCellOperations),
        inference_controller=None,
        trainer_id="actor",
        role="actor",
        with_ref=False,
    )


class TestDeploymentIdentity:
    async def test_the_identity_names_the_launch_this_controller_was_started_by(self):
        """A trainer answers for the deployment that launched it, not for the arguments a script hands it later."""
        controller = _controller(_args())

        assert await controller.get_deployment_identity() == DeploymentIdentity(
            run_uuid="0123456789abcdef", deploy_component="trainer"
        )
