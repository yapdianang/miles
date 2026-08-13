from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from enum import Enum

logger = logging.getLogger(__name__)


class InitState(Enum):
    NOT_STARTED = "not started"
    INITIALIZING = "initializing"
    COMPLETE = "complete"
    FAILED = "failed"


class InitOnce:
    def __init__(self, *, component: str) -> None:
        self._component = component
        self._state = InitState.NOT_STARTED

    @property
    def state(self) -> InitState:
        return self._state

    @property
    def is_initialized(self) -> bool:
        return self._state is InitState.COMPLETE

    @contextlib.contextmanager
    def guard(self) -> Iterator[None]:
        self.enter()
        try:
            yield
        except BaseException:
            self._state = InitState.FAILED
            logger.error(f"Initializing {self._component} failed, so nothing may drive it", exc_info=True)
            raise
        self.complete()

    def enter(self) -> None:
        assert self._state is InitState.NOT_STARTED, (
            f"{self._component} is {self._state.value} in this process, and initializing it again would "
            f"re-initialize a live system behind the back of whoever is already driving it; a restarted "
            f"orchestration script has to resume a complete component instead of initializing it again, and a "
            f"component whose init failed or never finished has to be replaced before anything drives it"
        )
        self._state = InitState.INITIALIZING

    def complete(self) -> None:
        assert (
            self._state is InitState.INITIALIZING
        ), f"{self._component} is {self._state.value}, so it is not being initialized right now"
        self._state = InitState.COMPLETE
        logger.info(f"{self._component} is now initialized")

    def assert_initialized(self) -> None:
        assert self.is_initialized, f"{self._component} is {self._state.value}, not initialized"
