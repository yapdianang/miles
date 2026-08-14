from pathlib import Path

import pytest
from tests.e2e.ft.conftest_ft import fault_injection as fi
from tests.e2e.ft.conftest_ft.scenario_random_crash import assert_healing

from miles.utils.audit_utils.event_logger.logger import EventLogger
from miles.utils.audit_utils.event_logger.models import CellReconfigureEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.external_utils import command_utils
from miles.utils.workers.types import ClusterBackend

_ROLLOUT_CELL_NAME = "rollout-engine-0"
_ACTOR_CELL_NAME = "actor-0"


def _injector(*, cell_type: str | None) -> fi.FaultInjectorHandle:
    """cell_type None is a mixed soak, which crashes both kinds of cell."""
    config = command_utils.ExecuteTrainConfig(cluster_backend=ClusterBackend.RAY)
    cell_types = [fi.ACTOR_CELL_TYPE, fi.ROLLOUT_CELL_TYPE] if cell_type is None else [cell_type]
    return fi.FaultInjectorHandle(
        base_url="http://control",
        seed=0,
        mean_interval_seconds_of_cell_type={one_type: 1e9 for one_type in cell_types},
        cell_fault_forms=fi.create_cell_fault_forms(base_url="http://control", config=config),
    )


def _actor_cell(name: str = _ACTOR_CELL_NAME) -> dict:
    return {
        "metadata": {"name": name, "labels": {"miles.io/cell-type": "actor"}},
        "status": {"phase": "Running", "conditions": [{"type": "Healthy", "status": "True"}]},
    }


def _note_actor_injections(injector: fi.FaultInjectorHandle, count: int, *, name: str = _ACTOR_CELL_NAME) -> None:
    witness = injector.recovery_witness
    for _ in range(count):
        witness.observe([_actor_cell(name)])
        witness.note_injected(name)


def _rollout_cell(state: fi.ObservedCellState) -> dict:
    phase = "Pending" if state is fi.ObservedCellState.PENDING else "Running"
    conditions = (
        []
        if phase == "Pending"
        else [
            {"type": "Healthy", "status": "True"},
            {"type": "Serving", "status": "True" if state is fi.ObservedCellState.SERVING else "False"},
        ]
    )
    return {
        "metadata": {"name": _ROLLOUT_CELL_NAME, "labels": {"miles.io/cell-type": "rollout"}},
        "status": {"phase": phase, "conditions": conditions},
    }


def _write_shrink_only_events(event_dir: Path) -> None:
    event_logger = EventLogger(log_dir=event_dir, source=SimpleProcessIdentity(component="main"))
    event_logger.log(
        CellReconfigureEvent,
        dict(rollout_id=2, quorum_id=1, src_cell_index=None, healed_cell_indices=[], alive_cell_indices_after=[0]),
        print_log=False,
    )
    event_logger.close()


def _write_healing_events(event_dir: Path, healed_cell_indices_per_event: list[list[int]]) -> None:
    event_logger = EventLogger(log_dir=event_dir, source=SimpleProcessIdentity(component="main"))
    for index, healed_cell_indices in enumerate(healed_cell_indices_per_event):
        event_logger.log(
            CellReconfigureEvent,
            dict(
                rollout_id=index + 2,
                quorum_id=index + 1,
                src_cell_index=0,
                healed_cell_indices=healed_cell_indices,
                alive_cell_indices_after=[0, 1],
            ),
            print_log=False,
        )
    event_logger.close()


class TestAssertHealing:
    def test_trainer_soak_rejects_missing_reconfigure_witness(self, tmp_path: Path) -> None:
        """A trainer-only soak whose accepted injections produced no healing event must fail."""
        _write_shrink_only_events(tmp_path / "events")
        injector = _injector(cell_type="actor")
        _note_actor_injections(injector, 3)

        with pytest.raises(AssertionError, match="Healing witness failed"):
            assert_healing(
                ("train",),
                injector=injector,
                event_dir=tmp_path / "events",
                context="a unit test of the soak assertions",
            )

    def test_trainer_soak_ignores_rollout_injections_when_counting_its_own(self, tmp_path: Path) -> None:
        """A mixed soak's engine crashes say nothing about trainer healing, so they must not be counted."""
        _write_shrink_only_events(tmp_path / "events")
        injector = _injector(cell_type=None)
        witness = injector.recovery_witness
        witness.observe([_rollout_cell(fi.ObservedCellState.SERVING)])
        for _ in range(3):
            witness.note_injected(_ROLLOUT_CELL_NAME)

        with pytest.raises(AssertionError, match="Soak proved too little"):
            assert_healing(
                (
                    "train",
                    "rollout",
                ),
                injector=injector,
                event_dir=tmp_path / "events",
                context="a unit test of the soak assertions",
            )

    def test_rollout_soak_rejects_unfinished_engine_recovery(self, tmp_path: Path) -> None:
        """A rollout-only soak that ends with an accepted injection still relaunching must fail."""
        injector = _injector(cell_type="rollout")
        witness = injector.recovery_witness
        witness.observe([_rollout_cell(fi.ObservedCellState.SERVING)])
        witness.note_injected(_ROLLOUT_CELL_NAME)
        witness.observe([_rollout_cell(fi.ObservedCellState.PENDING)])
        witness.observe([_rollout_cell(fi.ObservedCellState.SERVING)])
        witness.note_injected(_ROLLOUT_CELL_NAME)
        witness.observe([_rollout_cell(fi.ObservedCellState.PENDING)])

        with pytest.raises(AssertionError, match="Rollout recovery witness failed"):
            assert_healing(
                ("rollout",),
                injector=injector,
                event_dir=tmp_path / "events",
                context="a unit test of the soak assertions",
            )


class TestTrainerHealingPairing:
    def test_a_final_injection_that_never_healed_fails_even_though_the_floor_is_cleared(self, tmp_path: Path) -> None:
        """Regression: 3 crashes with 2 heals used to pass, leaving the run permanently degraded."""
        _write_healing_events(tmp_path / "events", [[0], [0]])
        injector = _injector(cell_type="actor")
        _note_actor_injections(injector, 3)

        with pytest.raises(AssertionError, match="Trainer recovery witness failed"):
            assert_healing(
                ("train",),
                injector=injector,
                event_dir=tmp_path / "events",
                context="a unit test of the soak assertions",
            )

    def test_two_cells_healed_by_one_reconfigure_event_count_as_two_healings(self, tmp_path: Path) -> None:
        """One reconfigure can readmit several cells, so counting events would under-count the healing."""
        _write_healing_events(tmp_path / "events", [[0, 1]])
        injector = _injector(cell_type="actor")
        _note_actor_injections(injector, 1, name="actor-0")
        _note_actor_injections(injector, 1, name="actor-1")

        assert_healing(
            ("train",),
            injector=injector,
            event_dir=tmp_path / "events",
            context="a unit test of the soak assertions",
        )

    def test_healing_a_cell_that_was_never_injected_does_not_pay_another_cells_debt(self, tmp_path: Path) -> None:
        """Counting healings without pairing them by cell index would call this a healthy soak."""
        _write_healing_events(tmp_path / "events", [[0], [0]])
        injector = _injector(cell_type="actor")
        _note_actor_injections(injector, 2, name="actor-1")

        with pytest.raises(AssertionError, match="Trainer recovery witness failed"):
            assert_healing(
                ("train",),
                injector=injector,
                event_dir=tmp_path / "events",
                context="a unit test of the soak assertions",
            )

    def test_every_injection_paired_with_a_healing_of_the_same_cell_passes(self, tmp_path: Path) -> None:
        """The assertion must stay invisible on the path a healthy soak actually takes."""
        _write_healing_events(tmp_path / "events", [[0], [1]])
        injector = _injector(cell_type="actor")
        _note_actor_injections(injector, 1, name="actor-0")
        _note_actor_injections(injector, 1, name="actor-1")

        assert_healing(
            ("train",),
            injector=injector,
            event_dir=tmp_path / "events",
            context="a unit test of the soak assertions",
        )
