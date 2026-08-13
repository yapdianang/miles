import dataclasses
import random
import threading
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
from tests.e2e.ft.conftest_ft import fault_injection as fi
from tests.e2e.ft.conftest_ft.fault_injection import RecoveryGate, cell_is_alive
from tests.e2e.ft.conftest_ft.modes import MODES, FTTestMode

from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.helm_backend.naming import RunNames
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.types import ClusterBackend

_NAMESPACE = "miles-e2e"
_RUN_ID = "abc123"


def _config(backend: ClusterBackend, *, namespace: str = _NAMESPACE) -> command_utils.ExecuteTrainConfig:
    return command_utils.ExecuteTrainConfig(cluster_backend=backend, namespace=namespace, run_id=_RUN_ID)


def _api_server_fault_forms() -> fi.CellFaultForms:
    return fi.create_cell_fault_forms(base_url="http://control", config=_config(ClusterBackend.RAY))


class _StubFaultForm(fi.BaseFaultForm):
    def __init__(self, form_name: str, on_inject: Callable[[dict, random.Random], None]) -> None:
        self._name = form_name
        self._on_inject = on_inject

    @property
    def name(self) -> str:
        return self._name

    def inject(self, cell: dict, rng: random.Random) -> None:
        self._on_inject(cell, rng)


def _fixed_fault_forms(forms: list[fi.BaseFaultForm]) -> fi.CellFaultForms:
    return {fi.ACTOR_CELL_TYPE: forms, fi.ROLLOUT_CELL_TYPE: forms}


def _cell(name: str, *, healthy: bool, cell_type: str = "actor", phase: str = "Running", serving: bool = True) -> dict:
    status = "True" if healthy else "False"
    conditions = [{"type": "Healthy", "status": status}]
    if cell_type == "rollout":
        conditions.append({"type": "Serving", "status": "True" if serving else "False"})
    return {
        "metadata": {"name": name, "labels": {"miles.io/cell-type": cell_type}},
        "status": {"phase": phase, "conditions": conditions},
    }


def _intervals(cell_types: tuple[str, ...], mean_interval_seconds: float) -> dict[str, float]:
    return {cell_type: mean_interval_seconds for cell_type in cell_types}


def _by_name(*cells: dict) -> dict[str, dict]:
    return {c["metadata"]["name"]: c for c in cells}


def _names(cells: list[dict]) -> set[str]:
    return {c["metadata"]["name"] for c in cells}


def test_cell_is_alive_true_only_when_healthy_condition_is_true() -> None:
    """cell_is_alive reflects the Healthy condition status."""
    assert cell_is_alive(_cell("c", healthy=True))
    assert not cell_is_alive(_cell("c", healthy=False))


def test_cell_is_alive_false_when_no_healthy_condition_present() -> None:
    """A cell with no Healthy condition is not considered alive."""
    assert not cell_is_alive({"metadata": {"name": "c"}, "status": {"conditions": []}})


def test_fresh_gate_counts_every_healthy_cell_as_alive() -> None:
    """With no outstanding injection the live set is just the healthy cells."""
    gate = RecoveryGate()
    cells = [_cell("c0", healthy=True), _cell("c1", healthy=False)]
    gate.observe(_by_name(*cells))
    assert _names(gate.genuinely_alive(cells)) == {"c0"}


def test_injected_cell_is_excluded_while_its_crash_is_still_undetected() -> None:
    """The api server's stale 'still healthy' view must not count a just-killed cell."""
    gate = RecoveryGate()
    cells = [_cell("c0", healthy=True), _cell("c1", healthy=True)]
    gate.note_injected("c0")
    gate.observe(_by_name(*cells))  # c0 really dead but still reported Healthy
    assert _names(gate.genuinely_alive(cells)) == {"c1"}


def test_injected_cell_counts_again_only_after_a_full_down_then_up_cycle() -> None:
    """A cell must be seen unhealthy and then healthy again before it rejoins the live set."""
    gate = RecoveryGate()
    healthy = [_cell("c0", healthy=True), _cell("c1", healthy=True)]
    down = [_cell("c0", healthy=False), _cell("c1", healthy=True)]
    gate.note_injected("c0")

    gate.observe(_by_name(*healthy))  # stale-alive
    assert _names(gate.genuinely_alive(healthy)) == {"c1"}
    gate.observe(_by_name(*down))  # detected down
    assert _names(gate.genuinely_alive(down)) == {"c1"}
    gate.observe(_by_name(*healthy))  # healed
    assert _names(gate.genuinely_alive(healthy)) == {"c0", "c1"}


def test_vanished_cell_counts_as_the_down_half_of_the_cycle() -> None:
    """A cell missing from the snapshot is treated as observed-down, then recovers when back."""
    gate = RecoveryGate()
    gate.note_injected("c0")
    gate.observe(_by_name(_cell("c1", healthy=True)))  # c0 absent == down
    healthy = [_cell("c0", healthy=True), _cell("c1", healthy=True)]
    gate.observe(_by_name(*healthy))
    assert _names(gate.genuinely_alive(healthy)) == {"c0", "c1"}


def test_allows_overlapping_crashes_while_one_cell_stays_alive() -> None:
    """The gate guards >=1 live replica, not 1-crash-at-a-time: with 3 cells two may be down."""
    gate = RecoveryGate()
    cells = [_cell("c0", healthy=True), _cell("c1", healthy=True), _cell("c2", healthy=True)]

    gate.note_injected("c0")
    gate.observe(_by_name(*cells))
    assert _names(gate.genuinely_alive(cells)) == {"c1", "c2"}  # 2 still alive -> a 2nd inject is allowed

    gate.note_injected("c1")
    gate.observe(_by_name(*cells))
    assert _names(gate.genuinely_alive(cells)) == {"c2"}  # now only 1 -> loop would skip


def _mock_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=payload)
    return resp


def test_loop_never_kills_the_last_live_cell_under_stale_liveness() -> None:
    """Regression: a perpetually-stale 'all healthy' view yields at most one kill (2 cells)."""
    cell_names = ["actor-0", "actor-1"]
    injected: list[str] = []
    stop_event = threading.Event()
    polls = {"n": 0}

    def fake_get(url: str, timeout: float) -> MagicMock:
        polls["n"] += 1
        if polls["n"] >= 6:
            stop_event.set()
        # Worst case: the injected cell's death is never detected (every cell always Healthy).
        return _mock_response({"items": [_cell(n, healthy=True) for n in cell_names]})

    def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
        injected.append(url.rsplit("/cells/", 1)[1].split("/")[0])
        return _mock_response({})

    with patch.object(fi, "requests") as mock_requests:
        mock_requests.get.side_effect = fake_get
        mock_requests.post.side_effect = fake_post
        fi.run_fault_injection_loop(
            base_url="http://control",
            seed=0,
            mean_interval_seconds_of_cell_type=_intervals(("actor", "rollout"), 1e-6),
            stop_event=stop_event,
            on_successful_injection=lambda: None,
            recovery_witness=fi.RecoveryWitness(),
            cell_fault_forms=_api_server_fault_forms(),
            poll_interval_seconds=1e-6,
        )

    assert len(injected) == 1, f"expected at most one injection, got {injected}"


def test_loop_injects_again_after_an_injected_cell_recovers() -> None:
    """Polling tracks a cell's down->up cycle between injections, so a second injection follows."""
    cell_names = ["actor-0", "actor-1"]
    injected: list[str] = []
    stop_event = threading.Event()
    down = {"name": None, "polls_left": 0}
    polls = {"n": 0}

    def fake_get(url: str, timeout: float) -> MagicMock:
        polls["n"] += 1
        if len(injected) >= 2 or polls["n"] >= 100:
            stop_event.set()
        items = [_cell(n, healthy=not (down["name"] == n and down["polls_left"] > 0)) for n in cell_names]
        if down["polls_left"] > 0:
            down["polls_left"] -= 1
        return _mock_response({"items": items})

    def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
        name = url.rsplit("/cells/", 1)[1].split("/")[0]
        injected.append(name)
        down["name"], down["polls_left"] = name, 3  # crashed cell reads unhealthy for a few polls, then heals
        return _mock_response({})

    with patch.object(fi, "requests") as mock_requests:
        mock_requests.get.side_effect = fake_get
        mock_requests.post.side_effect = fake_post
        fi.run_fault_injection_loop(
            base_url="http://control",
            seed=0,
            mean_interval_seconds_of_cell_type=_intervals(("actor", "rollout"), 1e-6),
            stop_event=stop_event,
            on_successful_injection=lambda: None,
            recovery_witness=fi.RecoveryWitness(),
            cell_fault_forms=_api_server_fault_forms(),
            poll_interval_seconds=1e-6,
        )

    assert len(injected) >= 2, f"expected a second injection after recovery, got {injected}"


def _typed_cell(name: str, cell_type: str, *, healthy: bool = True, serving: bool = True) -> dict:
    return _cell(name, healthy=healthy, cell_type=cell_type, serving=serving)


def _run_typed_injection_loop(cells: list[dict], *, cell_types: tuple[str, ...]) -> list[str]:
    injected: list[str] = []
    stop_event = threading.Event()
    polls = {"n": 0}

    def fake_get(url: str, timeout: float) -> MagicMock:
        polls["n"] += 1
        if polls["n"] >= 6:
            stop_event.set()
        return _mock_response({"items": cells})

    def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
        injected.append(url.rsplit("/cells/", 1)[1].split("/")[0])
        return _mock_response({})

    with patch.object(fi, "requests") as mock_requests:
        mock_requests.get.side_effect = fake_get
        mock_requests.post.side_effect = fake_post
        fi.run_fault_injection_loop(
            base_url="http://control",
            seed=0,
            mean_interval_seconds_of_cell_type=_intervals(cell_types, 1e-6),
            stop_event=stop_event,
            on_successful_injection=lambda: None,
            recovery_witness=fi.RecoveryWitness(),
            cell_fault_forms=_api_server_fault_forms(),
            poll_interval_seconds=1e-6,
        )

    return injected


def test_a_stop_that_arrives_while_listing_buys_no_further_injection() -> None:
    """Listing takes a whole poll interval, and a fault injected on the way out is one nothing is left polling to see recover."""
    injected: list[str] = []
    stop_event = threading.Event()

    def fake_get(url: str, timeout: float) -> MagicMock:
        stop_event.set()
        return _mock_response({"items": [_typed_cell("actor-0", "actor"), _typed_cell("actor-1", "actor")]})

    def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
        injected.append(url)
        return _mock_response({})

    with patch.object(fi, "requests") as mock_requests:
        mock_requests.get.side_effect = fake_get
        mock_requests.post.side_effect = fake_post
        fi.run_fault_injection_loop(
            base_url="http://control",
            seed=0,
            mean_interval_seconds_of_cell_type=_intervals(("actor",), 1e-9),
            stop_event=stop_event,
            on_successful_injection=lambda: None,
            recovery_witness=fi.RecoveryWitness(),
            cell_fault_forms=_api_server_fault_forms(),
            poll_interval_seconds=1e-6,
        )

    assert injected == []


def test_injection_can_be_restricted_to_one_kind_of_cell() -> None:
    """Rollout and trainer cells share one api server, so a run targets one kind at a time."""
    injected = _run_typed_injection_loop(
        [
            _typed_cell("actor-0", "actor"),
            _typed_cell("actor-1", "actor"),
            _typed_cell("rollout-engine-0", "rollout"),
            _typed_cell("rollout-engine-1", "rollout"),
        ],
        cell_types=("rollout",),
    )

    assert injected
    assert all(name.startswith("rollout-") for name in injected), injected


def test_the_live_replica_count_only_considers_the_targeted_kind() -> None:
    """A single rollout cell must not be killed just because trainer cells are also alive."""
    injected = _run_typed_injection_loop(
        [
            _typed_cell("actor-0", "actor"),
            _typed_cell("actor-1", "actor"),
            _typed_cell("rollout-engine-0", "rollout"),
        ],
        cell_types=("rollout",),
    )

    assert injected == []


def test_a_mixed_run_sees_every_targeted_kind() -> None:
    """A mixed-ft soak schedules both kinds, and must be able to crash either one."""
    injected = _run_typed_injection_loop(
        [
            _typed_cell("actor-0", "actor"),
            _typed_cell("actor-1", "actor"),
            _typed_cell("rollout-engine-0", "rollout"),
            _typed_cell("rollout-engine-1", "rollout"),
        ],
        cell_types=("actor", "rollout"),
    )

    assert injected


def test_a_mixed_run_still_keeps_one_replica_of_each_kind() -> None:
    """Counting kinds together would let the trainer cells license killing the last engine."""
    injected = _run_typed_injection_loop(
        [
            _typed_cell("actor-0", "actor"),
            _typed_cell("actor-1", "actor"),
            _typed_cell("rollout-engine-0", "rollout"),
        ],
        cell_types=("actor", "rollout"),
    )

    assert all(name.startswith("actor-") for name in injected), injected


_SERVING = fi.ObservedCellState.SERVING
_RUNNING_NOT_SERVING = fi.ObservedCellState.RUNNING_NOT_SERVING
_PENDING = fi.ObservedCellState.PENDING
_SUSPENDED = fi.ObservedCellState.SUSPENDED


def _staged(name: str, state: fi.ObservedCellState, *, cell_type: str = "rollout") -> dict:
    phase = {
        _SUSPENDED: "Suspended",
        _PENDING: "Pending",
        _RUNNING_NOT_SERVING: "Running",
        _SERVING: "Running",
    }[state]
    conditions: list[dict] = (
        [
            {"type": "Healthy", "status": "True"},
            {"type": "Serving", "status": "True" if state is _SERVING else "False"},
        ]
        if phase == "Running"
        else []
    )
    return {
        "metadata": {"name": name, "labels": {"miles.io/cell-type": cell_type}},
        "status": {"phase": phase, "conditions": conditions},
    }


def _witness_of(
    states: list[fi.ObservedCellState], *, inject_before: dict[int, int] | None = None
) -> fi.RecoveryWitness:
    witness = fi.RecoveryWitness()
    for index, state in enumerate(states):
        for _ in range((inject_before or {}).get(index, 0)):
            witness.note_injected("rollout-engine-0")
        witness.observe([_staged("rollout-engine-0", state)])
    return witness


def test_a_running_cell_that_is_not_in_the_router_is_not_serving() -> None:
    """The api server renders PendingWeights and Serving alike, so the Serving condition must split them."""
    assert fi.compute_observed_cell_state(_staged("c", _RUNNING_NOT_SERVING)) is _RUNNING_NOT_SERVING
    assert fi.compute_observed_cell_state(_staged("c", _SERVING)) is _SERVING


def test_observed_states_record_only_transitions() -> None:
    """Polling runs for the life of the training run, so repeats must not accumulate."""
    witness = _witness_of([_SERVING, _SERVING, _SUSPENDED, _SUSPENDED, _SERVING])

    assert witness.states_of_cell_name == {"rollout-engine-0": [_SERVING, _SUSPENDED, _SERVING]}


def test_a_colocated_cell_recovers_through_a_gated_relaunch() -> None:
    """A relaunched engine stays gated until the next weight update window puts it back in the router."""
    witness = _witness_of([_SERVING, _SUSPENDED, _PENDING, _RUNNING_NOT_SERVING, _SERVING], inject_before={1: 1})

    assert witness.num_completed_recoveries(cell_type="rollout") == 1
    assert witness.cells_with_unfinished_recovery(cell_type="rollout") == {}


def test_a_missed_suspended_sample_still_counts_as_a_recovery() -> None:
    """Suspension lasts only the resume delay, so a 2s poll can miss it entirely."""
    witness = _witness_of([_SERVING, _PENDING, _SERVING], inject_before={1: 1})

    assert witness.num_completed_recoveries(cell_type="rollout") == 1


def test_a_replacement_that_never_reaches_the_router_is_not_a_recovery() -> None:
    """Regression: a relaunched engine stuck at PendingWeights also reads Running, and must not pass."""
    witness = _witness_of([_SERVING, _PENDING, _RUNNING_NOT_SERVING], inject_before={1: 1})

    assert witness.num_completed_recoveries(cell_type="rollout") == 0
    assert witness.cells_with_unfinished_recovery(cell_type="rollout") == {"rollout-engine-0": 1}


def test_a_cell_that_was_never_injected_is_not_a_recovery_witness() -> None:
    """Otherwise a run that injected nothing would still pass the gated assertion."""
    witness = _witness_of([_PENDING, _SERVING])

    assert witness.num_injections(cell_type="rollout") == 0
    assert witness.num_completed_recoveries(cell_type="rollout") == 0


def test_skipping_the_relaunch_phase_is_not_a_recovery() -> None:
    """A cell that never left Running was never replaced, so it witnesses no healing."""
    witness = _witness_of([_SERVING, _RUNNING_NOT_SERVING, _SERVING], inject_before={1: 1})

    assert witness.num_completed_recoveries(cell_type="rollout") == 0


def test_each_accepted_injection_needs_its_own_completed_recovery() -> None:
    """Regression: a second crash accepted just before the run ends must not ride on the first heal."""
    witness = _witness_of([_SERVING, _PENDING, _SERVING, _SERVING], inject_before={1: 1, 3: 1})

    assert witness.num_injections(cell_type="rollout") == 2
    assert witness.num_completed_recoveries(cell_type="rollout") == 1
    assert witness.cells_with_unfinished_recovery(cell_type="rollout") == {"rollout-engine-0": 1}


def test_recoveries_of_another_cell_kind_do_not_count() -> None:
    """A mixed soak injects both kinds, and the rollout witness must only see rollout cells."""
    witness = fi.RecoveryWitness()
    witness.observe([_staged("actor-0", _SERVING, cell_type="actor")])
    witness.note_injected("actor-0")
    for state in [_PENDING, _SERVING]:
        witness.observe([_staged("actor-0", state, cell_type="actor")])

    assert witness.num_completed_recoveries(cell_type="rollout") == 0
    assert witness.num_completed_recoveries(cell_type="actor") == 1


def _mode(*ft_components: str) -> FTTestMode:
    return dataclasses.replace(next(iter(MODES.values())), ft_components=tuple(ft_components))


def _mean_intervals(*ft_components: str) -> dict[str, float]:
    return fi.compute_mean_interval_seconds_of_cell_type(
        tuple(ft_components), trainer_crash_interval_seconds=120.0, rollout_crash_interval_seconds=240.0
    )


def test_a_trainer_only_soak_schedules_actor_injections_only() -> None:
    """It must not crash engines that its assertions say nothing about."""
    assert _mean_intervals("train") == {"actor": 120.0}


def test_a_rollout_only_soak_schedules_rollout_injections_only() -> None:
    """Crashing trainer cells here would exercise a component this mode did not enable ft on."""
    assert _mean_intervals("rollout") == {"rollout": 240.0}


def test_a_mixed_soak_keeps_each_kind_on_the_cadence_it_would_have_alone() -> None:
    """Adding rollout to a soak must not dilute the trainer crash rate it was calibrated at."""
    assert _mean_intervals("train", "rollout") == {"actor": 120.0, "rollout": 240.0}


def test_a_kind_the_mode_does_not_enable_ft_on_gets_no_schedule_at_all() -> None:
    """An entry in the map is what makes the loop consider a kind, so a stray one crashes an unwatched component."""
    assert "rollout" not in _mean_intervals("train")


def test_stop_and_join_takes_one_last_snapshot_before_the_witness_is_read() -> None:
    """Regression: a recovery completing after the final poll must not be lost to a race."""
    handle = fi.FaultInjectorHandle(
        base_url="http://control",
        seed=0,
        mean_interval_seconds_of_cell_type=_intervals(("rollout",), 1e9),
        cell_fault_forms=_api_server_fault_forms(),
    )

    with patch.object(fi, "requests") as mock_requests:
        mock_requests.get.side_effect = lambda url, timeout: _mock_response(
            {"items": [_staged("rollout-engine-0", _SERVING)]}
        )
        handle.start()
        handle.stop_and_join(timeout_seconds=5)

    assert handle.recovery_witness.states_of_cell_name == {"rollout-engine-0": [_SERVING]}


class TestRecoveryWitnessPairing:
    def test_another_cells_relaunch_cannot_complete_the_injected_cells_recovery(self) -> None:
        """A sibling engine's relaunch-and-serve cycle must not discharge the injected cell's debt."""
        witness = fi.RecoveryWitness()
        witness.observe([_staged("rollout-engine-0", _SERVING), _staged("rollout-engine-1", _SERVING)])
        witness.note_injected("rollout-engine-0")
        for sibling_state in [_PENDING, _SERVING]:
            witness.observe([_staged("rollout-engine-0", _SERVING), _staged("rollout-engine-1", sibling_state)])

        assert witness.num_injections(cell_type="rollout") == 1
        assert witness.num_completed_recoveries(cell_type="rollout") == 0
        assert witness.cells_with_unfinished_recovery(cell_type="rollout") == {"rollout-engine-0": 1}

    def test_relaunch_observed_before_injection_does_not_count_as_recovery(self) -> None:
        """The cycle must be ordered injection then relaunch then serving, not merely present in the history."""
        witness = _witness_of([_SERVING, _PENDING, _SERVING], inject_before={2: 1})

        assert witness.num_injections(cell_type="rollout") == 1
        assert witness.num_completed_recoveries(cell_type="rollout") == 0
        assert witness.cells_with_unfinished_recovery(cell_type="rollout") == {"rollout-engine-0": 1}


class TestFaultInjectionLoopErrorHandling:
    def test_list_cells_failure_is_retried_without_recording_recovery(self) -> None:
        """A transient outage after injection must preserve pending recovery debt and retry."""
        cells = [_staged("rollout-engine-0", _SERVING), _staged("rollout-engine-1", _SERVING)]
        witness = fi.RecoveryWitness()
        injected: list[str] = []
        debt_around_failure: list[dict[str, int]] = []
        stop_event = threading.Event()
        polls = {"n": 0}

        def fake_get(url: str, timeout: float) -> MagicMock:
            polls["n"] += 1
            if polls["n"] in {2, 3}:
                debt_around_failure.append(witness.cells_with_unfinished_recovery(cell_type="rollout"))
            if polls["n"] == 2:
                raise RuntimeError("api server unreachable")
            if polls["n"] >= 6:
                stop_event.set()
            return _mock_response({"items": cells})

        def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
            injected.append(url.rsplit("/cells/", 1)[1].split("/")[0])
            return _mock_response({})

        with patch.object(fi, "requests") as mock_requests:
            mock_requests.get.side_effect = fake_get
            mock_requests.post.side_effect = fake_post
            fi.run_fault_injection_loop(
                base_url="http://control",
                seed=0,
                mean_interval_seconds_of_cell_type=_intervals(("actor", "rollout"), 1e-12),
                stop_event=stop_event,
                on_successful_injection=lambda: None,
                recovery_witness=witness,
                cell_fault_forms=_api_server_fault_forms(),
                poll_interval_seconds=1e-6,
            )

        assert len(injected) == 1, injected
        expected_debt: dict[str, int] = {injected[0]: 1}
        assert debt_around_failure == [expected_debt, expected_debt]
        assert witness.states_of_cell_name == {"rollout-engine-0": [_SERVING], "rollout-engine-1": [_SERVING]}
        assert witness.num_injections(cell_type="rollout") == 1
        assert witness.num_completed_recoveries(cell_type="rollout") == 0
        assert witness.cells_with_unfinished_recovery(cell_type="rollout") == expected_debt

    def test_failed_fault_post_is_not_counted_and_is_retried(self) -> None:
        """A rejected inject-fault call must leave the soak free to try again, and must not inflate the tally."""
        cells = [_staged("rollout-engine-0", _SERVING), _staged("rollout-engine-1", _SERVING)]
        witness = fi.RecoveryWitness()
        attempts: list[str] = []
        successes = {"n": 0}
        stop_event = threading.Event()
        polls = {"n": 0}

        def fake_get(url: str, timeout: float) -> MagicMock:
            polls["n"] += 1
            if polls["n"] >= 5:
                stop_event.set()
            return _mock_response({"items": cells})

        def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
            attempts.append(url.rsplit("/cells/", 1)[1].split("/")[0])
            if len(attempts) == 1:
                raise RuntimeError("inject-fault refused")
            return _mock_response({})

        def note_success() -> None:
            successes["n"] += 1

        with patch.object(fi, "requests") as mock_requests:
            mock_requests.get.side_effect = fake_get
            mock_requests.post.side_effect = fake_post
            fi.run_fault_injection_loop(
                base_url="http://control",
                seed=0,
                mean_interval_seconds_of_cell_type=_intervals(("actor", "rollout"), 1e-6),
                stop_event=stop_event,
                on_successful_injection=note_success,
                recovery_witness=witness,
                cell_fault_forms=_api_server_fault_forms(),
                poll_interval_seconds=1e-6,
            )

        assert len(attempts) == 2, attempts
        assert successes["n"] == 1
        assert witness.num_injections(cell_type="rollout") == 1


class TestMixedInjectionSelection:
    def test_mixed_run_injects_rollout_when_only_rollout_has_a_spare(self) -> None:
        """The mirror of the trainer case: mixed selection must not be hard-coded to actor cells."""
        injected = _run_typed_injection_loop(
            [
                _typed_cell("actor-0", "actor"),
                _typed_cell("rollout-engine-0", "rollout"),
                _typed_cell("rollout-engine-1", "rollout"),
            ],
            cell_types=("actor", "rollout"),
        )

        assert injected
        assert all(name.startswith("rollout-engine-") for name in injected), injected


class TestFaultFormsOfBackendAndCellType:
    def test_ray_draws_the_in_process_kills_for_a_trainer_cell(self) -> None:
        """Ray owns no pods, so the only fault it can be asked for is a kill inside the worker."""
        forms = _api_server_fault_forms()["actor"]

        assert [form.name for form in forms] == [f"inject_fault:{one.value}" for one in fi.FAILURE_MODES]

    def test_ray_draws_a_sigkill_only_for_a_rollout_cell(self) -> None:
        """The engine is a subprocess, and exit, segfault and deadlock are faults only its own code can commit."""
        forms = _api_server_fault_forms()["rollout"]

        assert [form.name for form in forms] == [f"inject_fault:{FailureMode.SIGKILL.value}"]

    def test_kubernetes_draws_the_kills_plus_pod_deletion_for_a_trainer_cell(self) -> None:
        """Trainer workers are served over rpc on k8s, so pod deletion joins the kills instead of replacing them."""
        compute = fi.create_cell_fault_forms(base_url="http://control", config=_config(ClusterBackend.KUBERNETES))

        forms = compute(_typed_cell("actor-0", "actor"))

        assert [form.name for form in forms] == [
            *(f"inject_fault:{one.value}" for one in fi.FAILURE_MODES),
            fi.DELETE_POD_FORM_NAME,
        ]

    def test_kubernetes_draws_pod_deletion_only_for_a_rollout_cell(self) -> None:
        """A k8s engine pod runs sglang as its entrypoint with no rpc server, so a kill would blow up at runtime."""
        compute = fi.create_cell_fault_forms(base_url="http://control", config=_config(ClusterBackend.KUBERNETES))

        forms = compute(_typed_cell("rollout-engine-0", "rollout"))

        assert [form.name for form in forms] == [fi.DELETE_POD_FORM_NAME]

    def test_every_kill_is_its_own_form_so_the_draw_stays_uniform(self) -> None:
        """Folding the kills into one form would make pod deletion half of every trainer injection."""
        compute = fi.create_cell_fault_forms(base_url="http://control", config=_config(ClusterBackend.KUBERNETES))

        assert len(compute(_typed_cell("actor-0", "actor"))) == len(fi.FAILURE_MODES) + 1

    def test_a_kubernetes_run_without_a_namespace_fails_before_the_soak_starts(self) -> None:
        """kubectl would otherwise delete pods in whatever namespace the kubeconfig happens to point at."""
        with pytest.raises(AssertionError, match="needs the namespace"):
            fi.create_cell_fault_forms(
                base_url="http://control", config=_config(ClusterBackend.KUBERNETES, namespace="")
            )

    def test_an_inject_fault_form_posts_the_failure_mode_it_was_built_for(self, monkeypatch) -> None:
        """The form's name must describe what it actually does, or a soak log explains nothing."""
        posted: list[tuple[str, dict]] = []
        requests = MagicMock()
        requests.post.side_effect = lambda url, json, timeout: posted.append((url, json)) or MagicMock()
        monkeypatch.setattr(fi, "requests", requests)

        forms = _api_server_fault_forms()["actor"]
        form = next(one for one in forms if one.name == f"inject_fault:{FailureMode.SEGFAULT.value}")
        form.inject(_typed_cell("actor-0", "actor"), random.Random(0))

        assert posted == [("http://control/api/v1/cells/actor-0/inject-fault", {"mode": "segfault", "sub_index": 0})]

    def test_the_delete_pod_form_never_reaches_the_api_server(self, monkeypatch) -> None:
        """Routing it through inject-fault would test the production path, not an outsider."""
        seen: list[dict] = []
        monkeypatch.setattr(fi, "delete_one_pod_of_cell", lambda **kwargs: seen.append(kwargs) or "pod")
        requests = MagicMock()
        monkeypatch.setattr(fi, "requests", requests)

        compute = fi.create_cell_fault_forms(base_url="http://control", config=_config(ClusterBackend.KUBERNETES))
        cell = _typed_cell("actor-0", "actor")
        next(one for one in compute(cell) if one.name == fi.DELETE_POD_FORM_NAME).inject(cell, random.Random(0))

        assert [one["cell_id"] for one in seen] == ["actor-0"]
        assert [one["release"] for one in seen] == [RunNames.release(run_id=_RUN_ID)]
        assert [one["namespace"] for one in seen] == [_NAMESPACE]
        requests.post.assert_not_called()

    def test_the_loop_injects_through_the_forms_of_the_cell_it_picked(self) -> None:
        """A pod deletion drawn by the loop must reach kubectl, not the api server's inject-fault route."""
        drawn: list[str] = []
        stop_event = threading.Event()
        polls = {"n": 0}

        def fake_get(url: str, timeout: float) -> MagicMock:
            polls["n"] += 1
            if polls["n"] >= 6:
                stop_event.set()
            return _mock_response({"items": [_typed_cell(f"actor-{i}", "actor") for i in range(3)]})

        with patch.object(fi, "requests") as mock_requests:
            mock_requests.get.side_effect = fake_get
            fi.run_fault_injection_loop(
                base_url="http://control",
                seed=0,
                mean_interval_seconds_of_cell_type=_intervals(("actor", "rollout"), 1e-12),
                stop_event=stop_event,
                on_successful_injection=lambda: None,
                recovery_witness=fi.RecoveryWitness(),
                cell_fault_forms=_fixed_fault_forms(
                    [_StubFaultForm(fi.DELETE_POD_FORM_NAME, lambda cell, rng: drawn.append(fi.DELETE_POD_FORM_NAME))]
                ),
                poll_interval_seconds=1e-6,
            )

            assert drawn == [fi.DELETE_POD_FORM_NAME, fi.DELETE_POD_FORM_NAME], drawn
            mock_requests.post.assert_not_called()


class TestRolloutSpareReadiness:
    def test_a_healthy_engine_that_is_not_in_the_router_is_not_a_spare(self) -> None:
        """Regression: a relaunched engine reads Healthy long before it can answer, so it is no replacement."""
        injected = _run_typed_injection_loop(
            [
                _typed_cell("rollout-engine-0", "rollout"),
                _typed_cell("rollout-engine-1", "rollout", serving=False),
            ],
            cell_types=("rollout",),
        )

        assert injected == []

    def test_two_serving_engines_still_leave_one_of_them_injectable(self) -> None:
        """The readiness rule must not block the case it was never meant to block."""
        injected = _run_typed_injection_loop(
            [
                _typed_cell("rollout-engine-0", "rollout"),
                _typed_cell("rollout-engine-1", "rollout"),
            ],
            cell_types=("rollout",),
        )

        assert injected

    def test_a_trainer_cell_is_judged_by_liveness_alone(self) -> None:
        """Trainer cells carry no Serving condition, so requiring one would stop every trainer soak."""
        assert fi._cell_can_serve(_typed_cell("actor-0", "actor"))

    def test_an_engine_that_cannot_serve_yet_is_still_injectable(self) -> None:
        """Crashing an engine mid-relaunch is a real fault window, and only the replica count needs it to serve."""
        injected = _run_typed_injection_loop(
            [
                _typed_cell("rollout-engine-0", "rollout"),
                _typed_cell("rollout-engine-1", "rollout"),
                _typed_cell("rollout-engine-2", "rollout", serving=False),
            ],
            cell_type="rollout",
        )

        assert injected

    def test_an_injector_that_outlives_the_join_fails_instead_of_racing_the_witness(self) -> None:
        """Reading the witness beside a still-running injector would assert on a half-updated tally."""
        released = threading.Event()
        entered = threading.Event()

        def slow_inject(cell: dict, rng: random.Random) -> None:
            entered.set()
            released.wait(timeout=30)

        handle = fi.FaultInjectorHandle(
            base_url="http://control",
            seed=0,
            mean_interval_seconds_of_cell_type=_intervals(("actor", "rollout"), 1e-12),
            cell_fault_forms=_fixed_fault_forms([_StubFaultForm("slow", slow_inject)]),
        )

        with patch.object(fi, "requests") as mock_requests:
            mock_requests.get.side_effect = lambda url, timeout: _mock_response(
                {"items": [_typed_cell(f"actor-{i}", "actor") for i in range(3)]}
            )
            handle.start()
            try:
                assert entered.wait(timeout=30)
                with pytest.raises(AssertionError, match="still mid-injection"):
                    handle.stop_and_join(timeout_seconds=0.2)
            finally:
                released.set()
                handle._thread.join(timeout=30)
