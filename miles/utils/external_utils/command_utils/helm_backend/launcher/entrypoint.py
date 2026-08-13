from __future__ import annotations

import json
import logging
import re
import shlex
from pathlib import Path
from typing import Any

import yaml

from miles.ray.specs.addressing import (
    AddressedWorker,
    compute_addressed_workers,
    compute_registration_addressed_workers,
    describe_registering_deployment,
    format_addressed_workers,
)
from miles.ray.specs.entrypoint import compute_specs
from miles.utils.arguments import parse_args
from miles.utils.external_utils.command_utils.base_backend import (
    CLUSTER_BACKEND_FLAG,
    ExecuteTrainConfig,
    ExecuteTrainRequest,
)
from miles.utils.external_utils.command_utils.common import (
    MOONCAKE_INIT_KWARGS_FLAG,
    ArgvManipulator,
    chart_dir,
    repo_base_dir,
    train_env_vars,
)
from miles.utils.external_utils.command_utils.helm_backend import naming
from miles.utils.external_utils.command_utils.helm_backend.launcher import manifest_diff
from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import CI_LABEL, Helm, Kubectl
from miles.utils.external_utils.command_utils.helm_backend.launcher.hot_restart import plan_hot_restart
from miles.utils.external_utils.command_utils.helm_backend.launcher.launch_record import (
    LaunchRecord,
    installed_launch_record_file,
)
from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import Manifest
from miles.utils.external_utils.command_utils.helm_backend.launcher.observability import farewell, with_observability
from miles.utils.external_utils.command_utils.helm_backend.launcher.observability.diagnosis import collect_diagnosis
from miles.utils.external_utils.command_utils.helm_backend.launcher.observability.pod_facts import pod_phase
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.builder import build_values
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import (
    InfraInfo,
    LaunchPlan,
    MooncakeInfo,
    MooncakePlan,
)
from miles.utils.external_utils.command_utils.helm_backend.naming import RunFiles, RunNames
from miles.utils.external_utils.command_utils.helm_backend.orchestrator.observer import wait_for_run
from miles.utils.external_utils.model_args_utils import shell_safe_model_args
from miles.utils.object_store import ObjectStoreBackend
from miles.utils.run_uuid import derive_run_uuid
from miles.utils.workers.serving.utils import override_argv
from miles.utils.workers.types import ClusterBackend, DeployComponent, DeploySelector
from miles.utils.workers.worker_provider.kubernetes.helm.naming import static_cell_addrs
from miles.utils.workers.worker_spec import BaseWorkerSpec

logger = logging.getLogger(__name__)

_RUN_UUID_FLAG = "--run-uuid"
_ENV_REPORT_FLAG = "--env-report"
_RUN_ID_PATTERN = re.compile(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?")


def execute_train(*, request: ExecuteTrainRequest, config: ExecuteTrainConfig) -> None:
    run_id = config.run_id
    assert _RUN_ID_PATTERN.fullmatch(
        run_id
    ), f"run_id {run_id!r} names every object this run installs, so it has to match {_RUN_ID_PATTERN.pattern}"
    component_values = {component.value for component in DeployComponent}
    assert not (named := sorted(component_values.intersection(run_id.split("-")))), (
        f"run_id {run_id!r} carries the component name(s) {named} among its dash separated parts, so its unsplit "
        f"release would carry the very name a split launch of another run installs its {named[0]} release under"
    )

    namespace = config.namespace
    base_argv = _compute_base_argv(request, run_id=run_id)
    with override_argv(base_argv):
        args = parse_args()
    selector = DeploySelector.of(args)
    assert selector == config.deploy_selector, (
        f"the run's pods are told {selector.value} while everything this launch installs is named after "
        f"{config.deploy_component}"
    )
    deploys_orchestration_script = selector.deploys_orchestration_script()
    release = RunNames.release(run_id=run_id, deploy_component=selector)
    mooncake_plan = MooncakeInfo.plan_of_args(args) if deploys_orchestration_script else None
    pod_argv = MooncakeInfo.with_cluster_master(
        base_argv, plan=mooncake_plan, host=MooncakeInfo.master_service_host(release, namespace)
    )

    specs = compute_specs(args)
    chart = chart_dir(repo_base_dir=repo_base_dir)
    shared_root = InfraInfo.shared_root(InfraInfo.load(chart, list(config.helm_values)))
    run_directory = RunFiles.run_dir(shared_root=shared_root, run_id=run_id)

    if config.ci_run:
        _uninstall_leftover_ci_releases(namespace, keep_run_id=run_id)
    Helm.build_dependencies(chart)

    installed_manifest = Helm.get_manifest(release, namespace)
    hot_restart = plan_hot_restart(
        args,
        components=config.hot_restart_components,
        selector=selector,
        release=release,
        installed_manifest=installed_manifest,
    )
    state_file = (
        _compute_state_file(
            installed_manifest=installed_manifest,
            run_directory=run_directory,
            release=release,
            restarts_orchestration=hot_restart.restarts_orchestration,
        )
        if deploys_orchestration_script
        else None
    )

    plan = LaunchPlan(
        run_id=run_id,
        release=release,
        namespace=namespace,
        state_file=str(state_file) if state_file is not None else "",
        orchestrator_command=["python", request.train_script, *pod_argv] if deploys_orchestration_script else [],
        worker_argv=pod_argv,
        env=train_env_vars(request, {}, config=config),
        colocate=bool(args.colocate),
        mooncake_plan=mooncake_plan,
        prepare_cmd=request.prepare_cmd,
        restart_at=hot_restart.restart_at,
        restart_pools=hot_restart.restart_pools,
    )
    values_path = RunFiles.new_values_file(run_directory=run_directory)
    record = LaunchRecord.compute(plan=plan, values_file=values_path)
    record_path = RunFiles.new_record_file(run_directory=run_directory)
    plan = plan.model_copy(
        update={
            "launch_record": _compute_pod_record_file(installed_manifest=installed_manifest, record_path=record_path),
        }
    )
    _write_helm_values(values_path, build_values(specs, plan).as_values())
    values_files: list[str | Path] = [*config.helm_values, values_path]

    if installed_manifest is not None:
        _assert_upgrade_only_resizes(
            installed_manifest=installed_manifest,
            release=release,
            namespace=namespace,
            chart=chart,
            values_files=values_files,
            force=config.force,
            rebuilt_object_keys=hot_restart.rebuilt_object_keys,
        )
    if installed_manifest is None or hot_restart.restarts_orchestration:
        _remove_pending_uninstall(release, namespace=namespace)

    record.write(path=record_path)
    logger.info(f"What this launch launched is recorded under {record_path}")

    Helm.upgrade(
        release=release,
        namespace=namespace,
        chart=chart,
        values_files=values_files,
        ci_run=config.ci_run,
    )

    if not deploys_orchestration_script:
        logger.info(
            f"Installed {release}, which carries no orchestration script: it has no training to finish, so it stays "
            f"up until you uninstall it with `helm uninstall {release} --namespace {namespace}`. "
            f"{_describe_reachable_addrs(args, specs=specs, release=release)}"
        )
        return

    if (registration := _describe_registration_addrs(args, specs=specs, release=release)) is not None:
        logger.info(registration)

    if selector.is_split():
        logger.info(
            f"The other deployments of this run share the object store of {release}, so give each of them "
            f"{_describe_shared_object_store(mooncake_plan, release=release, namespace=namespace)}"
        )

    _follow_until_finished(release=release, namespace=namespace, state_file=state_file)


def _describe_shared_object_store(plan: MooncakePlan | None, *, release: str, namespace: str) -> str:
    assert plan is not None, (
        f"{release} carries the orchestration script of a split run, so it runs the object store master the other "
        f"deployments redeem their references at, and a run without one shares nothing"
    )
    init_kwargs = MooncakeInfo.cluster_init_kwargs(plan, host=MooncakeInfo.master_service_host(release, namespace))
    return (
        f"--object-store-backend {ObjectStoreBackend.MOONCAKE.value} "
        f"{MOONCAKE_INIT_KWARGS_FLAG} {shlex.quote(json.dumps(init_kwargs))}"
    )


def _describe_reachable_addrs(args, *, specs: list[BaseWorkerSpec], release: str) -> str:
    if not (workers := compute_addressed_workers(args, selector=DeploySelector.of(args))):
        return describe_registering_deployment(args)
    return f"Reach it with {_format_static_addrs(workers, specs=specs, release=release)}"


def _describe_registration_addrs(args, *, specs: list[BaseWorkerSpec], release: str) -> str | None:
    if not (workers := compute_registration_addressed_workers(args)):
        return None
    return (
        f"{args.expected_registration_reporters} engine deployment(s) register their engines into this one, so "
        f"launch each of them with --deploy-component {DeployComponent.INFERENCE.value}[:<engine group name>] "
        f"{_format_static_addrs(workers, specs=specs, release=release)} and the same --registration-token"
    )


def _format_static_addrs(workers: list[AddressedWorker], *, specs: list[BaseWorkerSpec], release: str) -> str:
    specs_by_pool_id = {spec.name: spec for spec in specs}
    return format_addressed_workers(
        [
            (
                worker,
                static_cell_addrs(spec=specs_by_pool_id[worker.pool_id], release=release, cell_index=0)[
                    worker.port_name
                ],
            )
            for worker in workers
        ]
    )


def _follow_until_finished(*, release: str, namespace: str, state_file: Path) -> None:
    logger.info(f"Following every pod of {release}; ctrl+c stops watching, not the run")
    orchestrator_workload = naming.component_name(release, naming.ORCHESTRATOR_COMPONENT)

    with with_observability(namespace=namespace, selector=Kubectl.release_selector(release)):
        outcome = wait_for_run(
            state_file=state_file,
            read_pod_phase=lambda: pod_phase(namespace, orchestrator_workload),
        )

    if outcome.exit_code != 0:
        _collect_diagnosis(release=release, namespace=namespace, state_file=state_file)

    logger.info(farewell(namespace=namespace, release=release, workload=orchestrator_workload))
    if outcome.exit_code != 0:
        raise SystemExit(outcome.exit_code)


def _compute_base_argv(request: ExecuteTrainRequest, *, run_id: str) -> list[str]:
    argv = [*shlex.split(shell_safe_model_args(request.megatron_model_type)), *shlex.split(request.train_args)]
    assert not ArgvManipulator.declares(argv, _ENV_REPORT_FLAG), (
        f"{_ENV_REPORT_FLAG} is what this launcher tells the pods about the launch that installed them, and an "
        f"argument of that name outranks it, so the pods would report a launch that never happened; drop it"
    )
    argv = ArgvManipulator.with_flag(argv, CLUSTER_BACKEND_FLAG, ClusterBackend.KUBERNETES.value)
    # TODO: generate different run_uuid even for same run_id, but at the same time allow helm upgrading
    return ArgvManipulator.with_flag(argv, _RUN_UUID_FLAG, derive_run_uuid(run_id))


def _compute_pod_record_file(*, installed_manifest: Manifest | None, record_path: Path) -> str | None:
    if installed_manifest is None:
        return str(record_path)
    return installed_launch_record_file(manifest=installed_manifest)


def _compute_state_file(
    *, installed_manifest: Manifest | None, run_directory: Path, release: str, restarts_orchestration: bool
) -> Path:
    if installed_manifest is None or restarts_orchestration:
        return RunFiles.new_state_file(run_directory=run_directory)

    attached_state_file = installed_manifest.state_file(container=naming.ORCHESTRATOR_COMPONENT)
    assert attached_state_file is not None, (
        f"Run {release} is installed but its orchestrator names no state file, so this launch cannot tell what it "
        f"is watching; uninstall it, or launch under a new run id"
    )
    return attached_state_file


def _assert_upgrade_only_resizes(
    *,
    installed_manifest: Manifest,
    release: str,
    namespace: str,
    chart: Path,
    values_files: list[str | Path],
    force: bool,
    rebuilt_object_keys: frozenset[str],
) -> None:
    proposed_manifest = Helm.render_upgrade(
        release=release, namespace=namespace, chart=chart, values_files=values_files
    )
    diff = manifest_diff.diff_manifests(
        before=installed_manifest, after=proposed_manifest, rebuilt_object_keys=rebuilt_object_keys
    )

    if diff.is_allowed:
        logger.info(f"Run {release} already exists; upgrading it:\n{diff.summarize_scaling()}")
        return

    scope = (
        f"more than the objects this hot restart replaces ({sorted(rebuilt_object_keys)})"
        if rebuilt_object_keys
        else "more than its size"
    )
    message = (
        f"Run {release} already exists and the relaunch would change {scope}:\n"
        f"{diff.describe()}\n"
        f"launch under a new run id, or pass force=True to apply this anyway and accept the restarts"
    )
    if not force:
        raise SystemExit(message)
    logger.warning(f"forced: {message}")


def _releases_of_run(run_id: str) -> set[str]:
    return {
        RunNames.release(run_id=run_id, deploy_component=DeploySelector(component=component))
        for component in DeployComponent
    }


def _belongs_to_run(release: str, *, run_id: str) -> bool:
    unsplit = RunNames.release(run_id=run_id)
    components = _releases_of_run(run_id) - {unsplit}
    return release == unsplit or release in components or any(release.startswith(f"{one}-") for one in components)


def _uninstall_leftover_ci_releases(namespace: str, *, keep_run_id: str) -> list[str]:
    listed = Helm.list_releases(namespace=namespace, selector=f"{CI_LABEL}=true")
    releases = [release for release in listed if not _belongs_to_run(release, run_id=keep_run_id)]
    for release in releases:
        logger.info(f"Uninstalling the leftover ci release {release} before this run installs its own")
        Helm.uninstall(release=release, namespace=namespace)
    return releases


def _remove_pending_uninstall(release: str, *, namespace: str) -> None:
    job = RunNames.uninstall_job(release=release)
    logger.info(f"Deleting {job} if it is pending, so it cannot uninstall the release this launch installs")
    Kubectl.delete_job(job, namespace=namespace, check=True)


def _collect_diagnosis(*, release: str, namespace: str, state_file: Path) -> None:
    try:
        diagnosis = collect_diagnosis(
            namespace=namespace,
            output_dir=state_file.parent,
            selector=Kubectl.release_selector(release),
            state_file=state_file,
        )
    except Exception:
        logger.warning("Could not collect a diagnosis of the failed run", exc_info=True)
        return

    logger.info(f"The pods of this failed run are described under {diagnosis.directory}")
    if not diagnosis.is_complete:
        logger.warning(f"The diagnosis is incomplete, these could not be collected: {', '.join(diagnosis.missing)}")


def _write_helm_values(path: Path, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(values, default_flow_style=False, sort_keys=True))
