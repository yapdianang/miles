import re
from dataclasses import dataclass
from enum import Enum

from miles.utils.pydantic_utils import FrozenStrictBaseModel


class ClusterBackend(Enum):
    RAY = "ray"
    KUBERNETES = "kubernetes"


class WorkerCommBackend(Enum):
    RAY = "ray"
    RPC = "rpc"


class DeployComponent(Enum):
    ALL = "all"
    PRIMARY = "primary"
    TRAINER = "trainer"
    INFERENCE = "inference"

    def selects(self, component: "DeployComponent") -> bool:
        assert component is not DeployComponent.ALL, "`all` is a selector over components, never a component itself"
        if self is DeployComponent.ALL:
            return True
        if self is DeployComponent.PRIMARY and component is DeployComponent.INFERENCE:
            return True
        return self is component

    def deploys_orchestration_script(self) -> bool:
        return self.selects(DeployComponent.PRIMARY)

    def is_split(self) -> bool:
        return self is not DeployComponent.ALL

    def takes_instance(self) -> bool:
        return self in (DeployComponent.TRAINER, DeployComponent.INFERENCE)


DEPLOY_INSTANCE_SEPARATOR = ":"

_ENGINE_GROUP_NAME_PATTERN = re.compile(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?")


@dataclass(frozen=True)
class DeploySelector:
    component: DeployComponent
    instance: str | None = None

    @classmethod
    def parse(cls, value: str) -> "DeploySelector":
        name, separator, instance = value.partition(DEPLOY_INSTANCE_SEPARATOR)
        assert name in {one.value for one in DeployComponent}, (
            f"--deploy-component {value!r} names {name!r}, which is not one of "
            f"{[one.value for one in DeployComponent]}"
        )
        component = DeployComponent(name)
        assert not separator or instance, (
            f"--deploy-component {value!r} ends in {DEPLOY_INSTANCE_SEPARATOR!r} without naming an instance; "
            f"drop the separator to deploy every instance of {component.value}"
        )
        assert not instance or component.takes_instance(), (
            f"--deploy-component {value!r} names an instance of {component.value}, but a run deploys exactly one "
            f"of it; only {[one.value for one in DeployComponent if one.takes_instance()]} come in instances"
        )
        assert (
            component is not DeployComponent.INFERENCE
            or not instance
            or _ENGINE_GROUP_NAME_PATTERN.fullmatch(instance)
        ), (
            f"--deploy-component {value!r} names the engine group {instance!r}, which names the release this launch "
            f"installs and namespaces the pool ids it registers, so it has to match "
            f"{_ENGINE_GROUP_NAME_PATTERN.pattern}"
        )
        return cls(component=component, instance=instance or None)

    @classmethod
    def of(cls, args) -> "DeploySelector":
        return cls.parse(args.deploy_component)

    @property
    def value(self) -> str:
        if self.instance is None:
            return self.component.value
        return f"{self.component.value}{DEPLOY_INSTANCE_SEPARATOR}{self.instance}"

    def selects(self, component: DeployComponent, *, instance: str | None = None) -> bool:
        if not self.component.selects(component):
            return False
        return self.instance is None or instance is None or self.instance == instance

    def deploys_orchestration_script(self) -> bool:
        return self.component.deploys_orchestration_script()

    def is_split(self) -> bool:
        return self.component.is_split()


class DeploymentIdentity(FrozenStrictBaseModel):
    run_uuid: str
    deploy_component: str
    deploy_instance: str | None = None


_SUPPORTED_WORKER_COMM_BACKENDS = {
    ClusterBackend.RAY: (WorkerCommBackend.RAY, WorkerCommBackend.RPC),
    ClusterBackend.KUBERNETES: (WorkerCommBackend.RPC,),
}


def resolve_worker_comm_backend(*, cluster_backend: ClusterBackend, requested: str | None) -> WorkerCommBackend:
    if requested is None:
        return _SUPPORTED_WORKER_COMM_BACKENDS[cluster_backend][0]

    backend = WorkerCommBackend(requested)
    supported = _SUPPORTED_WORKER_COMM_BACKENDS[cluster_backend]
    assert backend in supported, (
        f"--worker-comm-backend {backend.value} is not available under --cluster-backend {cluster_backend.value}, "
        f"which speaks {[one.value for one in supported]}"
    )
    return backend
