from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from miles.utils.external_utils.command_utils.helm_backend.orchestrator.state import STATE_FILE_FLAG
from miles.utils.pydantic_utils import FrozenOpenBaseModel, FrozenStrictBaseModel

RESTART_AT_ANNOTATION = "miles.radixark.io/restart-at"


class EnvEntry(FrozenOpenBaseModel):
    name: str
    value: str = ""


class Container(FrozenOpenBaseModel):
    name: str
    command: list[str] = []
    env: list[EnvEntry] = []


class PodSpec(FrozenOpenBaseModel):
    containers: list[Container] = []


class PodTemplateMetadata(FrozenOpenBaseModel):
    annotations: dict[str, str] = {}


class PodTemplate(FrozenOpenBaseModel):
    metadata: PodTemplateMetadata | None = None
    spec: PodSpec | None = None


class ObjectSpec(FrozenOpenBaseModel):
    replicas: int | None = None
    template: PodTemplate | None = None


STATEFUL_SET_KIND = "StatefulSet"


def compute_manifest_object_key(*, kind: str, name: str) -> str:
    return f"{kind}/{name}"


class ObjectMetadata(FrozenOpenBaseModel):
    name: str


class ManifestObject(FrozenOpenBaseModel):
    kind: str
    metadata: ObjectMetadata
    spec: ObjectSpec | None = None

    @property
    def key(self) -> str:
        return compute_manifest_object_key(kind=self.kind, name=self.metadata.name)

    @property
    def replicas(self) -> int | None:
        return self.spec.replicas if self.spec is not None else None

    @property
    def restart_at(self) -> str | None:
        if self.spec is None or self.spec.template is None or self.spec.template.metadata is None:
            return None
        return self.spec.template.metadata.annotations.get(RESTART_AT_ANNOTATION)

    @property
    def body(self) -> dict[str, Any]:
        return self.model_dump(exclude_unset=True)

    @property
    def containers(self) -> list[Container]:
        if self.spec is None or self.spec.template is None:
            return []
        pod = self.spec.template.spec
        return list(pod.containers) if pod is not None else []

    def containers_named(self, container: str) -> list[Container]:
        if self.kind != STATEFUL_SET_KIND:
            return []
        return [described for described in self.containers if described.name == container]


class Manifest(FrozenStrictBaseModel):
    objects: list[ManifestObject]

    @classmethod
    def parse(cls, rendered: str) -> Manifest:
        return cls(objects=[document for document in yaml.safe_load_all(rendered) if document])

    @property
    def by_key(self) -> dict[str, ManifestObject]:
        return {described.key: described for described in self.objects}

    def restart_at(self, *, object_name: str) -> str | None:
        """The stamp a previous hot restart wrote, which an ordinary relaunch has to render unchanged."""
        described = self.by_key.get(compute_manifest_object_key(kind=STATEFUL_SET_KIND, name=object_name))
        return described.restart_at if described is not None else None

    def state_file(self, *, container: str) -> Path | None:
        for described in self.objects:
            for found in described.containers_named(container):
                if STATE_FILE_FLAG in found.command:
                    return Path(found.command[found.command.index(STATE_FILE_FLAG) + 1])
        return None
