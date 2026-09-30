from __future__ import annotations

import platform
import re
from importlib import metadata
from typing import Final, Literal, Protocol, cast

from jev_lab.contracts import RuntimeProvenance

_PackageName = Literal["laya", "torch", "transformers"]
_SystemFamily = Literal["Darwin", "Linux", "Windows", "Other"]

_PACKAGE_ALLOWLIST: Final[tuple[_PackageName, ...]] = ("laya", "torch", "transformers")
_SAFE_MACHINE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SAFE_PACKAGE_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.!+_-]{0,79}$")
_SAFE_DEVICE = re.compile(
    r"^(cpu|mps|cuda(:[0-9]+)?|xpu(:[0-9]+)?|npu(:[0-9]+)?|hpu(:[0-9]+)?|privateuseone(:[0-9]+)?)$"
)


class _AgentWithDevice(Protocol):
    device: object


class _DeviceLike(Protocol):
    type: object
    index: object


def _system_family(value: str) -> _SystemFamily:
    if value == "Darwin":
        return "Darwin"
    if value == "Linux":
        return "Linux"
    if value == "Windows":
        return "Windows"
    return "Other"


def _machine_architecture(value: str) -> str:
    return value if _SAFE_MACHINE.fullmatch(value) else "unknown"


def _device_identifier(inference_agent: object | None) -> str | None:
    if inference_agent is None:
        return None
    try:
        device = cast(_AgentWithDevice, inference_agent).device
    except Exception:  # noqa: BLE001 - unavailable device metadata is explicitly null
        return None

    if isinstance(device, str):
        candidate = device
    else:
        try:
            device_like = cast(_DeviceLike, device)
            device_type = device_like.type
            device_index = device_like.index
        except Exception:  # noqa: BLE001 - do not expose arbitrary device-object reprs
            return None
        if not isinstance(device_type, str):
            return None
        if device_index is None:
            candidate = device_type
        elif (
            isinstance(device_index, int)
            and not isinstance(device_index, bool)
            and device_index >= 0
        ):
            candidate = f"{device_type}:{device_index}"
        else:
            return None

    return candidate if _SAFE_DEVICE.fullmatch(candidate) else None


def collect_runtime_provenance(
    provider: str, *, inference_agent: object | None = None
) -> RuntimeProvenance:
    package_versions: dict[_PackageName, str] = {}
    inference_device = None
    if provider == "laya":
        for distribution in _PACKAGE_ALLOWLIST:
            try:
                version = metadata.version(distribution)
            except metadata.PackageNotFoundError:
                continue
            if _SAFE_PACKAGE_VERSION.fullmatch(version):
                package_versions[distribution] = version
        inference_device = _device_identifier(inference_agent)

    raw_machine = platform.machine()
    return RuntimeProvenance(
        python_version=platform.python_version(),
        system=_system_family(platform.system()),
        machine=_machine_architecture(raw_machine),
        package_versions=package_versions,
        inference_device=inference_device,
    )
