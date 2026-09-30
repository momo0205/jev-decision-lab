from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace

from jev_lab import runtime
from jev_lab.runtime import collect_runtime_provenance


def test_laya_runtime_provenance_contains_only_allowlisted_environment(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: list[str] = []

    def version(distribution: str) -> str:
        calls.append(distribution)
        if distribution == "torch":
            raise PackageNotFoundError(distribution)
        return {"laya": "0.3.21", "transformers": "4.55.0"}[distribution]

    monkeypatch.setattr(runtime.platform, "python_version", lambda: "3.12.4")
    monkeypatch.setattr(runtime.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(runtime.metadata, "version", version)

    provenance = collect_runtime_provenance(
        "laya", inference_agent=SimpleNamespace(device="cuda:0")
    )

    assert provenance.model_dump(mode="json") == {
        "python_version": "3.12.4",
        "system": "Darwin",
        "machine": "arm64",
        "package_versions": {"laya": "0.3.21", "transformers": "4.55.0"},
        "inference_device": "cuda:0",
    }
    assert calls == ["laya", "torch", "transformers"]


def test_unknown_platform_and_unexposed_device_stay_explicit(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(runtime.platform, "python_version", lambda: "3.12.4")
    monkeypatch.setattr(runtime.platform, "system", lambda: "NovelOS")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "")
    monkeypatch.setattr(
        runtime.metadata,
        "version",
        lambda _: (_ for _ in ()).throw(PackageNotFoundError()),
    )

    provenance = collect_runtime_provenance("laya", inference_agent=SimpleNamespace())

    assert provenance.system == "Other"
    assert provenance.machine == "unknown"
    assert provenance.package_versions == {}
    assert provenance.inference_device is None


def test_unrecognized_device_value_is_not_serialized(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(runtime.platform, "python_version", lambda: "3.12.4")
    monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "x86_64")

    provenance = collect_runtime_provenance(
        "laya", inference_agent=SimpleNamespace(device="cuda:0\nsecret")
    )

    assert provenance.inference_device is None


def test_non_laya_provider_does_not_query_laya_packages_or_capture_device(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(runtime.platform, "python_version", lambda: "3.12.4")
    monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "x86_64")

    def unexpected_package_lookup(_: str) -> str:
        raise AssertionError("non-Laya runs must not inspect optional Laya packages")

    monkeypatch.setattr(runtime.metadata, "version", unexpected_package_lookup)

    provenance = collect_runtime_provenance(
        "rules", inference_agent=SimpleNamespace(device="cuda:0")
    )

    assert provenance.package_versions == {}
    assert provenance.inference_device is None
