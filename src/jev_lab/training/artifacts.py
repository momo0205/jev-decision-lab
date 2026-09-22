from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from jev_lab.contracts import RouteLabel

ParameterValue = str | int | float | bool | None


class ClassifierArtifactMetadata(BaseModel):
    model_id: str = Field(min_length=1)
    provider: Literal["tfidf-logreg"] = "tfidf-logreg"
    created_at: datetime
    dataset_path: str = Field(min_length=1)
    dataset_sha256: str = Field(min_length=1)
    training_split: Literal["dev"] = "dev"
    training_sample_ids: list[str] = Field(min_length=1)
    training_family_ids: list[str] = Field(min_length=1)
    labels: list[RouteLabel] = Field(min_length=1)
    sklearn_version: str = Field(min_length=1)
    tfidf_params: dict[str, ParameterValue]
    classifier_params: dict[str, ParameterValue]
    random_seed: int
    training_duration_ms: float = Field(ge=0)
    git_commit: str = Field(min_length=1)


class ClassifierTrainingManifest(ClassifierArtifactMetadata):
    model_size_bytes: int = Field(ge=1)
    model_sha256: str = Field(min_length=1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_artifact(
    *, pipeline: object, metadata: ClassifierArtifactMetadata, output_dir: Path
) -> Path:
    """Persist a locally created pipeline without replacing an existing artifact."""
    import joblib  # type: ignore[import-untyped]

    target = output_dir / metadata.model_id
    temporary = output_dir / f"{metadata.model_id}.tmp"
    if target.exists() or temporary.exists():
        raise FileExistsError(target if target.exists() else temporary)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        model_path = temporary / "model.joblib"
        joblib.dump(pipeline, model_path)
        manifest = ClassifierTrainingManifest(
            **metadata.model_dump(),
            model_size_bytes=model_path.stat().st_size,
            model_sha256=sha256_file(model_path),
        )
        (temporary / "manifest.json").write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )
        (temporary / "label-map.json").write_text(
            json.dumps([label.value for label in manifest.labels], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (temporary / "training-metrics.json").write_text(
            json.dumps({"training_duration_ms": manifest.training_duration_ms}, indent=2),
            encoding="utf-8",
        )
        temporary.replace(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def load_verified_pipeline(model_dir: Path) -> tuple[object, ClassifierTrainingManifest]:
    """Load only a trusted local artifact after checking it for accidental corruption.

    Joblib files can execute code while loading. A SHA-256 manifest is an integrity
    check, not authentication; never load artifacts obtained from untrusted sources.
    """
    import joblib

    manifest_path = model_dir / "manifest.json"
    model_path = model_dir / "model.joblib"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest missing: {manifest_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"model missing: {model_path}")
    manifest = ClassifierTrainingManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if sha256_file(model_path) != manifest.model_sha256:
        raise ValueError("model hash mismatch")
    if model_path.stat().st_size != manifest.model_size_bytes:
        raise ValueError("model size mismatch")
    return joblib.load(model_path), manifest
