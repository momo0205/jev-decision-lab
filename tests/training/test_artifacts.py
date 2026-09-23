from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

import pytest

pytest.importorskip("joblib")

from jev_lab.contracts import RouteLabel
from jev_lab.training.artifacts import (
    ClassifierArtifactMetadata,
    load_verified_pipeline,
    write_artifact,
)

pytestmark = pytest.mark.classifier


class ArtifactPipeline:
    classes_: ClassVar[list[str]] = [label.value for label in RouteLabel]


def metadata(model_id: str = "model-1") -> ClassifierArtifactMetadata:
    return ClassifierArtifactMetadata(
        model_id=model_id,
        created_at=datetime.now(UTC),
        dataset_path="datasets/routing-v1.yaml",
        dataset_sha256="dataset-hash",
        training_sample_ids=["sample-1"],
        training_family_ids=["family-1"],
        labels=list(RouteLabel),
        sklearn_version="1.5.0",
        tfidf_params={"analyzer": "char"},
        classifier_params={"max_iter": 1000},
        random_seed=42,
        training_duration_ms=1.5,
        git_commit="abc123",
    )


def write_test_artifact(output_dir: Path, model_id: str = "model-1") -> Path:
    return write_artifact(
        pipeline=ArtifactPipeline(),
        metadata=metadata(model_id),
        output_dir=output_dir,
    )


def test_verified_load_round_trips_pipeline_and_manifest(tmp_path: Path) -> None:
    model_dir = write_test_artifact(tmp_path)

    pipeline, manifest = load_verified_pipeline(model_dir)

    assert isinstance(pipeline, ArtifactPipeline)
    assert manifest.model_id == "model-1"
    assert manifest.model_size_bytes == (model_dir / "model.joblib").stat().st_size
    assert manifest.training_split == "dev"


def test_verified_load_rejects_changed_model(tmp_path: Path) -> None:
    model_dir = write_test_artifact(tmp_path)
    (model_dir / "model.joblib").write_bytes(b"changed")

    with pytest.raises(ValueError, match="model hash mismatch"):
        load_verified_pipeline(model_dir)


def test_artifact_writer_refuses_existing_directory(tmp_path: Path) -> None:
    target = tmp_path / "same-id"
    write_test_artifact(tmp_path, model_id="same-id")

    with pytest.raises(FileExistsError):
        write_test_artifact(tmp_path, model_id="same-id")

    assert target.exists()


def test_verified_load_rejects_missing_manifest(tmp_path: Path) -> None:
    model_dir = tmp_path / "missing"
    model_dir.mkdir()
    (model_dir / "model.joblib").write_bytes(b"model")

    with pytest.raises(FileNotFoundError, match="manifest"):
        load_verified_pipeline(model_dir)


def test_verified_load_rejects_changed_label_map(tmp_path: Path) -> None:
    model_dir = write_test_artifact(tmp_path)
    (model_dir / "label-map.json").write_text('["search"]')

    with pytest.raises(ValueError, match="label map mismatch"):
        load_verified_pipeline(model_dir)
