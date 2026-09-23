import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from typer.testing import CliRunner

from jev_lab.cli import app


def test_classifier_run_requires_model_without_creating_run(tmp_path: Path) -> None:
    output = tmp_path / "runs"
    result = CliRunner().invoke(
        app,
        ["run", "--provider", "tfidf-logreg", "--split", "dev", "--output-dir", str(output)],
    )

    assert result.exit_code != 0
    assert "--model is required" in result.output
    assert not output.exists()


def test_train_rejects_calibration_before_artifact_creation(tmp_path: Path) -> None:
    output = tmp_path / "artifacts"
    result = CliRunner().invoke(
        app,
        [
            "train",
            "--provider",
            "tfidf-logreg",
            "--split",
            "calibration",
            "--model-id",
            "m1",
            "--output-dir",
            str(output),
        ],
    )

    assert result.exit_code != 0
    assert not output.exists()


@pytest.mark.classifier
def test_classifier_train_run_and_evaluate(tmp_path: Path) -> None:
    runner = CliRunner()
    artifacts = tmp_path / "artifacts"
    trained = runner.invoke(
        app,
        [
            "train",
            "--provider",
            "tfidf-logreg",
            "--split",
            "dev",
            "--model-id",
            "m1",
            "--output-dir",
            str(artifacts),
        ],
    )
    assert trained.exit_code == 0, trained.output
    run = runner.invoke(
        app,
        [
            "run",
            "--provider",
            "tfidf-logreg",
            "--model",
            str(artifacts / "m1"),
            "--split",
            "calibration",
            "--output-dir",
            str(tmp_path / "runs"),
        ],
    )
    assert run.exit_code == 0, run.output
    run_dir = Path(run.output.strip().splitlines()[-1])
    evaluated = runner.invoke(app, ["evaluate", "--run", str(run_dir)])
    assert evaluated.exit_code == 0, evaluated.output
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["probability_sample_count"] == 20
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["training_sample_count"] == 60


@pytest.mark.classifier
def test_classifier_run_rejects_artifact_for_different_dataset(tmp_path: Path) -> None:
    runner = CliRunner()
    artifacts = tmp_path / "artifacts"
    trained = runner.invoke(
        app,
        ["train", "--provider", "tfidf-logreg", "--split", "dev", "--model-id", "m1", "--output-dir", str(artifacts)],
    )
    assert trained.exit_code == 0
    manifest_path = artifacts / "m1" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["dataset_sha256"] = "different"
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "runs"

    result = runner.invoke(
        app,
        ["run", "--provider", "tfidf-logreg", "--model", str(artifacts / "m1"), "--split", "calibration", "--output-dir", str(output)],
    )

    assert result.exit_code != 0
    assert "dataset" in result.output.lower()
    assert not output.exists()


def test_classifier_model_help_warns_about_trusted_local_artifacts() -> None:
    result = CliRunner().invoke(app, ["run", "--help"])

    assert result.exit_code == 0
    assert "trusted local" in result.output.lower()


def test_train_dev_cli_flow_is_testable_without_classifier_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    observed: dict[str, object] = {}
    fake_training = ModuleType("jev_lab.training.classifier")

    def cross_validate(samples: list[object], folds: int, random_seed: int) -> SimpleNamespace:
        observed["samples"] = samples
        return SimpleNamespace(fold_count=folds, mean_accuracy=0.5, standard_deviation=0.1)

    def fit(
        samples: list[object],
        dataset_path: Path,
        output_dir: Path,
        model_id: str,
        git_commit: str,
        random_seed: int,
    ) -> Path:
        observed["train_samples"] = samples
        observed["dataset_path"] = dataset_path
        observed["git_commit"] = git_commit
        return output_dir / model_id

    fake_training.cross_validate_classifier = cross_validate  # type: ignore[attr-defined]
    fake_training.train_classifier = fit  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "jev_lab.training.classifier", fake_training)

    result = CliRunner().invoke(
        app,
        [
            "train",
            "--provider",
            "tfidf-logreg",
            "--split",
            "dev",
            "--model-id",
            "core-smoke",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "core-smoke" in result.output
    assert len(observed["samples"]) == 60
    assert observed["samples"] == observed["train_samples"]


def test_classifier_run_cli_dispatches_without_optional_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import UTC, datetime

    from jev_lab.contracts import (
        DecisionResult,
        RequestStatus,
        RouteLabel,
        RunMode,
        Split,
    )
    from jev_lab.dataset import dataset_sha256, load_dataset
    from jev_lab.training.artifacts import ClassifierTrainingManifest

    dev_samples = [sample for sample in load_dataset(Path("datasets/routing-v1.yaml")) if sample.split == Split.DEV]
    manifest = ClassifierTrainingManifest(
        model_id="fake-local-model",
        created_at=datetime.now(UTC),
        dataset_path="datasets/routing-v1.yaml",
        dataset_sha256=dataset_sha256(Path("datasets/routing-v1.yaml")),
        training_sample_ids=[sample.sample_id for sample in dev_samples],
        training_family_ids=sorted({sample.family_id for sample in dev_samples}),
        labels=list(RouteLabel),
        sklearn_version="not-loaded-in-this-test",
        tfidf_params={},
        classifier_params={},
        random_seed=42,
        training_duration_ms=1,
        git_commit="test-commit",
        model_size_bytes=1,
        model_sha256="fake-model-hash",
    )

    class FakeClassifier:
        name = "tfidf-logreg"
        model_version = manifest.model_id
        run_mode = RunMode.OFFLINE_DEVELOPMENT

        def __init__(self) -> None:
            self.manifest = manifest

        def decide(self, sample: object) -> DecisionResult:
            return DecisionResult(
                sample_id=sample.sample_id,  # type: ignore[attr-defined]
                label=RouteLabel.SEARCH,
                probabilities={
                    RouteLabel.SEARCH: 1.0,
                    RouteLabel.CODE: 0.0,
                    RouteLabel.DATABASE: 0.0,
                    RouteLabel.HUMAN_REVIEW: 0.0,
                },
                abstained=False,
                latency_ms=0.1,
                estimated_cost_usd=0,
                provider=self.name,
                model_version=self.model_version,
                request_status=RequestStatus.SUCCESS,
                error=None,
                run_mode=self.run_mode,
            )

    monkeypatch.setattr(
        "jev_lab.cli.ClassifierProvider.from_artifact", lambda _path: FakeClassifier()
    )
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--provider",
            "tfidf-logreg",
            "--model",
            "trusted-local-path",
            "--split",
            "calibration",
            "--output-dir",
            str(tmp_path / "runs"),
        ],
    )

    assert result.exit_code == 0, result.output
    run_dir = Path(result.output.strip().splitlines()[-1])
    saved_manifest = json.loads((run_dir / "manifest.json").read_text())
    assert saved_manifest["training_sample_count"] == 60
