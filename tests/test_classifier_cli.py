import json
from pathlib import Path

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
