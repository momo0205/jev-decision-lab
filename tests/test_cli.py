import json
from pathlib import Path

from typer.testing import CliRunner

from jev_lab.cli import app


def test_root_help_is_available_offline() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "dataset" in result.stdout


def test_run_exposes_provider_and_split_as_options() -> None:
    result = CliRunner().invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--provider" in result.stdout
    assert "--split" in result.stdout


def test_rules_workflow_runs_in_process(tmp_path: Path) -> None:
    runner = CliRunner()
    run = runner.invoke(
        app,
        ["run", "--provider", "rules", "--split", "dev", "--output-dir", str(tmp_path / "runs")],
    )
    assert run.exit_code == 0
    run_dir = Path(run.stdout.strip())
    evaluated = runner.invoke(app, ["evaluate", "--run", str(run_dir)])
    assert evaluated.exit_code == 0
    reported = runner.invoke(
        app,
        [
            "report",
            "--run",
            str(run_dir),
            "--report-dir",
            str(tmp_path / "reports"),
            "--public-dir",
            str(tmp_path / "public"),
        ],
    )
    assert reported.exit_code == 0
    assert (tmp_path / "public" / f"{run_dir.name}.json").exists()


def test_run_rejects_invalid_dataset_before_creating_artifacts(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("[]")
    monkeypatch.setattr("jev_lab.cli.DATASET", invalid)
    output = tmp_path / "runs"
    result = CliRunner().invoke(
        app, ["run", "--provider", "rules", "--split", "dev", "--output-dir", str(output)]
    )
    assert result.exit_code != 0
    assert not output.exists()


def test_run_persists_validated_threshold(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--provider",
            "rules",
            "--split",
            "dev",
            "--threshold",
            "0.75",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0
    manifest = json.loads((Path(result.stdout.strip()) / "manifest.json").read_text())
    assert manifest["threshold"] == 0.75


def test_missing_deepseek_key_prints_guidance_and_is_not_live(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    result = CliRunner().invoke(
        app,
        ["run", "--provider", "deepseek", "--split", "calibration", "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 0
    lines = result.stdout.strip().splitlines()
    assert "DEEPSEEK_API_KEY" in lines[0]
    manifest = json.loads((Path(lines[-1]) / "manifest.json").read_text())
    assert manifest["run_mode"] == "offline-development"
