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
