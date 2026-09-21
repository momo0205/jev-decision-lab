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
