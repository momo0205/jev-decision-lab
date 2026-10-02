import json
import re
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import jev_lab.cli as cli_module
from jev_lab.cli import app
from jev_lab.contracts import RouteLabel, RoutingSample, Split
from jev_lab.providers.laya import LayaDependencyError

ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def test_run_records_runtime_provenance_without_loading_repository_dataset(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    sample = RoutingSample(
        sample_id="synthetic-dev-1",
        family_id="synthetic-family-1",
        input="搜索公开资料",
        expected=RouteLabel.SEARCH,
        acceptable=[RouteLabel.SEARCH],
        risk="low",
        difficulty="clear",
        rationale="仅供 CLI 集成测试使用",
        source="synthetic",
        split=Split.DEV,
    )

    class FakeLayaAgent:
        device = "mps"

        def predict(
            self, state: str, questions: dict[str, dict[str, object]], *, lang: str
        ) -> dict[str, object]:
            return {
                "answers": {
                    "route": {
                        "type": "choice",
                        "choice": "search",
                        "probabilities": {
                            "search": 0.7,
                            "code": 0.1,
                            "database": 0.1,
                            "human_review": 0.1,
                        },
                    }
                }
            }

    agent = FakeLayaAgent()
    monkeypatch.setattr(cli_module, "load_dataset", lambda _: [sample])
    monkeypatch.setattr(cli_module, "validate_dataset", lambda _: None)
    monkeypatch.setattr(cli_module, "dataset_sha256", lambda _: "synthetic-dataset-hash")
    monkeypatch.setattr(
        cli_module.LayaProvider,
        "from_default_checkpoint",
        lambda: cli_module.LayaProvider(agent),
    )
    passed_agents: list[object | None] = []
    runtime_payload = {
        "python_version": "3.12.4",
        "system": "Darwin",
        "machine": "arm64",
        "package_versions": {"laya": "0.3.21"},
        "inference_device": "mps",
    }

    def collect_runtime(
        provider: str, *, inference_agent: object | None = None
    ) -> dict[str, object]:
        assert provider == "laya"
        passed_agents.append(inference_agent)
        return runtime_payload

    monkeypatch.setattr(cli_module, "collect_runtime_provenance", collect_runtime, raising=False)

    result = CliRunner().invoke(
        app,
        [
            "run",
            "--provider",
            "laya",
            "--split",
            "dev",
            "--output-dir",
            str(tmp_path / "runs"),
        ],
    )

    assert result.exit_code == 0, result.output
    run_dir = Path(result.stdout.strip())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert passed_agents == [agent]
    assert manifest["runtime"] == runtime_payload


def test_root_help_is_available_offline() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "dataset" in result.stdout


def test_run_exposes_provider_and_split_as_options() -> None:
    result = CliRunner().invoke(app, ["run", "--help"], color=True)
    assert result.exit_code == 0
    plain_help = ANSI_ESCAPE.sub("", result.stdout)
    assert "--provider" in plain_help
    assert "--split" in plain_help


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


def test_missing_laya_extra_prints_install_guidance_before_creating_run(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    def missing_laya() -> None:
        raise LayaDependencyError("Laya is optional; install it with `pip install -e '.[laya]'`.")

    monkeypatch.setattr(
        "jev_lab.cli.LayaProvider",
        SimpleNamespace(from_default_checkpoint=missing_laya),
        raising=False,
    )
    output = tmp_path / "runs"
    result = CliRunner().invoke(
        app,
        ["run", "--provider", "laya", "--split", "calibration", "--output-dir", str(output)],
    )

    plain_output = ANSI_ESCAPE.sub("", result.output)
    assert result.exit_code != 0
    assert "[laya]" in plain_output
    assert not output.exists()
