import json
import os
import subprocess
from pathlib import Path


def run_cli(arguments: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(Path(".venv/bin/jev-lab")), *arguments],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def test_complete_rules_workflow_without_keys_or_network(tmp_path: Path) -> None:
    env = dict(os.environ)
    env.pop("DEEPSEEK_API_KEY", None)
    env.pop("TYPESAFE_API_KEY", None)
    env["HTTP_PROXY"] = "http://127.0.0.1:1"
    env["HTTPS_PROXY"] = "http://127.0.0.1:1"
    validate = run_cli(["dataset", "validate"], env)
    run = run_cli(
        ["run", "--provider", "rules", "--split", "dev", "--output-dir", str(tmp_path / "runs")],
        env,
    )
    run_dir = Path(run.stdout.strip().splitlines()[-1])
    assert run_cli(["evaluate", "--run", str(run_dir)], env).returncode == 0
    assert (
        run_cli(
            [
                "report",
                "--run",
                str(run_dir),
                "--report-dir",
                str(tmp_path / "reports"),
                "--public-dir",
                str(tmp_path / "public"),
            ],
            env,
        ).returncode
        == 0
    )
    assert "valid: 100 samples" in validate.stdout
    public = tmp_path / "public" / f"{run_dir.name}.json"
    assert public.exists()
    assert "input" not in json.loads(public.read_text())
