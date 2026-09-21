import json
from datetime import UTC, datetime
from pathlib import Path

from jev_lab.contracts import RouteLabel, RoutingSample, RunManifest, RunMode, Split
from jev_lab.runner import run_experiment


class ExplodingProvider:
    name = "jev"
    model_version = "test"
    run_mode = RunMode.LIVE_EVALUATION

    def decide(self, sample: RoutingSample):  # type: ignore[no-untyped-def]
        raise TimeoutError("secret raw message")


def test_runner_records_failure_without_fallback(tmp_path: Path) -> None:
    sample = RoutingSample(
        sample_id="s1",
        family_id="f1",
        input="查天气",
        expected=RouteLabel.SEARCH,
        acceptable=[RouteLabel.SEARCH],
        risk="low",
        difficulty="clear",
        rationale="test",
        source="synthetic",
        split=Split.DEV,
    )
    manifest = RunManifest(
        run_id="r1",
        provider="jev",
        model_version="test",
        run_mode=RunMode.LIVE_EVALUATION,
        split=Split.DEV,
        dataset_path="data",
        dataset_sha256="hash",
        git_commit="commit",
        created_at=datetime.now(UTC),
    )
    artifact = run_experiment(ExplodingProvider(), [sample], tmp_path, manifest)
    rows = [json.loads(line) for line in artifact.read_text().splitlines()]
    assert rows[0]["provider"] == "jev"
    assert rows[0]["request_status"] == "failed"
    assert rows[0]["label"] is None
    assert rows[0]["error"] == "TimeoutError"
