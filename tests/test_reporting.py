import json
from datetime import UTC, datetime
from pathlib import Path

from jev_lab.contracts import RunManifest, RunMode, Split
from jev_lab.metrics import EvaluationMetrics
from jev_lab.reporting import write_report


def test_report_names_recorded_replay_without_live_claim(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    manifest = RunManifest(
        run_id="r1",
        provider="jev",
        model_version="fixture",
        run_mode=RunMode.RECORDED_REPLAY,
        split=Split.DEV,
        dataset_path="data",
        dataset_sha256="hash",
        git_commit="commit",
        threshold=0.7,
        model_artifact_sha256="artifact-hash",
        training_sample_count=60,
        training_duration_ms=12.5,
        model_size_bytes=2048,
        created_at=datetime.now(UTC),
    )
    (run / "manifest.json").write_text(manifest.model_dump_json())
    metrics = EvaluationMetrics(
        accuracy=None,
        coverage=0,
        selective_accuracy=None,
        success_rate=0,
        confusion_matrix={},
        per_class={},
        high_risk_false_approvals=0,
        probability_sample_count=0,
        brier_score=None,
        calibration=[],
        p50_latency_ms=None,
        p95_latency_ms=None,
        total_known_cost_usd=None,
        mean_known_cost_usd=None,
    )
    (run / "metrics.json").write_text(metrics.model_dump_json())
    (run / "results.jsonl").write_text("")
    markdown, public_json = write_report(run, tmp_path / "reports", tmp_path / "public")
    assert "recorded-replay" in markdown.read_text()
    assert "live-evaluation" not in markdown.read_text()
    assert json.loads(public_json.read_text())["run_mode"] == "recorded-replay"
    assert json.loads(public_json.read_text())["threshold"] == 0.7
    assert "not available" in markdown.read_text()
    assert "artifact-hash" in markdown.read_text()
    public = json.loads(public_json.read_text())
    assert public["local_model_provenance"]["training_sample_count"] == 60
