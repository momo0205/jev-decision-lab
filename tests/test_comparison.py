from datetime import UTC, datetime
from pathlib import Path

import pytest

from jev_lab.comparison import compare_runs
from jev_lab.contracts import RunManifest, RunMode, Split
from jev_lab.metrics import EvaluationMetrics


def build_run(path: Path, provider: str, dataset_hash: str = "hash") -> Path:
    path.mkdir()
    manifest = RunManifest(
        run_id=path.name,
        provider=provider,
        model_version="v1",
        run_mode=RunMode.OFFLINE_DEVELOPMENT,
        split=Split.CALIBRATION,
        dataset_path="data",
        dataset_sha256=dataset_hash,
        git_commit="commit",
        threshold=0.7,
        created_at=datetime.now(UTC),
        training_sample_count=60 if provider == "tfidf-logreg" else None,
        training_duration_ms=12 if provider == "tfidf-logreg" else None,
        model_size_bytes=1024 if provider == "tfidf-logreg" else None,
    )
    metrics = EvaluationMetrics(
        accuracy=0.8,
        coverage=0.9,
        selective_accuracy=0.85,
        success_rate=1,
        confusion_matrix={},
        per_class={},
        high_risk_false_approvals=0,
        probability_sample_count=20,
        brier_score=0.2,
        calibration=[],
        p50_latency_ms=1,
        p95_latency_ms=2,
        total_known_cost_usd=0 if provider == "tfidf-logreg" else None,
        mean_known_cost_usd=0 if provider == "tfidf-logreg" else None,
    )
    (path / "manifest.json").write_text(manifest.model_dump_json())
    (path / "metrics.json").write_text(metrics.model_dump_json())
    return path


def test_compare_runs_keeps_missing_costs_explicit(tmp_path: Path) -> None:
    classifier_run = build_run(tmp_path / "classifier", "tfidf-logreg")
    jev_run = build_run(tmp_path / "jev", "jev")

    rows = compare_runs([classifier_run, jev_run])

    assert rows[0].mean_known_cost_usd == 0.0
    assert rows[0].training_sample_count == 60
    assert rows[1].mean_known_cost_usd is None


def test_compare_runs_rejects_different_dataset_hashes(tmp_path: Path) -> None:
    first = build_run(tmp_path / "first", "rules", "one")
    second = build_run(tmp_path / "second", "jev", "two")

    with pytest.raises(ValueError, match="dataset hash"):
        compare_runs([first, second])
