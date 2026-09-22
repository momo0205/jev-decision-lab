from pathlib import Path

from pydantic import BaseModel

from jev_lab.contracts import RunManifest, RunMode, Split
from jev_lab.metrics import EvaluationMetrics


class ComparisonRow(BaseModel):
    provider: str
    model_version: str
    run_mode: RunMode
    dataset_sha256: str
    split: Split
    threshold: float | None
    accuracy: float | None
    coverage: float
    selective_accuracy: float | None
    brier_score: float | None
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    mean_known_cost_usd: float | None
    training_sample_count: int | None
    training_duration_ms: float | None
    model_size_bytes: int | None


def compare_runs(run_dirs: list[Path]) -> list[ComparisonRow]:
    if not run_dirs:
        return []
    rows: list[ComparisonRow] = []
    dataset_hashes: set[str] = set()
    splits: set[Split] = set()
    for run_dir in run_dirs:
        manifest = RunManifest.model_validate_json((run_dir / "manifest.json").read_text())
        metrics = EvaluationMetrics.model_validate_json((run_dir / "metrics.json").read_text())
        dataset_hashes.add(manifest.dataset_sha256)
        splits.add(manifest.split)
        rows.append(
            ComparisonRow(
                provider=manifest.provider,
                model_version=manifest.model_version,
                run_mode=manifest.run_mode,
                dataset_sha256=manifest.dataset_sha256,
                split=manifest.split,
                threshold=manifest.threshold,
                accuracy=metrics.accuracy,
                coverage=metrics.coverage,
                selective_accuracy=metrics.selective_accuracy,
                brier_score=metrics.brier_score,
                p50_latency_ms=metrics.p50_latency_ms,
                p95_latency_ms=metrics.p95_latency_ms,
                mean_known_cost_usd=metrics.mean_known_cost_usd,
                training_sample_count=manifest.training_sample_count,
                training_duration_ms=manifest.training_duration_ms,
                model_size_bytes=manifest.model_size_bytes,
            )
        )
    if len(dataset_hashes) != 1:
        raise ValueError("cannot compare different dataset hashes")
    if len(splits) != 1:
        raise ValueError("cannot compare different splits")
    return rows
