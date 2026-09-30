from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from jev_lab.contracts import DecisionResult, RunManifest
from jev_lab.metrics import EvaluationMetrics
from jev_lab.security import JSONValue, assert_public_safe


def _display(value: object) -> str:
    return "not available" if value is None else str(value)


def _runtime_lines(manifest: RunManifest) -> str:
    runtime = manifest.runtime
    if runtime is None:
        return """## Runtime provenance

- Python version: not available
- OS family: not available
- Machine architecture: not available
- Package versions: not available
- Inference device: not available"""

    packages = ", ".join(
        f"{name}={version}" for name, version in sorted(runtime.package_versions.items())
    )
    return f"""## Runtime provenance

- Python version: {runtime.python_version}
- OS family: {runtime.system}
- Machine architecture: {runtime.machine}
- Package versions: {_display(packages)}
- Inference device: {_display(runtime.inference_device)}"""


def write_report(run_dir: Path, report_dir: Path, public_dir: Path) -> tuple[Path, Path]:
    manifest = RunManifest.model_validate_json((run_dir / "manifest.json").read_text())
    metrics = EvaluationMetrics.model_validate_json((run_dir / "metrics.json").read_text())
    results = [
        DecisionResult.model_validate_json(line)
        for line in (run_dir / "results.jsonl").read_text().splitlines()
    ]
    failures = Counter(result.error or "unspecified" for result in results if result.error)
    markdown = f"""# Decision evaluation: {manifest.run_id}

## Provenance

- provider: {manifest.provider}
- model: {manifest.model_version}
- run mode: {manifest.run_mode.value}
- git commit: {manifest.git_commit}

{_runtime_lines(manifest)}

## Dataset and split

- dataset sha256: {manifest.dataset_sha256}
- split: {manifest.split.value}
- decision threshold: {_display(manifest.threshold)}

## Decision quality

- accuracy: {_display(metrics.accuracy)}
- high-risk false approvals: {metrics.high_risk_false_approvals}

## Selective decisions

- coverage: {metrics.coverage}
- selective accuracy: {_display(metrics.selective_accuracy)}

## Probability quality

- Brier score: {_display(metrics.brier_score)}
- probability samples: {metrics.probability_sample_count}

## Runtime and cost

- P50 latency ms: {_display(metrics.p50_latency_ms)}
- P95 latency ms: {_display(metrics.p95_latency_ms)}
- known total cost USD: {_display(metrics.total_known_cost_usd)}

## Local model provenance

- model artifact sha256: {_display(manifest.model_artifact_sha256)}
- training samples: {_display(manifest.training_sample_count)}
- training duration ms: {_display(manifest.training_duration_ms)}
- model size bytes: {_display(manifest.model_size_bytes)}

## Failures

{json.dumps(failures, ensure_ascii=False, sort_keys=True)}

## Limitations

This report describes `{manifest.run_mode.value}` evidence only. It does not authorize automated actions.
"""
    public_payload: dict[str, JSONValue] = {
        "run_id": manifest.run_id,
        "provider": manifest.provider,
        "model_version": manifest.model_version,
        "run_mode": manifest.run_mode.value,
        "dataset_sha256": manifest.dataset_sha256,
        "git_commit": manifest.git_commit,
        "created_at": manifest.created_at.isoformat(),
        "threshold": manifest.threshold,
        "metrics": metrics.model_dump(mode="json"),
        "failure_categories": dict(failures),
        "runtime_provenance": (
            manifest.runtime.model_dump(mode="json") if manifest.runtime is not None else None
        ),
        "local_model_provenance": {
            "model_artifact_sha256": manifest.model_artifact_sha256,
            "training_sample_count": manifest.training_sample_count,
            "training_duration_ms": manifest.training_duration_ms,
            "model_size_bytes": manifest.model_size_bytes,
        },
    }
    assert_public_safe(public_payload)
    report_dir.mkdir(parents=True, exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = report_dir / f"{manifest.run_id}.md"
    public_path = public_dir / f"{manifest.run_id}.json"
    markdown_path.write_text(markdown, encoding="utf-8")
    public_path.write_text(
        json.dumps(public_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return markdown_path, public_path
