from __future__ import annotations

import json
from pathlib import Path

from jev_lab.contracts import DecisionResult, RequestStatus, RoutingSample, RunManifest
from jev_lab.providers.base import DecisionProvider


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def run_experiment(
    provider: DecisionProvider,
    samples: list[RoutingSample],
    output_dir: Path,
    manifest: RunManifest,
) -> Path:
    run_dir = output_dir / manifest.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    results: list[DecisionResult] = []
    for sample in samples:
        try:
            result = provider.decide(sample)
        except Exception as exc:  # noqa: BLE001 - provider boundary must preserve the run
            result = DecisionResult(
                sample_id=sample.sample_id,
                label=None,
                probabilities=None,
                abstained=True,
                latency_ms=0.0,
                estimated_cost_usd=None,
                provider=provider.name,
                model_version=provider.model_version,
                request_status=RequestStatus.FAILED,
                error=type(exc).__name__,
                run_mode=provider.run_mode,
            )
        results.append(result)
    if results and not any(result.request_status == RequestStatus.SUCCESS for result in results):
        manifest = manifest.model_copy(update={"run_mode": results[0].run_mode})
    _atomic_write(run_dir / "manifest.json", manifest.model_dump_json(indent=2))
    payload = "".join(
        json.dumps(row.model_dump(mode="json"), ensure_ascii=False) + "\n" for row in results
    )
    artifact = run_dir / "results.jsonl"
    _atomic_write(artifact, payload)
    return artifact
