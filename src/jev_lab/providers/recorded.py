from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from jev_lab.contracts import DecisionResult, RequestStatus, RouteLabel, RoutingSample, RunMode


class RecordedRow(BaseModel):
    sample_id: str
    provider: str
    model_version: str
    recorded_at: str = Field(min_length=1)
    source_run_hash: str = Field(min_length=1)
    label: RouteLabel
    probabilities: dict[RouteLabel, float] | None
    run_mode: RunMode


class RecordedProvider:
    run_mode = RunMode.RECORDED_REPLAY

    def __init__(self, path: Path, declared_provider: str, model_version: str) -> None:
        self.name = declared_provider
        self.model_version = model_version
        rows = [RecordedRow.model_validate_json(line) for line in path.read_text().splitlines()]
        if any(row.run_mode != RunMode.RECORDED_REPLAY for row in rows):
            raise ValueError("recorded fixture must declare recorded-replay")
        if any(
            row.provider != declared_provider or row.model_version != model_version for row in rows
        ):
            raise ValueError("recorded fixture provenance mismatch")
        ids = [row.sample_id for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate sample_id in recorded fixture")
        self.rows = {row.sample_id: row for row in rows}

    def decide(self, sample: RoutingSample) -> DecisionResult:
        if sample.sample_id not in self.rows:
            raise KeyError("unknown sample in recorded fixture")
        row = self.rows[sample.sample_id]
        return DecisionResult(
            sample_id=sample.sample_id,
            label=row.label,
            probabilities=row.probabilities,
            abstained=False,
            latency_ms=0,
            estimated_cost_usd=None,
            provider=self.name,
            model_version=self.model_version,
            request_status=RequestStatus.SUCCESS,
            error=None,
            run_mode=self.run_mode,
        )
