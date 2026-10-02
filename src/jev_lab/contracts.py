from __future__ import annotations

import math
from datetime import datetime
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StringEnum(str, Enum):
    """String enum with stable JSON values."""


class RouteLabel(StringEnum):
    SEARCH = "search"
    CODE = "code"
    DATABASE = "database"
    HUMAN_REVIEW = "human_review"


class Split(StringEnum):
    DEV = "dev"
    CALIBRATION = "calibration"
    TEST = "test"


class RunMode(StringEnum):
    OFFLINE_DEVELOPMENT = "offline-development"
    RECORDED_REPLAY = "recorded-replay"
    LIVE_EVALUATION = "live-evaluation"


class RequestStatus(StringEnum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class RoutingSample(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    sample_id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    input: str = Field(min_length=1)
    expected: RouteLabel
    acceptable: list[RouteLabel] = Field(min_length=1)
    risk: Literal["low", "medium", "high"]
    difficulty: Literal["clear", "ambiguous", "adversarial"]
    rationale: str = Field(min_length=1)
    source: Literal["synthetic", "redacted"]
    split: Split

    @model_validator(mode="after")
    def expected_is_acceptable(self) -> RoutingSample:
        if self.expected not in self.acceptable:
            raise ValueError("expected label must be included in acceptable labels")
        if len(set(self.acceptable)) != len(self.acceptable):
            raise ValueError("acceptable labels must be unique")
        return self


class DecisionResult(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    sample_id: str = Field(min_length=1)
    label: RouteLabel | None
    probabilities: dict[RouteLabel, float] | None
    abstained: bool
    latency_ms: float = Field(ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    provider: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    request_status: RequestStatus
    error: str | None
    run_mode: RunMode

    @model_validator(mode="after")
    def validate_probabilities(self) -> DecisionResult:
        if self.probabilities is not None:
            if not self.probabilities:
                raise ValueError("probabilities cannot be empty")
            if any(not math.isfinite(value) for value in self.probabilities.values()):
                raise ValueError("probabilities must be finite")
            if any(value < 0 or value > 1 for value in self.probabilities.values()):
                raise ValueError("probabilities must be in [0, 1]")
            if abs(sum(self.probabilities.values()) - 1.0) > 1e-6:
                raise ValueError("probabilities must sum to one")
        return self


class RuntimeProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    python_version: str = Field(
        min_length=1,
        max_length=32,
        pattern=r"^[0-9]+\.[0-9]+(\.[0-9]+)?([A-Za-z0-9.+_-]{0,24})$",
    )
    system: Literal["Darwin", "Linux", "Windows", "Other"]
    machine: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    package_versions: dict[
        Literal["laya", "torch", "transformers"],
        Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9.!+_-]{0,79}$")],
    ] = Field(default_factory=dict)
    inference_device: str | None = Field(
        default=None,
        pattern=r"^(cpu(:[0-9]+)?|mps(:[0-9]+)?|cuda(:[0-9]+)?|xpu(:[0-9]+)?|npu(:[0-9]+)?|hpu(:[0-9]+)?|privateuseone(:[0-9]+)?)$",
    )


class RunManifest(BaseModel):
    run_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    run_mode: RunMode
    split: Split
    dataset_path: str
    dataset_sha256: str
    git_commit: str
    threshold: float | None = Field(default=None, ge=0, le=1)
    model_artifact_sha256: str | None = None
    training_sample_count: int | None = Field(default=None, ge=1)
    training_duration_ms: float | None = Field(default=None, ge=0)
    model_size_bytes: int | None = Field(default=None, ge=1)
    created_at: datetime
    runtime: RuntimeProvenance | None = None
