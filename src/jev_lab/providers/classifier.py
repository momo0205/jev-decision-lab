from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Protocol, cast

from jev_lab.contracts import DecisionResult, RequestStatus, RouteLabel, RoutingSample, RunMode
from jev_lab.training.artifacts import ClassifierTrainingManifest, load_verified_pipeline


class PredictivePipeline(Protocol):
    classes_: Sequence[str]

    def predict_proba(self, values: list[str]) -> Sequence[Sequence[float]]: ...


class ClassifierProvider:
    name = "tfidf-logreg"
    run_mode = RunMode.OFFLINE_DEVELOPMENT

    def __init__(
        self, pipeline: PredictivePipeline, manifest: ClassifierTrainingManifest
    ) -> None:
        self.pipeline = pipeline
        self.manifest = manifest
        self.model_version = manifest.model_id

    @classmethod
    def from_artifact(cls, model_dir: Path) -> ClassifierProvider:
        pipeline, manifest = load_verified_pipeline(model_dir)
        return cls(cast(PredictivePipeline, pipeline), manifest)

    def decide(self, sample: RoutingSample) -> DecisionResult:
        started = perf_counter()
        try:
            classes = [RouteLabel(str(value)) for value in self.pipeline.classes_]
            if len(classes) != len(set(classes)) or set(classes) != set(RouteLabel):
                raise ValueError("invalid classes")
            rows = self.pipeline.predict_proba([sample.input])
            if len(rows) != 1 or len(rows[0]) != len(classes):
                raise ValueError("invalid probability shape")
            probabilities = {label: float(value) for label, value in zip(classes, rows[0])}
            if any(not math.isfinite(value) or value < 0 or value > 1 for value in probabilities.values()):
                raise ValueError("invalid probability")
            if abs(sum(probabilities.values()) - 1.0) > 1e-6:
                raise ValueError("probabilities must sum to one")
            selected = max(probabilities, key=probabilities.__getitem__)
            return DecisionResult(
                sample_id=sample.sample_id,
                label=selected,
                probabilities=probabilities,
                abstained=False,
                latency_ms=(perf_counter() - started) * 1000,
                estimated_cost_usd=0.0,
                provider=self.name,
                model_version=self.model_version,
                request_status=RequestStatus.SUCCESS,
                error=None,
                run_mode=self.run_mode,
            )
        except (KeyError, TypeError, ValueError, IndexError):
            return self._failed(sample, started, "invalid_model_output")
        except Exception:  # noqa: BLE001 - prediction boundary preserves the experiment
            return self._failed(sample, started, "prediction_error")

    def _failed(self, sample: RoutingSample, started: float, error: str) -> DecisionResult:
        return DecisionResult(
            sample_id=sample.sample_id,
            label=None,
            probabilities=None,
            abstained=True,
            latency_ms=(perf_counter() - started) * 1000,
            estimated_cost_usd=0.0,
            provider=self.name,
            model_version=self.model_version,
            request_status=RequestStatus.FAILED,
            error=error,
            run_mode=self.run_mode,
        )
