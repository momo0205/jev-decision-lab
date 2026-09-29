from __future__ import annotations

import math
from importlib import import_module
from time import perf_counter
from typing import Protocol, cast

from jev_lab.contracts import DecisionResult, RequestStatus, RouteLabel, RoutingSample, RunMode

MODEL_ID = "convaiinnovations/laya-multilingual"
MODEL_REVISION = "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67"
MODEL_VERSION = f"{MODEL_ID}@{MODEL_REVISION}"
_PROBABILITY_SUM_TOLERANCE = 5e-4

ROUTE_QUESTIONS: dict[str, dict[str, object]] = {
    "route": {
        "type": "choice",
        "instructions": "应由哪个执行器负责这个请求？",
        "criteria": {
            "search": "检索和核验外部公开信息",
            "code": "阅读、修改或测试代码",
            "database": "只读查询结构化数据库",
            "human_review": "涉及高风险或不可逆操作，转交人工复核",
        },
    }
}


class LayaDependencyError(RuntimeError):
    """The optional Laya runtime is not installed or could not be loaded."""


class LayaPredictor(Protocol):
    def predict(
        self, state: str, questions: dict[str, dict[str, object]], *, lang: str
    ) -> dict[str, object]: ...


class LayaAgentFactory(Protocol):
    def __call__(self, model_id: str, *, revision: str) -> LayaPredictor: ...


class LayaProvider:
    name = "laya"
    model_version = MODEL_VERSION
    run_mode = RunMode.OFFLINE_DEVELOPMENT

    def __init__(self, agent: LayaPredictor) -> None:
        self.agent = agent

    @classmethod
    def from_default_checkpoint(cls) -> LayaProvider:
        try:
            laya_module = import_module("laya")
        except ImportError as exc:
            raise LayaDependencyError(
                "Laya is optional; install it with `python -m pip install -e '.[laya]'`."
            ) from exc

        agent_factory = getattr(laya_module, "Agent", None)
        if not callable(agent_factory):
            raise LayaDependencyError(
                "The installed Laya package does not expose Agent; install the pinned `.[laya]` extra."
            )
        try:
            agent = cast(LayaAgentFactory, agent_factory)(MODEL_ID, revision=MODEL_REVISION)
        except Exception as exc:
            raise LayaDependencyError(
                "Could not load the pinned Laya checkpoint. Check the optional install and model cache/network."
            ) from exc
        return cls(agent)

    def decide(self, sample: RoutingSample) -> DecisionResult:
        started = perf_counter()
        try:
            output = self.agent.predict(sample.input, ROUTE_QUESTIONS, lang="zh")
            label, probabilities = self._parse_output(output)
            return DecisionResult(
                sample_id=sample.sample_id,
                label=label,
                probabilities=probabilities,
                abstained=False,
                latency_ms=(perf_counter() - started) * 1000,
                estimated_cost_usd=None,
                provider=self.name,
                model_version=self.model_version,
                request_status=RequestStatus.SUCCESS,
                error=None,
                run_mode=self.run_mode,
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            return self._failed(sample, started, "invalid_model_output")
        except Exception:  # noqa: BLE001 - preserve the run without provider fallback
            return self._failed(sample, started, "prediction_error")

    @staticmethod
    def _parse_output(output: object) -> tuple[RouteLabel, dict[RouteLabel, float]]:
        if not isinstance(output, dict):
            raise TypeError("Laya response must be an object")
        answers = output.get("answers")
        if not isinstance(answers, dict):
            raise TypeError("Laya response has no answers")
        route = answers.get("route")
        if not isinstance(route, dict) or route.get("type") != "choice":
            raise ValueError("Laya response has no route choice")
        label = RouteLabel(route.get("choice"))
        raw_probabilities = route.get("probabilities")
        if not isinstance(raw_probabilities, dict) or set(raw_probabilities) != {
            item.value for item in RouteLabel
        }:
            raise ValueError("Laya response must contain every route probability")

        values: dict[RouteLabel, float] = {}
        for key, raw_value in raw_probabilities.items():
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise TypeError("Laya probabilities must be numeric")
            value = float(raw_value)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Laya probabilities must be finite and in [0, 1]")
            values[RouteLabel(key)] = value

        total = sum(values.values())
        if total <= 0 or abs(total - 1.0) > _PROBABILITY_SUM_TOLERANCE:
            raise ValueError("Laya probabilities do not sum to one")
        if values[label] + _PROBABILITY_SUM_TOLERANCE < max(values.values()):
            raise ValueError("Laya choice does not match its probabilities")

        return label, {key: value / total for key, value in values.items()}

    def _failed(self, sample: RoutingSample, started: float, error: str) -> DecisionResult:
        return DecisionResult(
            sample_id=sample.sample_id,
            label=None,
            probabilities=None,
            abstained=True,
            latency_ms=(perf_counter() - started) * 1000,
            estimated_cost_usd=None,
            provider=self.name,
            model_version=self.model_version,
            request_status=RequestStatus.FAILED,
            error=error,
            run_mode=self.run_mode,
        )
