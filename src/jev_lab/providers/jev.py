from __future__ import annotations

import os
from collections.abc import Callable
from time import perf_counter
from typing import Protocol

import httpx

from jev_lab.contracts import DecisionResult, RequestStatus, RouteLabel, RoutingSample, RunMode


class JevClient(Protocol):
    def choose(
        self, state: dict[str, object], question: dict[str, object]
    ) -> dict[str, object]: ...


def _default_client_factory(api_key: str) -> JevClient:
    try:
        from typesafe_sdk import TypeSafeClient  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ModuleNotFoundError("typesafe-sdk") from exc
    return TypeSafeClient(api_key=api_key)  # type: ignore[no-any-return]


class JevProvider:
    name = "jev"
    model_version = "jev-early-access"
    run_mode = RunMode.LIVE_EVALUATION

    def __init__(
        self,
        api_key: str | None,
        client: JevClient | None = None,
        client_factory: Callable[[str], JevClient] = _default_client_factory,
        max_retries: int = 1,
    ) -> None:
        self.api_key = api_key
        self.client = client
        self.client_factory = client_factory
        self.max_retries = max_retries

    @classmethod
    def from_environment(
        cls, client_factory: Callable[[str], JevClient] = _default_client_factory
    ) -> JevProvider:
        return cls(os.environ.get("TYPESAFE_API_KEY") or None, client_factory=client_factory)

    def decide(self, sample: RoutingSample) -> DecisionResult:
        if not self.api_key:
            return self._result(sample, RequestStatus.SKIPPED, "missing_credentials")
        if self.client is None:
            try:
                self.client = self.client_factory(self.api_key)
            except ModuleNotFoundError:
                return self._result(sample, RequestStatus.SKIPPED, "dependency_missing")
        started = perf_counter()
        for attempt in range(self.max_retries + 1):
            try:
                raw = self.client.choose(
                    {"input": sample.input},
                    {"route": {label.value: label.value for label in RouteLabel}},
                )
                label = RouteLabel(str(raw["choice"]))
                raw_probabilities = raw.get("probabilities")
                probabilities = None
                if raw_probabilities is not None:
                    if not isinstance(raw_probabilities, dict) or set(raw_probabilities) != {
                        label.value for label in RouteLabel
                    }:
                        raise ValueError
                    probabilities = {
                        RouteLabel(str(key)): float(value)
                        for key, value in raw_probabilities.items()
                    }
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
            except (httpx.TransportError, TimeoutError):
                if attempt < self.max_retries:
                    continue
                return self._result(sample, RequestStatus.FAILED, "transport_error", started)
            except (KeyError, TypeError, ValueError):
                return self._result(sample, RequestStatus.FAILED, "invalid_response", started)
        raise AssertionError("unreachable")

    def _result(
        self, sample: RoutingSample, status: RequestStatus, error: str, started: float | None = None
    ) -> DecisionResult:
        return DecisionResult(
            sample_id=sample.sample_id,
            label=None,
            probabilities=None,
            abstained=True,
            latency_ms=(perf_counter() - started) * 1000 if started else 0,
            estimated_cost_usd=None,
            provider=self.name,
            model_version=self.model_version,
            request_status=status,
            error=error,
            run_mode=self.run_mode,
        )
