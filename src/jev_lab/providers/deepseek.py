from __future__ import annotations

import json
import os
from time import perf_counter
from typing import Protocol, cast

import httpx

from jev_lab.contracts import DecisionResult, RequestStatus, RouteLabel, RoutingSample, RunMode


class CompletionClient(Protocol):
    def complete(self, payload: dict[str, object], timeout_seconds: float) -> dict[str, object]: ...


class HttpxDeepSeekClient:
    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def complete(self, payload: dict[str, object], timeout_seconds: float) -> dict[str, object]:
        response = httpx.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                "response_format": {"type": "json_object"},
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        parsed = json.loads(body["choices"][0]["message"]["content"])
        if not isinstance(parsed, dict):
            raise TypeError("DeepSeek response must be an object")
        return cast(dict[str, object], parsed)


class DeepSeekProvider:
    name = "deepseek"

    def __init__(
        self,
        api_key: str | None,
        client: CompletionClient | None = None,
        model: str = "deepseek-chat",
        timeout_seconds: float = 20,
        max_retries: int = 1,
    ) -> None:
        self.api_key = api_key
        self.model_version = model
        self.client = (
            client
            if client is not None
            else (HttpxDeepSeekClient(api_key, model) if api_key else None)
        )
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    @property
    def run_mode(self) -> RunMode:
        return RunMode.LIVE_EVALUATION if self.api_key else RunMode.OFFLINE_DEVELOPMENT

    @classmethod
    def from_environment(cls, client: CompletionClient | None = None) -> DeepSeekProvider:
        return cls(os.environ.get("DEEPSEEK_API_KEY") or None, client=client)

    def decide(self, sample: RoutingSample) -> DecisionResult:
        if not self.api_key:
            return self._result(sample, RequestStatus.SKIPPED, error="missing_credentials")
        if self.client is None:
            return self._result(sample, RequestStatus.SKIPPED, error="dependency_missing")
        payload: dict[str, object] = {
            "input": sample.input,
            "labels": {label.value: label.value for label in RouteLabel},
            "instruction": "Return JSON with one label and optional complete probabilities.",
        }
        started = perf_counter()
        for attempt in range(self.max_retries + 1):
            try:
                raw = self.client.complete(payload, self.timeout_seconds)
                label = RouteLabel(str(raw["label"]))
                probability_raw = raw.get("probabilities")
                probabilities = None
                if probability_raw is not None:
                    if not isinstance(probability_raw, dict) or set(probability_raw) != {
                        item.value for item in RouteLabel
                    }:
                        raise ValueError
                    probabilities = {
                        RouteLabel(str(key)): float(value) for key, value in probability_raw.items()
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
            latency_ms=(perf_counter() - started) * 1000 if started else 0.0,
            estimated_cost_usd=None,
            provider=self.name,
            model_version=self.model_version,
            request_status=status,
            error=error,
            run_mode=self.run_mode,
        )
