from __future__ import annotations

from types import SimpleNamespace

import pytest

from jev_lab.contracts import RequestStatus, RouteLabel, RoutingSample, RunMode, Split
from jev_lab.providers.laya import LayaDependencyError, LayaProvider


class FakeLayaAgent:
    def __init__(self, result: dict[str, object] | None = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[tuple[object, object, str | None]] = []

    def predict(self, state: object, questions: object, lang: str | None = None) -> dict[str, object]:
        self.calls.append((state, questions, lang))
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def sample() -> RoutingSample:
    return RoutingSample(
        sample_id="route-search-001",
        family_id="family-search-001",
        input="查一下公开资料",
        expected=RouteLabel.SEARCH,
        acceptable=[RouteLabel.SEARCH],
        risk="low",
        difficulty="clear",
        rationale="需要检索公开信息",
        source="synthetic",
        split=Split.DEV,
    )


def answer(
    *,
    choice: str = "search",
    probabilities: dict[str, float] | None = None,
) -> dict[str, object]:
    return {
        "model": "laya-rl-agent",
        "answers": {
            "route": {
                "type": "choice",
                "choice": choice,
                "probabilities": probabilities
                or {"search": 0.4, "code": 0.3, "database": 0.2, "human_review": 0.1001},
                "confidence": 0.4,
                "answer_confidence": 0.4,
                "action": {"act_probability": 1.0},
            }
        },
        "usage": {"input_tokens": 12, "output_tokens": 0},
    }


def test_provider_maps_choice_and_normalizes_rounded_probabilities() -> None:
    agent = FakeLayaAgent(answer())
    result = LayaProvider(agent).decide(sample())

    assert result.request_status == RequestStatus.SUCCESS
    assert result.label == RouteLabel.SEARCH
    assert result.probabilities is not None
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    assert result.probabilities[RouteLabel.SEARCH] == pytest.approx(0.4 / 1.0001)
    assert result.run_mode == RunMode.OFFLINE_DEVELOPMENT
    assert result.estimated_cost_usd is None
    assert agent.calls[0][0] == "查一下公开资料"
    assert agent.calls[0][2] == "zh"


@pytest.mark.parametrize(
    "payload",
    [
        {"model": "laya-rl-agent", "answers": {}},
        answer(probabilities={"search": 0.7, "code": 0.3}),
        answer(probabilities={"search": 0.7, "code": 0.4, "database": 0.0, "human_review": 0.0}),
        answer(choice="not-a-route"),
    ],
)
def test_provider_rejects_invalid_model_output_without_fallback(payload: dict[str, object]) -> None:
    result = LayaProvider(FakeLayaAgent(payload)).decide(sample())

    assert result.request_status == RequestStatus.FAILED
    assert result.label is None
    assert result.probabilities is None
    assert result.abstained is True
    assert result.error == "invalid_model_output"
    assert result.provider == "laya"


def test_provider_sanitizes_inference_exception() -> None:
    result = LayaProvider(FakeLayaAgent(error=RuntimeError("private input and token=secret"))).decide(
        sample()
    )

    assert result.request_status == RequestStatus.FAILED
    assert result.error == "prediction_error"
    assert "secret" not in (result.error or "")
    assert result.label is None


def test_default_checkpoint_loader_uses_pinned_model_and_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded: dict[str, object] = {}

    class FakeAgent:
        def __init__(self, model_id: str, *, revision: str) -> None:
            loaded["model_id"] = model_id
            loaded["revision"] = revision

    module = SimpleNamespace(Agent=FakeAgent)
    monkeypatch.setattr("jev_lab.providers.laya.import_module", lambda name: module)

    provider = LayaProvider.from_default_checkpoint()

    assert loaded == {
        "model_id": "convaiinnovations/laya-multilingual",
        "revision": "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67",
    }
    assert provider.model_version.endswith("e4e9ddf21a7b1903b7acffd8814ad4307bf63a67")


def test_missing_optional_package_has_install_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_package(name: str) -> object:
        assert name == "laya"
        raise ModuleNotFoundError("no module named laya")

    monkeypatch.setattr("jev_lab.providers.laya.import_module", missing_package)

    with pytest.raises(LayaDependencyError, match=r"\[laya\]"):
        LayaProvider.from_default_checkpoint()
