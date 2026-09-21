from types import SimpleNamespace

from jev_lab.contracts import RequestStatus, RouteLabel, RoutingSample, RunMode, Split
from jev_lab.providers.jev import JevProvider, TypeSafeSDKAdapter


def sample_with(text: str) -> RoutingSample:
    return RoutingSample(
        sample_id="s",
        family_id="f",
        input=text,
        expected=RouteLabel.SEARCH,
        acceptable=[RouteLabel.SEARCH],
        risk="low",
        difficulty="clear",
        rationale="x",
        source="synthetic",
        split=Split.DEV,
    )


def test_jev_missing_access_is_explicitly_skipped(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    result = JevProvider.from_environment(
        client_factory=lambda key: (_ for _ in ()).throw(AssertionError())
    ).decide(sample_with("查天气"))
    assert result.request_status == RequestStatus.SKIPPED
    assert result.error == "missing_credentials"


def test_jev_invalid_probability_map_fails_closed() -> None:
    class Client:
        def choose(self, state, question):  # type: ignore[no-untyped-def]
            return {"choice": "search", "probabilities": {"other": 1.0}}

    result = JevProvider("key", client=Client()).decide(sample_with("查天气"))
    assert result.error == "invalid_response"


def test_jev_valid_choice_is_normalized() -> None:
    class Client:
        def choose(self, state, question):  # type: ignore[no-untyped-def]
            return {"choice": "search"}

    result = JevProvider("key", client=Client()).decide(sample_with("查天气"))
    assert result.label == RouteLabel.SEARCH
    assert result.estimated_cost_usd is None


def test_sdk_adapter_uses_system_one_choice_surface() -> None:
    class RawClient:
        def system_one(self, *, state, questions):  # type: ignore[no-untyped-def]
            assert "route" in questions
            return SimpleNamespace(choices={"route": SimpleNamespace(choice="search")})

    assert TypeSafeSDKAdapter(RawClient()).choose(
        {"input": "查天气"}, {"route": {"search": "检索"}}
    ) == {"choice": "search", "probabilities": None}


def test_jev_missing_credentials_are_not_live_evidence(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    result = JevProvider.from_environment().decide(sample_with("查天气"))
    assert result.run_mode == RunMode.OFFLINE_DEVELOPMENT


def test_jev_nan_probabilities_fail_closed() -> None:
    class Client:
        def choose(self, state, question):  # type: ignore[no-untyped-def]
            return {
                "choice": "search",
                "probabilities": {label.value: "NaN" for label in RouteLabel},
            }

    assert (
        JevProvider("key", client=Client()).decide(sample_with("查天气")).error
        == "invalid_response"
    )
