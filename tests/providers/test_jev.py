from jev_lab.contracts import RequestStatus, RouteLabel, RoutingSample, Split
from jev_lab.providers.jev import JevProvider


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
