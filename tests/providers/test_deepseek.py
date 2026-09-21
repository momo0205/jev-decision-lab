from jev_lab.contracts import RequestStatus, RouteLabel, RoutingSample, Split
from jev_lab.providers.deepseek import DeepSeekProvider


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


class FakeClient:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls = 0

    def complete(self, payload: dict[str, object], timeout_seconds: float) -> dict[str, object]:
        self.calls += 1
        return self.response


def test_missing_key_is_skipped_without_network(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    client = FakeClient({"label": "search"})
    result = DeepSeekProvider.from_environment(client=client).decide(sample_with("查天气"))
    assert result.request_status == RequestStatus.SKIPPED
    assert result.error == "missing_credentials"
    assert client.calls == 0


def test_rejects_malformed_probabilities() -> None:
    provider = DeepSeekProvider(
        "key", client=FakeClient({"label": "search", "probabilities": {"search": 1.2}})
    )
    result = provider.decide(sample_with("查天气"))
    assert result.request_status == RequestStatus.FAILED
    assert result.error == "invalid_response"


def test_valid_response_has_unknown_cost() -> None:
    provider = DeepSeekProvider("key", client=FakeClient({"label": "search"}))
    result = provider.decide(sample_with("查天气"))
    assert result.label == RouteLabel.SEARCH
    assert result.estimated_cost_usd is None
