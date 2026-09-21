import pytest
from pydantic import ValidationError

from jev_lab.contracts import (
    DecisionResult,
    RequestStatus,
    RouteLabel,
    RoutingSample,
    RunMode,
    Split,
)


def test_routing_sample_rejects_expected_label_outside_acceptable() -> None:
    with pytest.raises(ValidationError):
        RoutingSample(
            sample_id="s1",
            family_id="f1",
            input="查资料",
            expected=RouteLabel.SEARCH,
            acceptable=[RouteLabel.CODE],
            risk="low",
            difficulty="clear",
            rationale="需要搜索",
            source="synthetic",
            split=Split.DEV,
        )


def test_decision_result_keeps_missing_probability_and_cost_explicit() -> None:
    result = DecisionResult(
        sample_id="s1",
        label=RouteLabel.SEARCH,
        probabilities=None,
        abstained=False,
        latency_ms=1.0,
        estimated_cost_usd=None,
        provider="rules",
        model_version="rules-v1",
        request_status=RequestStatus.SUCCESS,
        error=None,
        run_mode=RunMode.OFFLINE_DEVELOPMENT,
    )
    assert result.probabilities is None
    assert result.estimated_cost_usd is None


def test_decision_result_rejects_invalid_probability_sum() -> None:
    with pytest.raises(ValidationError):
        DecisionResult(
            sample_id="s1",
            label=RouteLabel.SEARCH,
            probabilities={RouteLabel.SEARCH: 0.6, RouteLabel.CODE: 0.2},
            abstained=False,
            latency_ms=1.0,
            estimated_cost_usd=None,
            provider="remote",
            model_version="v1",
            request_status=RequestStatus.SUCCESS,
            error=None,
            run_mode=RunMode.LIVE_EVALUATION,
        )


def test_routing_sample_rejects_duplicate_acceptable_labels() -> None:
    with pytest.raises(ValidationError, match="acceptable labels must be unique"):
        RoutingSample(
            sample_id="s1",
            family_id="f1",
            input="查资料",
            expected=RouteLabel.SEARCH,
            acceptable=[RouteLabel.SEARCH, RouteLabel.SEARCH],
            risk="low",
            difficulty="clear",
            rationale="需要搜索",
            source="synthetic",
            split=Split.DEV,
        )
