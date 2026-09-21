import pytest

from jev_lab.contracts import (
    DecisionResult,
    RequestStatus,
    RouteLabel,
    RoutingSample,
    RunMode,
    Split,
)
from jev_lab.metrics import evaluate


def sample(sample_id: str, expected: RouteLabel, risk: str = "low") -> RoutingSample:
    return RoutingSample(
        sample_id=sample_id,
        family_id=sample_id,
        input="x",
        expected=expected,
        acceptable=[expected],
        risk=risk,
        difficulty="clear",
        rationale="x",
        source="synthetic",
        split=Split.DEV,
    )


def result(
    sample_id: str,
    label: RouteLabel | None,
    *,
    abstained: bool = False,
    probabilities: dict[RouteLabel, float] | None = None,
    latency: float = 10,
) -> DecisionResult:
    return DecisionResult(
        sample_id=sample_id,
        label=label,
        probabilities=probabilities,
        abstained=abstained,
        latency_ms=latency,
        estimated_cost_usd=None,
        provider="p",
        model_version="v",
        request_status=RequestStatus.SUCCESS,
        error=None,
        run_mode=RunMode.OFFLINE_DEVELOPMENT,
    )


def test_zero_coverage_is_defined_without_division_error() -> None:
    metrics = evaluate([sample("s", RouteLabel.SEARCH)], [result("s", None, abstained=True)])
    assert metrics.coverage == 0.0
    assert metrics.selective_accuracy is None


def test_known_confusion_matrix_and_accuracy() -> None:
    samples = [sample(str(i), label) for i, label in enumerate(RouteLabel)]
    results = [result("0", RouteLabel.CODE)] + [
        result(str(i), label) for i, label in list(enumerate(RouteLabel))[1:]
    ]
    metrics = evaluate(samples, results)
    assert metrics.accuracy == 0.75
    assert metrics.confusion_matrix["search"]["code"] == 1


def test_brier_ignores_results_without_probabilities() -> None:
    samples = [sample("a", RouteLabel.SEARCH), sample("b", RouteLabel.CODE)]
    probs = {label: (0.7 if label == RouteLabel.SEARCH else 0.1) for label in RouteLabel}
    metrics = evaluate(
        samples, [result("a", RouteLabel.SEARCH, probabilities=probs), result("b", RouteLabel.CODE)]
    )
    assert metrics.probability_sample_count == 1
    assert metrics.brier_score == pytest.approx(0.12)


def test_high_risk_wrong_automatic_label_is_counted() -> None:
    metrics = evaluate(
        [sample("s", RouteLabel.HUMAN_REVIEW, "high")], [result("s", RouteLabel.CODE)]
    )
    assert metrics.high_risk_false_approvals == 1
