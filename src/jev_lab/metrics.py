from __future__ import annotations

import math
from collections import defaultdict

from pydantic import BaseModel

from jev_lab.contracts import DecisionResult, RequestStatus, RouteLabel, RoutingSample


class ClassMetrics(BaseModel):
    precision: float | None
    recall: float | None


class CalibrationBucket(BaseModel):
    lower: float
    upper: float
    count: int
    mean_confidence: float | None
    accuracy: float | None


class EvaluationMetrics(BaseModel):
    accuracy: float | None
    coverage: float
    selective_accuracy: float | None
    success_rate: float
    confusion_matrix: dict[str, dict[str, int]]
    per_class: dict[str, ClassMetrics]
    high_risk_false_approvals: int
    probability_sample_count: int
    brier_score: float | None
    calibration: list[CalibrationBucket]
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    total_known_cost_usd: float | None
    mean_known_cost_usd: float | None


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def evaluate(
    samples: list[RoutingSample], results: list[DecisionResult], threshold: float | None = None
) -> EvaluationMetrics:
    by_id = {result.sample_id: result for result in results}
    labels = [label.value for label in RouteLabel]
    confusion = {expected: {actual: 0 for actual in labels} for expected in labels}
    decided: list[tuple[RoutingSample, DecisionResult]] = []
    accepted: list[tuple[RoutingSample, DecisionResult]] = []
    successes = 0
    high_risk_false_approvals = 0
    probability_rows: list[tuple[RoutingSample, DecisionResult]] = []

    for sample in samples:
        current = by_id.get(sample.sample_id)
        if current is None or current.request_status != RequestStatus.SUCCESS:
            continue
        successes += 1
        confidence = max(current.probabilities.values()) if current.probabilities else None
        if current.abstained or current.label is None:
            continue
        decided.append((sample, current))
        confusion[sample.expected.value][current.label.value] += 1
        if (
            sample.risk == "high"
            and sample.expected == RouteLabel.HUMAN_REVIEW
            and current.label != RouteLabel.HUMAN_REVIEW
        ):
            high_risk_false_approvals += 1
        if current.probabilities is not None and set(current.probabilities) == set(RouteLabel):
            probability_rows.append((sample, current))
        thresholded = threshold is not None and (confidence is None or confidence < threshold)
        if not thresholded:
            accepted.append((sample, current))

    baseline_correct = sum(result.label in sample.acceptable for sample, result in decided)
    accuracy = baseline_correct / len(decided) if decided else None
    accepted_correct = sum(result.label in sample.acceptable for sample, result in accepted)
    selective_accuracy = accepted_correct / len(accepted) if accepted else None
    per_class: dict[str, ClassMetrics] = {}
    for label in labels:
        true_positive = confusion[label][label]
        predicted = sum(confusion[expected][label] for expected in labels)
        actual = sum(confusion[label].values())
        per_class[label] = ClassMetrics(
            precision=true_positive / predicted if predicted else None,
            recall=true_positive / actual if actual else None,
        )

    brier_values: list[float] = []
    buckets: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    for sample, current in probability_rows:
        assert current.probabilities is not None
        brier_values.append(
            sum(
                (current.probabilities[label] - (1.0 if label == sample.expected else 0.0)) ** 2
                for label in RouteLabel
            )
        )
        confidence = max(current.probabilities.values())
        bucket = min(int(confidence * 5), 4)
        buckets[bucket].append((confidence, current.label in sample.acceptable))
    calibration = []
    for index in range(5):
        rows = buckets[index]
        calibration.append(
            CalibrationBucket(
                lower=index / 5,
                upper=(index + 1) / 5,
                count=len(rows),
                mean_confidence=sum(r[0] for r in rows) / len(rows) if rows else None,
                accuracy=sum(r[1] for r in rows) / len(rows) if rows else None,
            )
        )

    successful_results = [
        result for result in results if result.request_status == RequestStatus.SUCCESS
    ]
    latencies = [result.latency_ms for result in successful_results]
    costs = [
        result.estimated_cost_usd
        for result in successful_results
        if result.estimated_cost_usd is not None
    ]
    return EvaluationMetrics(
        accuracy=accuracy,
        coverage=len(accepted) / len(samples) if samples else 0.0,
        selective_accuracy=selective_accuracy,
        success_rate=successes / len(samples) if samples else 0.0,
        confusion_matrix=confusion,
        per_class=per_class,
        high_risk_false_approvals=high_risk_false_approvals,
        probability_sample_count=len(probability_rows),
        brier_score=sum(brier_values) / len(brier_values) if brier_values else None,
        calibration=calibration,
        p50_latency_ms=_percentile(latencies, 0.5),
        p95_latency_ms=_percentile(latencies, 0.95),
        total_known_cost_usd=sum(costs) if costs else None,
        mean_known_cost_usd=sum(costs) / len(costs) if costs else None,
    )
