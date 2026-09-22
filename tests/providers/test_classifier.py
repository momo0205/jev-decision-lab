from datetime import UTC, datetime

from jev_lab.contracts import RequestStatus, RouteLabel, RoutingSample, RunMode, Split
from jev_lab.providers.classifier import ClassifierProvider
from jev_lab.training.artifacts import ClassifierTrainingManifest


class FakePipeline:
    def __init__(self, classes: list[str], probabilities: list[list[float]]) -> None:
        self.classes_ = classes
        self.probabilities = probabilities

    def predict_proba(self, values: list[str]) -> list[list[float]]:
        assert values
        return self.probabilities


def manifest() -> ClassifierTrainingManifest:
    return ClassifierTrainingManifest(
        model_id="m1",
        created_at=datetime.now(UTC),
        dataset_path="data.yaml",
        dataset_sha256="hash",
        training_sample_ids=["s1"],
        training_family_ids=["f1"],
        labels=list(RouteLabel),
        sklearn_version="1.9",
        tfidf_params={},
        classifier_params={},
        random_seed=42,
        training_duration_ms=1,
        git_commit="abc",
        model_size_bytes=1,
        model_sha256="model-hash",
    )


def sample() -> RoutingSample:
    return RoutingSample(
        sample_id="s1",
        family_id="f1",
        input="查一下资料",
        expected=RouteLabel.SEARCH,
        acceptable=[RouteLabel.SEARCH],
        risk="low",
        difficulty="clear",
        rationale="search",
        source="synthetic",
        split=Split.DEV,
    )


def test_provider_maps_probabilities_using_pipeline_classes() -> None:
    pipeline = FakePipeline(
        classes=["human_review", "search", "database", "code"],
        probabilities=[[0.1, 0.6, 0.2, 0.1]],
    )
    result = ClassifierProvider(pipeline, manifest()).decide(sample())

    assert result.label == RouteLabel.SEARCH
    assert result.probabilities == {
        RouteLabel.HUMAN_REVIEW: 0.1,
        RouteLabel.SEARCH: 0.6,
        RouteLabel.DATABASE: 0.2,
        RouteLabel.CODE: 0.1,
    }
    assert result.run_mode == RunMode.OFFLINE_DEVELOPMENT
    assert result.estimated_cost_usd == 0.0


def test_provider_rejects_incomplete_class_set() -> None:
    pipeline = FakePipeline(classes=["search", "code"], probabilities=[[0.7, 0.3]])

    result = ClassifierProvider(pipeline, manifest()).decide(sample())

    assert result.request_status == RequestStatus.FAILED
    assert result.error == "invalid_model_output"
    assert result.abstained is True
