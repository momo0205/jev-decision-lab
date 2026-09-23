from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import sklearn  # type: ignore[import-untyped]
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import accuracy_score  # type: ignore[import-untyped]
from sklearn.model_selection import StratifiedGroupKFold  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]

from jev_lab.contracts import RouteLabel, RoutingSample, Split
from jev_lab.dataset import dataset_sha256
from jev_lab.training.artifacts import ClassifierArtifactMetadata, write_artifact


class CrossValidationFold(BaseModel):
    fold_index: int = Field(ge=0)
    accuracy: float = Field(ge=0, le=1)
    training_sample_ids: list[str]
    validation_sample_ids: list[str]
    training_family_ids: list[str]
    validation_family_ids: list[str]


class CrossValidationResult(BaseModel):
    folds: list[CrossValidationFold]
    mean_accuracy: float = Field(ge=0, le=1)
    standard_deviation: float = Field(ge=0)
    random_seed: int
    fold_count: int = Field(ge=2)


def validate_training_samples(samples: list[RoutingSample]) -> None:
    if not samples:
        raise ValueError("training samples cannot be empty")
    if any(sample.split != Split.DEV for sample in samples):
        raise ValueError("dev samples only")
    ids = [sample.sample_id for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate training sample id")
    present = {sample.expected for sample in samples}
    if present != set(RouteLabel):
        raise ValueError("training samples must contain every route label")


def _pipeline(random_seed: int) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000, class_weight="balanced", random_state=random_seed
                ),
            ),
        ]
    )


def cross_validate_classifier(
    samples: list[RoutingSample], folds: int = 3, random_seed: int = 42
) -> CrossValidationResult:
    validate_training_samples(samples)
    families_by_label: dict[RouteLabel, set[str]] = {label: set() for label in RouteLabel}
    for sample in samples:
        families_by_label[sample.expected].add(sample.family_id)
    if folds < 2:
        raise ValueError("fold count must be at least two")
    if folds > min(len(families) for families in families_by_label.values()):
        raise ValueError("fold count exceeds smallest per-label family count")
    texts = [sample.input for sample in samples]
    labels = [sample.expected.value for sample in samples]
    groups = [sample.family_id for sample in samples]
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=random_seed)
    rows: list[CrossValidationFold] = []
    for index, (training, validation) in enumerate(splitter.split(texts, labels, groups)):
        model = _pipeline(random_seed)
        model.fit([texts[i] for i in training], [labels[i] for i in training])
        predicted = model.predict([texts[i] for i in validation])
        rows.append(
            CrossValidationFold(
                fold_index=index,
                accuracy=float(accuracy_score([labels[i] for i in validation], predicted)),
                training_sample_ids=[samples[i].sample_id for i in training],
                validation_sample_ids=[samples[i].sample_id for i in validation],
                training_family_ids=sorted({groups[i] for i in training}),
                validation_family_ids=sorted({groups[i] for i in validation}),
            )
        )
    scores = [row.accuracy for row in rows]
    mean = sum(scores) / len(scores)
    deviation = math.sqrt(sum((score - mean) ** 2 for score in scores) / len(scores))
    return CrossValidationResult(
        folds=rows,
        mean_accuracy=mean,
        standard_deviation=deviation,
        random_seed=random_seed,
        fold_count=folds,
    )


def train_classifier(
    samples: list[RoutingSample],
    dataset_path: Path,
    output_dir: Path,
    model_id: str,
    git_commit: str,
    random_seed: int = 42,
) -> Path:
    validate_training_samples(samples)
    started = perf_counter()
    model = _pipeline(random_seed)
    model.fit([sample.input for sample in samples], [sample.expected.value for sample in samples])
    duration_ms = (perf_counter() - started) * 1000
    metadata = ClassifierArtifactMetadata(
        model_id=model_id,
        created_at=datetime.now(UTC),
        dataset_path=str(dataset_path),
        dataset_sha256=dataset_sha256(dataset_path),
        training_sample_ids=[sample.sample_id for sample in samples],
        training_family_ids=sorted({sample.family_id for sample in samples}),
        labels=list(RouteLabel),
        sklearn_version=sklearn.__version__,
        tfidf_params={"analyzer": "char", "ngram_min": 2, "ngram_max": 5, "min_df": 1},
        classifier_params={"max_iter": 1000, "class_weight": "balanced"},
        random_seed=random_seed,
        training_duration_ms=duration_ms,
        git_commit=git_commit,
    )
    return write_artifact(pipeline=model, metadata=metadata, output_dir=output_dir)
