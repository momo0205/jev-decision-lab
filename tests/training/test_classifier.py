from pathlib import Path

import pytest

from jev_lab.contracts import RouteLabel, Split
from jev_lab.dataset import load_dataset
from jev_lab.training.artifacts import load_verified_pipeline
from jev_lab.training.classifier import cross_validate_classifier, train_classifier

DATASET = Path("datasets/routing-v1.yaml")


def dev_samples():  # type: ignore[no-untyped-def]
    return [sample for sample in load_dataset(DATASET) if sample.split == Split.DEV]


def test_training_rejects_non_dev_before_writing(tmp_path: Path) -> None:
    samples = dev_samples()
    samples.append(
        samples[0].model_copy(update={"sample_id": "cal-1", "split": Split.CALIBRATION})
    )
    output = tmp_path / "artifacts"

    with pytest.raises(ValueError, match="dev samples only"):
        train_classifier(samples, DATASET, output, "m1", "abc")

    assert not output.exists()


def test_cross_validation_keeps_families_in_one_fold() -> None:
    result = cross_validate_classifier(dev_samples(), folds=3, random_seed=42)

    assert len(result.folds) == 3
    for fold in result.folds:
        assert set(fold.training_family_ids).isdisjoint(fold.validation_family_ids)


def test_same_seed_produces_same_cross_validation_scores() -> None:
    first = cross_validate_classifier(dev_samples(), folds=3, random_seed=42)
    second = cross_validate_classifier(dev_samples(), folds=3, random_seed=42)

    assert first == second


def test_cross_validation_rejects_more_folds_than_smallest_label_family_count() -> None:
    with pytest.raises(ValueError, match="fold count exceeds"):
        cross_validate_classifier(dev_samples(), folds=4)


def test_training_writes_loadable_complete_label_model(tmp_path: Path) -> None:
    model_dir = train_classifier(dev_samples(), DATASET, tmp_path, "m1", "abc")

    pipeline, manifest = load_verified_pipeline(model_dir)

    assert pipeline is not None
    assert manifest.labels == list(RouteLabel)
    assert len(manifest.training_sample_ids) == 60
    assert len(manifest.training_family_ids) == 15
