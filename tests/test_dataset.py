from collections import Counter
from pathlib import Path

import pytest

from jev_lab.contracts import RouteLabel, RoutingSample, Split
from jev_lab.dataset import dataset_sha256, load_dataset, validate_dataset


def sample(sample_id: str, family_id: str, split: Split) -> RoutingSample:
    return RoutingSample(
        sample_id=sample_id,
        family_id=family_id,
        input="查询一项资料",
        expected=RouteLabel.SEARCH,
        acceptable=[RouteLabel.SEARCH],
        risk="low",
        difficulty="clear",
        rationale="需要外部检索",
        source="synthetic",
        split=split,
    )


def test_rejects_family_leakage_across_splits() -> None:
    samples = [sample("a", "shared", Split.DEV), sample("b", "shared", Split.TEST)]
    with pytest.raises(ValueError, match="family.*shared"):
        validate_dataset(samples, expected_size=2)


def test_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="duplicate sample_id"):
        validate_dataset([sample("a", "f1", Split.DEV), sample("a", "f2", Split.DEV)], 2)


def test_routing_v1_has_frozen_shape() -> None:
    samples = load_dataset(Path("datasets/routing-v1.yaml"))
    assert Counter(s.split for s in samples) == {
        Split.DEV: 60,
        Split.CALIBRATION: 20,
        Split.TEST: 20,
    }
    assert Counter(s.expected for s in samples) == {label: 25 for label in RouteLabel}
    assert any(s.difficulty == "adversarial" for s in samples)
    assert any(s.risk == "high" and s.expected == RouteLabel.HUMAN_REVIEW for s in samples)


def test_dataset_hash_is_deterministic() -> None:
    path = Path("datasets/routing-v1.yaml")
    assert dataset_sha256(path) == dataset_sha256(path)
    assert len(dataset_sha256(path)) == 64


def test_difficulty_labels_have_semantic_evidence() -> None:
    samples = load_dataset(Path("datasets/routing-v1.yaml"))
    adversarial = [sample.input for sample in samples if sample.difficulty == "adversarial"]
    ambiguous = [sample.input for sample in samples if sample.difficulty == "ambiguous"]
    assert all(
        any(marker in text for marker in ("忽略", "伪装", "不要转人工", "系统提示"))
        for text in adversarial
    )
    assert all(
        any(marker in text for marker in ("不确定", "可能", "没有说明", "先看看"))
        for text in ambiguous
    )
