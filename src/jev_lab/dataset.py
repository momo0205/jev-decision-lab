from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from pydantic import TypeAdapter

from jev_lab.contracts import RoutingSample, Split

_SAMPLES = TypeAdapter(list[RoutingSample])


def load_dataset(path: Path) -> list[RoutingSample]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypeError("dataset root must be a list")
    return _SAMPLES.validate_python(raw)


def validate_dataset(samples: list[RoutingSample], expected_size: int = 100) -> None:
    if len(samples) != expected_size:
        raise ValueError(f"expected {expected_size} samples, got {len(samples)}")
    ids = [sample.sample_id for sample in samples]
    duplicates = sorted(sample_id for sample_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate sample_id: {', '.join(duplicates)}")

    family_splits: dict[str, set[Split]] = defaultdict(set)
    for sample in samples:
        family_splits[sample.family_id].add(sample.split)
    leaked = sorted(family for family, splits in family_splits.items() if len(splits) > 1)
    if leaked:
        raise ValueError(f"family crosses splits: {', '.join(leaked)}")

    if expected_size == 100:
        expected = {Split.DEV: 60, Split.CALIBRATION: 20, Split.TEST: 20}
        actual = Counter(sample.split for sample in samples)
        if actual != expected:
            raise ValueError(f"invalid split counts: {dict(actual)}")


def dataset_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
