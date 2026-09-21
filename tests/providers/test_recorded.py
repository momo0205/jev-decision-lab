import json
from pathlib import Path

import pytest

from jev_lab.contracts import RouteLabel, RoutingSample, Split
from jev_lab.providers.recorded import RecordedProvider


def test_recorded_provider_cannot_claim_live_mode(tmp_path: Path) -> None:
    path = tmp_path / "fixture.jsonl"
    path.write_text(
        json.dumps(
            {
                "sample_id": "s",
                "provider": "jev",
                "model_version": "fixture",
                "recorded_at": "2026-09-21T00:00:00Z",
                "source_run_hash": "abc",
                "label": "search",
                "probabilities": None,
                "run_mode": "live-evaluation",
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="recorded-replay"):
        RecordedProvider(path, "jev", "fixture")


def test_recorded_provider_rejects_duplicate_ids(tmp_path: Path) -> None:
    row = {
        "sample_id": "s",
        "provider": "jev",
        "model_version": "fixture",
        "recorded_at": "2026-09-21T00:00:00Z",
        "source_run_hash": "abc",
        "label": "search",
        "probabilities": None,
        "run_mode": "recorded-replay",
    }
    path = tmp_path / "fixture.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        RecordedProvider(path, "jev", "fixture")


def test_recorded_provider_returns_fixture_decision(tmp_path: Path) -> None:
    row = {
        "sample_id": "s",
        "provider": "jev",
        "model_version": "fixture",
        "recorded_at": "2026-09-21T00:00:00Z",
        "source_run_hash": "abc",
        "label": "search",
        "probabilities": None,
        "run_mode": "recorded-replay",
    }
    path = tmp_path / "fixture.jsonl"
    path.write_text(json.dumps(row) + "\n")
    provider = RecordedProvider(path, "jev", "fixture")
    sample = RoutingSample(
        sample_id="s",
        family_id="f",
        input="x",
        expected=RouteLabel.SEARCH,
        acceptable=[RouteLabel.SEARCH],
        risk="low",
        difficulty="clear",
        rationale="x",
        source="synthetic",
        split=Split.DEV,
    )
    assert provider.decide(sample).label == RouteLabel.SEARCH
