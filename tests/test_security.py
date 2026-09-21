import pytest

from jev_lab.security import PublicationSafetyError, assert_public_safe


def test_public_sanitizer_rejects_nested_secret_and_request_id() -> None:
    payload = {"details": {"authorization": "Bearer secret", "request_id": "req_123"}}
    with pytest.raises(PublicationSafetyError):
        assert_public_safe(payload)


def test_public_safety_accepts_aggregate_evidence() -> None:
    assert_public_safe({"provider": "rules", "metrics": {"accuracy": 0.8}})
