import pytest

from jev_lab.security import PublicationSafetyError, assert_public_safe, sanitize_public


def test_public_sanitizer_rejects_nested_secret_and_request_id() -> None:
    payload = {"details": {"authorization": "Bearer secret", "request_id": "req_123"}}
    with pytest.raises(PublicationSafetyError):
        assert_public_safe(payload)


def test_public_safety_accepts_aggregate_evidence() -> None:
    assert_public_safe({"provider": "rules", "metrics": {"accuracy": 0.8}})


def test_sanitize_public_copies_nested_safe_values() -> None:
    value = {"items": [{"provider": "rules"}], "ok": True}
    assert sanitize_public(value) == value
