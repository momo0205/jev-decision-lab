import os

import pytest

from jev_lab.contracts import RequestStatus, RouteLabel, RoutingSample, Split
from jev_lab.providers.deepseek import DeepSeekProvider
from jev_lab.providers.jev import JevProvider

SAMPLE = RoutingSample(
    sample_id="live-smoke",
    family_id="live-smoke",
    input="检索官方 Jev 文档",
    expected=RouteLabel.SEARCH,
    acceptable=[RouteLabel.SEARCH],
    risk="low",
    difficulty="clear",
    rationale="显式远程 smoke test",
    source="synthetic",
    split=Split.DEV,
)


@pytest.mark.live_deepseek
@pytest.mark.skipif(not os.environ.get("DEEPSEEK_API_KEY"), reason="DEEPSEEK_API_KEY absent")
def test_deepseek_live_smoke() -> None:
    assert (
        DeepSeekProvider.from_environment().decide(SAMPLE).request_status == RequestStatus.SUCCESS
    )


@pytest.mark.live_jev
@pytest.mark.skipif(not os.environ.get("TYPESAFE_API_KEY"), reason="TYPESAFE_API_KEY absent")
def test_jev_live_smoke() -> None:
    assert JevProvider.from_environment().decide(SAMPLE).request_status == RequestStatus.SUCCESS
