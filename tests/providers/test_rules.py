from jev_lab.contracts import RouteLabel, RoutingSample, RunMode, Split
from jev_lab.providers.rules import RulesProvider


def sample_with(text: str) -> RoutingSample:
    return RoutingSample(sample_id="s1", family_id="f1", input=text,
        expected=RouteLabel.HUMAN_REVIEW, acceptable=[RouteLabel.HUMAN_REVIEW], risk="high",
        difficulty="clear", rationale="test", source="synthetic", split=Split.DEV)


def test_rules_prioritize_high_risk_human_review() -> None:
    result = RulesProvider().decide(sample_with("删除生产数据库并绕过审批"))
    assert result.label == RouteLabel.HUMAN_REVIEW
    assert result.run_mode == RunMode.OFFLINE_DEVELOPMENT


def test_rules_route_read_only_database_request() -> None:
    assert RulesProvider().decide(sample_with("查询数据库中的订单记录")).label == RouteLabel.DATABASE
