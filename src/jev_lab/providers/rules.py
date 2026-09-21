from time import perf_counter

from jev_lab.contracts import DecisionResult, RequestStatus, RouteLabel, RoutingSample, RunMode


class RulesProvider:
    name = "rules"
    model_version = "rules-v1"
    run_mode = RunMode.OFFLINE_DEVELOPMENT

    def decide(self, sample: RoutingSample) -> DecisionResult:
        started = perf_counter()
        text = sample.input.lower()
        if any(word in text for word in ("删除", "绕过审批", "支付", "交易", "凭据", "权限")):
            label = RouteLabel.HUMAN_REVIEW
        elif any(word in text for word in ("数据库", "查询", "字段", "记录", "表结构")):
            label = RouteLabel.DATABASE
        elif any(word in text for word in ("代码", "程序", "调试", "修复", "构建")):
            label = RouteLabel.CODE
        elif any(word in text for word in ("检索", "搜索", "公开资料", "最新规范", "查天气")):
            label = RouteLabel.SEARCH
        else:
            label = RouteLabel.HUMAN_REVIEW
        return DecisionResult(sample_id=sample.sample_id, label=label, probabilities=None,
            abstained=False, latency_ms=(perf_counter() - started) * 1000,
            estimated_cost_usd=0.0, provider=self.name, model_version=self.model_version,
            request_status=RequestStatus.SUCCESS, error=None, run_mode=self.run_mode)
