from typing import Protocol, runtime_checkable

from jev_lab.contracts import DecisionResult, RoutingSample, RunMode


@runtime_checkable
class DecisionProvider(Protocol):
    name: str
    model_version: str
    run_mode: RunMode

    def decide(self, sample: RoutingSample) -> DecisionResult: ...
