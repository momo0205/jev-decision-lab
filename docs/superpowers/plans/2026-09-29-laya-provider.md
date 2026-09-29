# Optional Laya Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible optional local Laya baseline to Jev Decision Lab without changing its default offline dependency set.

**Architecture:** Implement the existing `DecisionProvider` protocol and use the pinned Laya checkpoint behind a lazy optional dependency. Keep the runner and shared result contract intact; validate and normalize the typed choice response, then record it through the existing experiment flow.

**Tech Stack:** Python 3.11+, Typer, Pydantic, pytest, optional Laya 0.3.21 / PyTorch / Transformers.

**Spec:** `docs/superpowers/specs/2026-09-29-laya-provider-design.md`

## Global Constraints

- Keep the default install and offline CI free of Laya, PyTorch, and Transformers.
- Preserve the existing `DecisionProvider` and `DecisionResult` contract.
- Never silently fall back from Laya to another provider.
- Do not use the held-out `test` split during implementation or local tuning.
- Do not use secrets or write to the website repository.

## Review Focus

- If the optional package is absent, normal CLI import must still work and explicit Laya selection must fail with install guidance before creating a run.
- Missing or malformed Laya `answers.route` must produce a sanitized failed result, not a fabricated label.
- Rounded probabilities may not sum exactly to one; normalize only after validating every required label and value.
- Exceptions must not leak input text, local paths, or credentials into `DecisionResult.error`.
- Laya's action probability and entropy confidence are not interchangeable with option probabilities; neither may be used as an implicit abstention rule.

---

### Task 1: Add the optional Laya provider and reproducible CLI path

**Files:**
- Modify: `pyproject.toml`
- Create: `src/jev_lab/providers/laya.py`
- Modify: `src/jev_lab/cli.py`
- Modify: `tests/test_offline_acceptance.py`
- Create: `tests/providers/test_laya.py`
- Modify: `tests/test_cli.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `DecisionProvider`, `RoutingSample`, `DecisionResult`, `RequestStatus`, `RouteLabel`, `RunMode`.
- Produces: `LayaProvider(agent: LayaPredictor)`, `LayaProvider.from_default_checkpoint() -> LayaProvider`, and CLI provider value `laya`.
- `LayaPredictor.predict(state, questions, lang=...)` returns the Laya SDK mapping; the adapter reads `answers.route.choice` and `answers.route.probabilities`.

- [x] **Step 1: Write failing provider tests** for choice/probability mapping, rounded probability normalization, and invalid/missing/partial outputs returning sanitized `FAILED` results without fallback.
- [x] **Step 2: Run `PYTHONPATH=src ../../.venv/bin/pytest tests/providers/test_laya.py -q`** and confirm the expected import/attribute failure.
- [x] **Step 3: Implement the lazy optional adapter** with fixed model ID/revision, offline run mode, unknown monetary cost (`None`), stable provider/model identifiers, and safe error codes.
- [x] **Step 4: Run focused provider tests** and confirm all pass.
- [x] **Step 5: Add failing CLI/offline tests** proving ordinary CLI imports do not import `laya`/`torch`, and selecting Laya without its extra gives install guidance and creates no run directory.
- [x] **Step 6: Run the new CLI tests and confirm RED**, then add the optional dependency group and `--provider laya` dispatch.
- [x] **Step 7: Update README** with the optional install/run commands, pinned checkpoint provenance, first-run download/cache behavior, and explicit cold-start/cost interpretation; keep test split use gated by the existing freeze protocol.
- [x] **Step 8: Run focused tests, `PYTHONPATH=src ../../.venv/bin/pytest -m 'not live_deepseek and not live_jev' -q`, Ruff, and Mypy.** Expected: all pass; default CI imports no Laya/PyTorch, no network or key required.
- [ ] **Step 9: Commit** the task as `feat: add optional local Laya provider`.
