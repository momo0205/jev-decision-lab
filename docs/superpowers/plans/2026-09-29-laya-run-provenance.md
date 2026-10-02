# Laya Run Provenance and Calibration Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add privacy-bounded runtime provenance to new experiment runs and reports, while keeping the calibration error-cluster note private and retiring the exposed test split from future evaluation claims.

**Architecture:** Add a validated `RuntimeProvenance` contract and a small collector that reads only platform family, architecture, Python version, allowlisted Laya package versions, and an explicitly exposed device. The CLI stores it on new manifests; reporting serializes the same bounded object. Keep the calibration audit note under the repository's ignored `reports/private/` directory.

**Tech Stack:** Python 3.11+, Pydantic 2, `importlib.metadata`, pytest, existing Typer CLI and report writer.

**Spec:** `docs/superpowers/specs/2026-09-29-laya-run-provenance-design.md`

## Global Constraints

- Do not load, inspect, run, tune against, or publish results from the existing `test` split.
- Runtime provenance may contain only Python version, OS family, machine architecture, `laya`/`torch`/`transformers` versions for Laya runs, and an explicitly exposed actual inference device.
- Never record hostnames, usernames, absolute paths, environment variables, credentials, prompts, or sample text.
- Missing optional package metadata or device information stays absent/null; never infer an execution device from available hardware.
- Existing manifests without runtime provenance remain valid.
- The private audit note stays under git-ignored `reports/private/`; do not add it to source control or the public website.

## Review Focus

- An unexpected provider or missing optional distribution must not leak arbitrary package metadata or fail run creation; test allowlist scoping and missing metadata.
- An absent, unrecognized, or unsafe device value must not be serialized; test unavailable and rejected device values.
- Old manifests without the new field must still parse and produce reports; test both contract and reporting compatibility.
- CLI provenance wiring must not require reading the repository dataset in its new unit test; use a synthetic in-memory dev sample and monkeypatch dataset access.
- Public report output must include only the bounded runtime fields and pass the existing public-safety validator; assert exact keys and absence of private fields.

---

### Task 1: Add bounded runtime provenance to manifests and new runs

**Files:**
- Create: `src/jev_lab/runtime.py`
- Modify: `src/jev_lab/contracts.py`
- Modify: `src/jev_lab/cli.py`
- Create: `tests/test_runtime.py`
- Modify: `tests/test_contracts.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Produces `RuntimeProvenance` in `jev_lab.contracts` with fields `python_version: str`, `system: Literal["Darwin", "Linux", "Windows", "Other"]`, `machine: str`, `package_versions: dict[Literal["laya", "torch", "transformers"], str]`, and `inference_device: str | None`.
- Produces `collect_runtime_provenance(provider: str, *, inference_agent: object | None = None) -> RuntimeProvenance` in `jev_lab.runtime`.
- `RunManifest.runtime` is optional and defaults to `None`.
- The CLI passes the loaded Laya agent to the collector only for Laya runs; other providers collect platform provenance but no Laya package or device data.

- [x] **Step 1: Write failing runtime-collector tests**
  - Assert system is normalized to an OS family and machine is architecture-only.
  - For provider `laya`, assert only `laya`, `torch`, and `transformers` metadata is queried; missing metadata is omitted.
  - Assert a device is recorded only for a recognized value exposed by the supplied agent; absent, unsafe, or unavailable values become `None`.
  - Assert a non-Laya provider does not query optional Laya package metadata or record a device.

- [x] **Step 2: Run the targeted tests and observe the expected missing-module/API failures**

  Run: `PYTHONPATH=src ../../.venv/bin/pytest -q tests/test_runtime.py`

  Expected: FAIL because `jev_lab.runtime` and the collector do not exist yet.

- [x] **Step 3: Implement the minimal collector and validated contract**
  - Use `platform`, `sys`, and `importlib.metadata.version`; catch `PackageNotFoundError` and omit absent packages.
  - Normalize the OS family and reject unrecognized device identifiers instead of stringifying arbitrary objects.
  - Add `runtime: RuntimeProvenance | None = None` to `RunManifest`.

- [x] **Step 4: Add backward-compatibility and CLI-wiring tests**
  - Validate a legacy `RunManifest` JSON without `runtime`.
  - Invoke the CLI with a synthetic dev sample while monkeypatching dataset load/validation/hash; assert the emitted manifest contains the collector result. Do not read the repository dataset in this test.

- [x] **Step 5: Run targeted tests and verify GREEN**

  Run: `PYTHONPATH=src ../../.venv/bin/pytest -q tests/test_runtime.py tests/test_contracts.py tests/test_cli.py::test_run_records_runtime_provenance_without_loading_repository_dataset`

  Expected: PASS; only synthetic fixtures are used and no provider is run on repository samples.

---

### Task 2: Expose bounded provenance in reports and preserve a private audit note

**Files:**
- Modify: `src/jev_lab/reporting.py`
- Modify: `tests/test_reporting.py`
- Create locally, git-ignored: `reports/private/laya-calibration-audit.md`

**Interfaces:**
- Consumes `RunManifest.runtime` from Task 1.
- Markdown reports show a `Runtime provenance` section with available values and explicit “not available” values for legacy manifests.
- Public aggregate JSON uses a `runtime_provenance` object or `null` and contains no per-sample input/output data.
- The private audit note records only the calibration-family pattern, limitations, and test-split handling; it is not committed or published.

- [x] **Step 1: Write failing report tests**
  - Assert runtime fields appear in Markdown and in the public aggregate JSON.
  - Assert the JSON runtime object contains exactly the approved keys and passes `assert_public_safe`.
  - Assert a legacy manifest without runtime still produces both reports and represents runtime as unavailable/null.

- [x] **Step 2: Run the targeted report tests and observe the expected failures**

  Run: `PYTHONPATH=src ../../.venv/bin/pytest -q tests/test_reporting.py`

  Expected: FAIL because reporting does not yet serialize runtime provenance.

- [x] **Step 3: Implement report serialization**
  - Render only validated runtime fields in Markdown and the public aggregate JSON.
  - Do not add sample-level data or expose `dataset_path` through this change.

- [x] **Step 4: Write the private calibration audit note**
  - Preserve the already-reviewed family-level summary: the three calibration false approvals cluster in one sample family; this is a diagnostic on a small calibration split, not generalization or causal evidence.
  - State that some test-record input/expected-label context was accidentally surfaced during review, no provider ran on that split, and it is no longer an untouched blind holdout.
  - Do not include any test record ID, input, expected label, prediction, or metric.

- [x] **Step 5: Verify reports, privacy, and ignored-note placement**

  Run: `PYTHONPATH=src ../../.venv/bin/pytest -q tests/test_reporting.py`

  Expected: PASS.

  Run: `git check-ignore -q reports/private/laya-calibration-audit.md && git diff --check`

  Expected: exit 0; the private note remains outside the tracked diff.

---

### Final verification boundary

- Run only the targeted synthetic tests named above plus any additional tests that do not read or execute the retired test split.
- Do not run the broad offline acceptance suite if it loads repository test records or executes a test-split provider.
- Review the final diff and the private note separately; keep the private note out of the staged/committed changes.
- No merge, push, or website publication is part of this implementation plan.
