# Jev Decision Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline-first Python lab that compares deterministic rules, DeepSeek structured decisions, and Jev decisions on a versioned Chinese Agent-routing dataset without presenting mocks as live evidence.

**Architecture:** A typed dataset feeds interchangeable providers through one runner and one `DecisionResult` contract. Metrics and reports consume immutable run artifacts; raw artifacts remain local while an explicit sanitizer produces reviewable public summaries. Remote providers are optional and never block the offline rules baseline.

**Tech Stack:** Python 3.11+, Pydantic 2.x, Typer, PyYAML, httpx, pytest, pytest-cov, Ruff, MyPy

**Spec:** `docs/superpowers/specs/2026-09-21-jev-decision-lab-design.md`

## Global Constraints

- The repository is public; never commit API keys, `.env`, private prompts, raw live responses, request IDs, or reversible sensitive traces.
- A clean machine with no network and no API keys must validate the dataset, run rules, calculate metrics, generate a report, and pass all offline tests.
- The first dataset has exactly 100 Chinese samples: 60 `dev`, 20 `calibration`, and 20 `test`; a `family_id` may occur in only one split.
- Labels are exactly `search`, `code`, `database`, and `human_review`.
- `recorded-replay` and mocks must never be labeled as `live-evaluation`.
- Missing probabilities and unverifiable costs stay `null`; do not synthesize confidence or cost.
- Remote providers use explicit timeouts and bounded retry, and never silently fall back during an evaluation run.
- The held-out test split is run only after the rules, prompts, questions, and thresholds have been frozen.
- Website publication remains a manual, reviewed operation outside this repository.

## Review Focus

- A duplicate or related sample family crossing splits must make dataset validation fail before any run begins; Task 2 pins this.
- A remote provider returning an allowed label with malformed or non-normalized probabilities must be rejected rather than silently scored; Tasks 6 and 7 pin this.
- Zero accepted decisions after thresholding must report `coverage=0` and an undefined selective accuracy without dividing by zero; Task 4 pins this.
- A provider exception or timeout must create a failed result for that provider and sample without changing provider identity or invoking another provider; Task 3 pins this.
- Public export must reject nested secrets and sensitive request metadata, not only top-level fields; Task 5 pins this.

---

## File Map

- `pyproject.toml`: package metadata, dependencies, CLI entry point, test/lint/type configuration.
- `src/jev_lab/contracts.py`: labels, samples, provider results, run manifests and metric models.
- `src/jev_lab/dataset.py`: YAML loading, dataset validation and stable hashing.
- `src/jev_lab/providers/base.py`: provider protocol and provider configuration errors.
- `src/jev_lab/providers/rules.py`: deterministic routing baseline.
- `src/jev_lab/providers/deepseek.py`: OpenAI-compatible DeepSeek structured-output adapter.
- `src/jev_lab/providers/jev.py`: TypeSafe SDK adapter behind a small injectable client protocol.
- `src/jev_lab/providers/recorded.py`: replay adapter with provenance enforcement.
- `src/jev_lab/runner.py`: one-provider experiment loop and JSONL artifact writer.
- `src/jev_lab/metrics.py`: classification, selective-decision, calibration and latency metrics.
- `src/jev_lab/reporting.py`: Markdown/JSON report generation.
- `src/jev_lab/security.py`: recursive public-export sanitizer and sensitive-value scanner.
- `src/jev_lab/cli.py`: dataset, run, evaluate and report commands.
- `datasets/routing-v1.yaml`: the 100 versioned routing samples.
- `datasets/dataset-card.md`: construction, labeling, split and limitation documentation.
- `docs/getting-jev-access.md`: access, local configuration and live smoke-test guide.
- `tests/`: unit, contract and offline end-to-end tests.

### Task 1: Package foundation and typed contracts

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `README.md`
- Create: `src/jev_lab/__init__.py`
- Create: `src/jev_lab/contracts.py`
- Create: `src/jev_lab/cli.py`
- Test: `tests/test_contracts.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: none.
- Produces: `RouteLabel`, `Split`, `RunMode`, `RequestStatus`, `RoutingSample`, `DecisionResult`, `RunManifest`; CLI command `jev-lab --help`.

- [ ] **Step 1: Write failing contract and CLI tests**

```python
# tests/test_contracts.py
import pytest
from pydantic import ValidationError
from jev_lab.contracts import DecisionResult, RequestStatus, RouteLabel, RoutingSample, RunMode, Split

def test_routing_sample_rejects_expected_label_outside_acceptable() -> None:
    with pytest.raises(ValidationError):
        RoutingSample(sample_id="s1", family_id="f1", input="查资料", expected=RouteLabel.SEARCH,
                      acceptable=[RouteLabel.CODE], risk="low", difficulty="clear",
                      rationale="需要搜索", source="synthetic", split=Split.DEV)

def test_decision_result_keeps_missing_probability_and_cost_explicit() -> None:
    result = DecisionResult(sample_id="s1", label=RouteLabel.SEARCH, probabilities=None,
                            abstained=False, latency_ms=1.0, estimated_cost_usd=None,
                            provider="rules", model_version="rules-v1",
                            request_status=RequestStatus.SUCCESS, error=None,
                            run_mode=RunMode.OFFLINE_DEVELOPMENT)
    assert result.probabilities is None
    assert result.estimated_cost_usd is None
```

```python
# tests/test_cli.py
from typer.testing import CliRunner
from jev_lab.cli import app

def test_root_help_is_available_offline() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "dataset" in result.stdout
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_contracts.py tests/test_cli.py -q`  
Expected: FAIL because `jev_lab` does not exist.

- [ ] **Step 3: Add package configuration and exact contracts**

Configure Python `>=3.11`, runtime dependencies `pydantic>=2,<3`, `typer>=0.12,<1`, `PyYAML>=6,<7`, `httpx>=0.27,<1`, and dev dependencies `pytest`, `pytest-cov`, `ruff`, `mypy`. Define string enums with the four labels, three splits, three run modes, and statuses `success`, `failed`, `skipped`. Add Pydantic validators that require `expected in acceptable`, non-negative latency/cost, allowed probability keys only, values in `[0,1]`, and probability sum within `1e-6` of one.

```python
class RoutingSample(BaseModel):
    sample_id: str
    family_id: str
    input: str
    expected: RouteLabel
    acceptable: list[RouteLabel]
    risk: Literal["low", "medium", "high"]
    difficulty: Literal["clear", "ambiguous", "adversarial"]
    rationale: str
    source: Literal["synthetic", "redacted"]
    split: Split

class DecisionResult(BaseModel):
    sample_id: str
    label: RouteLabel | None
    probabilities: dict[RouteLabel, float] | None
    abstained: bool
    latency_ms: float
    estimated_cost_usd: float | None
    provider: str
    model_version: str
    request_status: RequestStatus
    error: str | None
    run_mode: RunMode
```

Add an empty Typer root with a `dataset` sub-app. Ignore `.venv/`, `.env`, `runs/`, private reports, caches and coverage output; do not ignore `public/`. `.env.example` contains only empty `DEEPSEEK_API_KEY=` and `TYPESAFE_API_KEY=` keys.

- [ ] **Step 4: Run tests, lint and types**

Run: `python -m pytest tests/test_contracts.py tests/test_cli.py -q && ruff check . && mypy src`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore .env.example README.md src tests
git commit -m "feat: establish Jev lab contracts"
```

### Task 2: Versioned routing dataset and leakage validation

**Files:**
- Create: `src/jev_lab/dataset.py`
- Create: `datasets/routing-v1.yaml`
- Create: `datasets/dataset-card.md`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `RoutingSample`, `Split` from Task 1.
- Produces: `load_dataset(path: Path) -> list[RoutingSample]`, `validate_dataset(samples, expected_size=100) -> None`, `dataset_sha256(path: Path) -> str`.

- [ ] **Step 1: Write failing dataset validation tests**

```python
def test_rejects_family_leakage_across_splits() -> None:
    samples = [sample("a", "shared", Split.DEV), sample("b", "shared", Split.TEST)]
    with pytest.raises(ValueError, match="family.*shared"):
        validate_dataset(samples, expected_size=2)

def test_routing_v1_has_frozen_shape() -> None:
    samples = load_dataset(Path("datasets/routing-v1.yaml"))
    assert Counter(s.split for s in samples) == {Split.DEV: 60, Split.CALIBRATION: 20, Split.TEST: 20}
    assert set(s.expected for s in samples) == set(RouteLabel)
    assert any(s.difficulty == "adversarial" for s in samples)
    assert any(s.risk == "high" and s.expected == RouteLabel.HUMAN_REVIEW for s in samples)
```

Also test duplicate IDs, blank input/rationale, exact size, and deterministic SHA-256.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_dataset.py -q`  
Expected: FAIL because loader and dataset do not exist.

- [ ] **Step 3: Implement validation and author all 100 samples**

Implement safe YAML loading, Pydantic parsing, exact split counts, unique IDs, family isolation and stable file hashing. Author 25 families of four related variants, placing 15 families in `dev`, five in `calibration`, and five in `test`. Across the 100 records, balance the four expected labels at 25 each. Each split must contain clear, ambiguous and adversarial cases; all high-risk destructive, permission-changing, payment, trading and credential requests route to `human_review` unless the request is only asking for documentation search.

Use stable IDs `route-<label>-001` through `route-<label>-025`. Every record contains a concrete Chinese input, a non-empty rationale and an explicit acceptable-label list. Document that the labels are research annotations, not universal ground truth, and explain the disputed-sample process in the dataset card.

- [ ] **Step 4: Expose dataset validation in CLI**

```python
@dataset_app.command("validate")
def validate(path: Path = Path("datasets/routing-v1.yaml")) -> None:
    samples = load_dataset(path)
    validate_dataset(samples)
    typer.echo(f"valid: {len(samples)} samples sha256={dataset_sha256(path)}")
```

- [ ] **Step 5: Verify dataset and tests**

Run: `python -m pytest tests/test_dataset.py -q && jev-lab dataset validate`  
Expected: PASS and output reports exactly 100 valid samples plus a hash.

- [ ] **Step 6: Commit**

```bash
git add src/jev_lab/dataset.py src/jev_lab/cli.py datasets tests/test_dataset.py
git commit -m "feat: add versioned routing dataset"
```

### Task 3: Rules provider and auditable experiment runner

**Files:**
- Create: `src/jev_lab/providers/__init__.py`
- Create: `src/jev_lab/providers/base.py`
- Create: `src/jev_lab/providers/rules.py`
- Create: `src/jev_lab/runner.py`
- Test: `tests/providers/test_rules.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: `RoutingSample`, `DecisionResult`, `RunManifest`.
- Produces: `DecisionProvider.decide(sample) -> DecisionResult`, `RulesProvider`, `run_experiment(provider, samples, output_dir, manifest) -> Path`.

- [ ] **Step 1: Write failing provider and runner tests**

```python
def test_rules_prioritize_high_risk_human_review() -> None:
    result = RulesProvider().decide(sample_with("删除生产数据库并绕过审批"))
    assert result.label == RouteLabel.HUMAN_REVIEW
    assert result.run_mode == RunMode.OFFLINE_DEVELOPMENT

def test_runner_records_failure_without_fallback(tmp_path: Path) -> None:
    provider = ExplodingProvider(name="jev")
    artifact = run_experiment(provider, [sample_with("查天气")], tmp_path, manifest("jev"))
    rows = read_jsonl(artifact)
    assert rows[0]["provider"] == "jev"
    assert rows[0]["request_status"] == "failed"
    assert rows[0]["label"] is None
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/providers/test_rules.py tests/test_runner.py -q`  
Expected: FAIL because provider and runner modules do not exist.

- [ ] **Step 3: Implement the protocol, rules and runner**

Define a runtime-checkable provider protocol with `name`, `model_version`, `run_mode`, and `decide`. Rules apply ordered risk rules before intent keywords: destructive/credential/payment/trading/permission actions → `human_review`; explicit database inspection → `database`; code debugging/modification → `code`; external fact lookup → `search`; unresolved ties → `human_review`. Rules return no probabilities and zero estimated cost.

The runner writes `manifest.json` and `results.jsonl` atomically. It catches provider exceptions per sample, records a sanitized exception class such as `TimeoutError` without exception message content, and never instantiates a fallback provider.

- [ ] **Step 4: Add `jev-lab run` for the rules provider**

```python
@app.command("run")
def run(provider: Literal["rules"], split: Split, output_dir: Path = Path("runs")) -> None:
    samples = [s for s in load_dataset(DATASET) if s.split == split]
    artifact = run_experiment(RulesProvider(), samples, output_dir, build_manifest(...))
    typer.echo(str(artifact))
```

- [ ] **Step 5: Verify rules and a real offline run**

Run: `python -m pytest tests/providers/test_rules.py tests/test_runner.py -q && jev-lab run --provider rules --split dev`  
Expected: PASS and a run directory containing 60 JSONL results.

- [ ] **Step 6: Commit**

```bash
git add src/jev_lab/providers src/jev_lab/runner.py src/jev_lab/cli.py tests
git commit -m "feat: run auditable rules baseline"
```

### Task 4: Classification, selective, calibration and runtime metrics

**Files:**
- Create: `src/jev_lab/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: ordered pairs of `RoutingSample` and `DecisionResult`.
- Produces: `EvaluationMetrics`, `evaluate(samples, results, threshold=None) -> EvaluationMetrics`.

- [ ] **Step 1: Write failing exact-value tests**

```python
def test_zero_coverage_is_defined_without_division_error() -> None:
    metrics = evaluate(SAMPLES, all_abstained_results())
    assert metrics.coverage == 0.0
    assert metrics.selective_accuracy is None

def test_known_confusion_matrix_and_accuracy() -> None:
    metrics = evaluate(four_balanced_samples(), three_correct_one_wrong())
    assert metrics.accuracy == 0.75
    assert metrics.confusion_matrix["search"]["code"] == 1

def test_brier_ignores_results_without_probabilities() -> None:
    metrics = evaluate(SAMPLES, results_with_one_probability_row())
    assert metrics.probability_sample_count == 1
    assert metrics.brier_score == pytest.approx(EXPECTED_BRIER)
```

Add tests for acceptable alternate labels, failed results, high-risk false approval, even/odd latency percentiles and threshold-caused abstention.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_metrics.py -q`  
Expected: FAIL because metrics do not exist.

- [ ] **Step 3: Implement deterministic metrics**

Compute overall accuracy over successful non-abstained decisions, four-by-four confusion matrix, per-class precision/recall, coverage over all eligible samples, selective accuracy, high-risk false approvals, Brier score only for rows with complete probabilities, five calibration buckets, provider success rate, P50/P95 latency and total/mean known cost. Return `None` rather than zero when a denominator or evidence is absent.

- [ ] **Step 4: Add CLI evaluation**

`jev-lab evaluate --run <run-directory>` loads the manifest, exact dataset hash and results, refuses a hash mismatch, writes `metrics.json`, and prints accuracy, coverage and success rate.

- [ ] **Step 5: Verify**

Run: `python -m pytest tests/test_metrics.py -q && jev-lab evaluate --run <rules-run-directory>`  
Expected: PASS and `metrics.json` is created without network access.

- [ ] **Step 6: Commit**

```bash
git add src/jev_lab/metrics.py src/jev_lab/cli.py tests/test_metrics.py
git commit -m "feat: evaluate decision quality and coverage"
```

### Task 5: Reports, recursive sanitization and public export

**Files:**
- Create: `src/jev_lab/security.py`
- Create: `src/jev_lab/reporting.py`
- Test: `tests/test_security.py`
- Test: `tests/test_reporting.py`

**Interfaces:**
- Consumes: manifest, metrics and result rows.
- Produces: `sanitize_public(value) -> JSONValue`, `assert_public_safe(value) -> None`, `write_report(run_dir, report_dir, public_dir) -> tuple[Path, Path]`.

- [ ] **Step 1: Write failing nested-secret and provenance tests**

```python
def test_public_sanitizer_rejects_nested_secret_and_request_id() -> None:
    payload = {"details": {"authorization": "Bearer secret", "request_id": "req_123"}}
    with pytest.raises(PublicationSafetyError):
        assert_public_safe(payload)

def test_report_names_recorded_replay_without_live_claim(tmp_path: Path) -> None:
    markdown, public_json = write_report(recorded_run(tmp_path), tmp_path / "reports", tmp_path / "public")
    assert "recorded-replay" in markdown.read_text()
    assert "live-evaluation" not in markdown.read_text()
    assert json.loads(public_json.read_text())["run_mode"] == "recorded-replay"
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_security.py tests/test_reporting.py -q`  
Expected: FAIL because reporting and security modules do not exist.

- [ ] **Step 3: Implement recursive policy and report rendering**

Walk dictionaries and lists recursively. Reject key names matching `api_key`, `authorization`, `token`, `secret`, `request_id`, `raw_response`, or `trace`; reject values matching common secret prefixes. Public output includes aggregate metrics, dataset hash, Git commit, provider/model, run mode, timestamp and redacted failure categories, but excludes sample inputs and raw outputs.

Markdown report sections are: provenance, dataset/split, decision quality, selective decisions, probability quality, runtime/cost, failures, limitations. Missing metrics render as `not available`, never `0`.

- [ ] **Step 4: Add report CLI**

`jev-lab report --run <run-directory>` writes a full Markdown report under `reports/<run-id>.md` and safe JSON under `public/<run-id>.json`, then runs `assert_public_safe` on the final JSON.

- [ ] **Step 5: Verify report generation and secret scan**

Run: `python -m pytest tests/test_security.py tests/test_reporting.py -q && jev-lab report --run <rules-run-directory>`  
Expected: PASS; both outputs exist and the public JSON contains no inputs.

- [ ] **Step 6: Commit**

```bash
git add src/jev_lab/security.py src/jev_lab/reporting.py src/jev_lab/cli.py tests
git commit -m "feat: generate safe evidence reports"
```

### Task 6: Optional DeepSeek structured-output provider

**Files:**
- Create: `src/jev_lab/providers/deepseek.py`
- Test: `tests/providers/test_deepseek.py`
- Modify: `src/jev_lab/cli.py`
- Modify: `.env.example`

**Interfaces:**
- Consumes: `RoutingSample`, `DecisionResult`, provider protocol.
- Produces: `DeepSeekProvider(api_key, client, model="deepseek-chat", timeout_seconds=20, max_retries=1)`.

- [ ] **Step 1: Write failing configuration and response-contract tests**

```python
def test_missing_key_is_skipped_without_network() -> None:
    provider = DeepSeekProvider.from_environment(client=FailIfCalledClient())
    result = provider.decide(sample_with("查天气"))
    assert result.request_status == RequestStatus.SKIPPED
    assert result.error == "missing_credentials"

def test_rejects_malformed_probabilities() -> None:
    provider = DeepSeekProvider("key", client=FakeClient(label="search", probabilities={"search": 1.2}))
    result = provider.decide(sample_with("查天气"))
    assert result.request_status == RequestStatus.FAILED
    assert result.error == "invalid_response"
```

Also test allowed labels, timeout, one bounded retry, cost remaining null without a pinned price, and no raw response in errors.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/providers/test_deepseek.py -q`  
Expected: FAIL because the provider does not exist.

- [ ] **Step 3: Implement the injectable OpenAI-compatible adapter**

Send the four label definitions and require JSON with `label` plus optional complete `probabilities`. Parse locally through Pydantic. Set `run_mode=live-evaluation` only for an actual client call; missing credentials produce a skipped result without creating an HTTP client. Use a 20-second timeout and at most one retry for transport errors. Do not retry schema-invalid responses.

- [ ] **Step 4: Wire CLI provider selection**

Extend `--provider` to `rules|deepseek|jev|recorded`. Construct only the selected provider. A missing DeepSeek key completes the run with skipped rows and a clear console warning; it does not fail offline commands.

- [ ] **Step 5: Verify with fakes and optional smoke marker**

Run: `python -m pytest tests/providers/test_deepseek.py -q`  
Expected: PASS without network. If `DEEPSEEK_API_KEY` is present, run `python -m pytest -m live_deepseek -q`; otherwise record `SKIPPED (credential absent)`.

- [ ] **Step 6: Commit**

```bash
git add src/jev_lab/providers/deepseek.py src/jev_lab/cli.py .env.example tests/providers/test_deepseek.py
git commit -m "feat: add optional DeepSeek decisions"
```

### Task 7: Jev adapter, recorded replay and access guide

**Files:**
- Create: `src/jev_lab/providers/jev.py`
- Create: `src/jev_lab/providers/recorded.py`
- Create: `docs/getting-jev-access.md`
- Create: `tests/providers/test_jev.py`
- Create: `tests/providers/test_recorded.py`
- Create: `tests/fixtures/recorded/jev-routing.jsonl`
- Modify: `src/jev_lab/cli.py`

**Interfaces:**
- Consumes: provider protocol and contracts.
- Produces: `JevProvider.from_environment(client_factory=...)`, `RecordedProvider(path, declared_provider, model_version)`.

- [ ] **Step 1: Write failing Jev boundary and replay-provenance tests**

```python
def test_jev_missing_access_is_explicitly_skipped() -> None:
    result = JevProvider.from_environment(client_factory=fail_factory).decide(sample_with("查天气"))
    assert result.request_status == RequestStatus.SKIPPED
    assert result.error == "missing_credentials"

def test_jev_invalid_probability_map_fails_closed() -> None:
    provider = JevProvider("key", client=FakeJevClient(choice="search", probabilities={"other": 1.0}))
    assert provider.decide(sample_with("查天气")).error == "invalid_response"

def test_recorded_provider_cannot_claim_live_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="recorded-replay"):
        RecordedProvider(write_fixture(tmp_path, run_mode="live-evaluation"), "jev", "fixture")
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/providers/test_jev.py tests/providers/test_recorded.py -q`  
Expected: FAIL because both adapters do not exist.

- [ ] **Step 3: Implement Jev behind a minimal client boundary**

The internal client protocol accepts state plus one Choice question and returns choice plus optional probabilities. The SDK import is lazy so the package works without `typesafe-sdk`. Missing SDK yields `skipped/dependency_missing`; missing key yields `skipped/missing_credentials`. Map only the four exact labels, validate probabilities through `DecisionResult`, use bounded retry for transport failures, and strip raw SDK exception messages.

- [ ] **Step 4: Implement recorded replay with immutable provenance**

Each fixture row includes `sample_id`, `provider`, `model_version`, `recorded_at`, `source_run_hash`, `label`, optional probabilities, and `run_mode: recorded-replay`. Reject duplicate IDs, missing provenance, unknown samples or any other run mode. The committed fixture is synthetic and clearly marked `model_version: synthetic-contract-fixture`; it verifies plumbing only.

- [ ] **Step 5: Write the access and smoke-test guide**

Document TypeSafe early-access registration, where to place `TYPESAFE_API_KEY`, how to install the optional SDK extra, how to run one explicitly marked live smoke test, expected skipped states, how to inspect output without printing secrets, and the rule that only successful live runs may change the website from “实验尚未开始”. Include the Vercel AI Gateway route as an alternative requiring a future adapter, not as an already supported path.

- [ ] **Step 6: Verify offline adapters**

Run: `python -m pytest tests/providers/test_jev.py tests/providers/test_recorded.py -q && jev-lab run --provider recorded --split dev`  
Expected: PASS without SDK, network or key; output is labeled `recorded-replay`.

- [ ] **Step 7: Commit**

```bash
git add src/jev_lab/providers src/jev_lab/cli.py docs/getting-jev-access.md tests
git commit -m "feat: add Jev boundary and recorded replay"
```

### Task 8: Offline acceptance, CI and operator documentation

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `tests/test_offline_acceptance.py`
- Modify: `README.md`
- Modify: `datasets/dataset-card.md`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: a documented, CI-enforced offline workflow and an explicit live-evaluation handoff.

- [ ] **Step 1: Write the failing subprocess acceptance test**

```python
def test_complete_rules_workflow_without_keys_or_network(tmp_path: Path, monkeypatch) -> None:
    env = minimal_env_without_model_keys()
    validate = run_cli(["dataset", "validate"], env=env)
    run = run_cli(["run", "--provider", "rules", "--split", "dev", "--output-dir", str(tmp_path)], env=env)
    run_dir = Path(run.stdout.strip().splitlines()[-1])
    assert run_cli(["evaluate", "--run", str(run_dir)], env=env).returncode == 0
    assert run_cli(["report", "--run", str(run_dir)], env=env).returncode == 0
    assert "valid: 100 samples" in validate.stdout
    assert (Path("public") / f"{run_dir.name}.json").exists()
```

The test blocks outbound sockets and removes both API-key environment variables.

- [ ] **Step 2: Run acceptance test and verify RED**

Run: `python -m pytest tests/test_offline_acceptance.py -q`  
Expected: FAIL until CLI paths and output locations work as one workflow.

- [ ] **Step 3: Make the smallest integration corrections**

Align CLI exit codes, printed run-directory path, dataset hash verification and report output options. Do not add new product features. README must contain installation, five offline commands, run-mode definitions, current evidence status, remote-provider setup links, public/private artifact boundaries and the investment/production disclaimer that this lab makes no automated decisions.

- [ ] **Step 4: Add CI with no secrets**

The workflow runs on pushes and pull requests with Python 3.11 and executes:

```bash
python -m pip install -e '.[dev]'
ruff check .
mypy src
pytest -m 'not live_deepseek and not live_jev' --cov=jev_lab --cov-fail-under=90
jev-lab dataset validate
```

Do not add repository secrets or live jobs.

- [ ] **Step 5: Run the complete verification suite**

Run: `ruff check . && mypy src && pytest -m 'not live_deepseek and not live_jev' --cov=jev_lab --cov-fail-under=90 && jev-lab dataset validate`  
Expected: PASS; 100 samples validate; no live test runs.

- [ ] **Step 6: Inspect public artifacts and repository history for secrets**

Run: `rg -n --hidden -g '!.git/**' -e 'sk-[A-Za-z0-9_-]{12,}' -e 'Bearer [A-Za-z0-9._-]{12,}' -e 'TYPESAFE_API_KEY=.+' -e 'DEEPSEEK_API_KEY=.+' .`  
Expected: no matches containing values; `.env.example` contains empty assignments only.

- [ ] **Step 7: Commit**

```bash
git add .github README.md datasets/dataset-card.md tests/test_offline_acceptance.py
git commit -m "test: verify offline Jev lab workflow"
```

## Final Review and Release Gate

- [ ] Use `superpowers:requesting-code-review` for a whole-branch review against the design and this plan.
- [ ] Fix accepted findings using `superpowers:receiving-code-review` and TDD.
- [ ] Use `superpowers:verification-before-completion` and rerun the exact Task 8 verification command.
- [ ] Confirm no `live-evaluation` result is claimed unless a real credentialed call was executed and preserved with valid provenance.
- [ ] Use `superpowers:finishing-a-development-branch` to choose merge/push handling.
- [ ] Create the public GitHub repository only after the secret scan and user confirmation of the target GitHub account.
- [ ] Do not update `notes.ironmao.com` until the public report has passed manual evidence and privacy review.
