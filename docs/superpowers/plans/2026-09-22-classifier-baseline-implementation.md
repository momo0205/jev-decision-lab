# Classifier Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible offline TF-IDF + Logistic Regression baseline that can be trained on `dev`, calibrated and evaluated through the existing Jev Decision Lab pipeline, while recording its training and change costs honestly.

**Architecture:** Training is isolated under `jev_lab.training`, produces a hash-verified immutable artifact, and never runs inside prediction. A `ClassifierProvider` adapts the frozen model to the existing `DecisionProvider` contract. Existing metrics and reports consume the same `DecisionResult`, with optional fixed provenance fields added to `RunManifest` for local-model engineering costs.

**Tech Stack:** Python 3.11+, Pydantic 2, Typer, scikit-learn 1.5+, joblib, pytest, Ruff, MyPy, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-22-classifier-baseline-design.md`

## Global Constraints

- The default `.[dev]` installation must remain usable without scikit-learn, network access, API keys, or model downloads.
- Classifier dependencies live in the optional `classifier` extra and support Python 3.11+.
- Only `Split.DEV` samples may train a classifier; `calibration` and `test` are rejected before artifact directories are created.
- Related samples are grouped by `family_id` in cross-validation.
- Training, threshold selection, and final test remain distinct phases; the implementation must not inspect test labels while training or calibrating.
- Existing rule, recorded, DeepSeek, and Jev behavior must remain backward-compatible.
- A classifier load failure must never silently fall back to rules.
- The first implementation uses character TF-IDF plus Logistic Regression; no Transformer dependency or pretrained-model download is added.
- Reports must not claim real Jev performance until a live Jev run exists.

## File Structure

```text
src/jev_lab/
├── contracts.py                    # optional classifier provenance on RunManifest
├── cli.py                          # train command and classifier provider selection
├── reporting.py                    # display fixed local-model provenance
├── providers/
│   └── classifier.py               # frozen pipeline → DecisionResult adapter
└── training/
    ├── __init__.py
    ├── artifacts.py                # manifest schema, hashes, verified persistence
    └── classifier.py               # validation, grouped CV and model fitting
tests/
├── training/
│   ├── test_artifacts.py
│   └── test_classifier_training.py
├── providers/test_classifier.py
├── test_classifier_cli.py
└── test_reporting.py
docs/research/
├── classifier-change-cost-protocol.md
└── jev-vs-classifier-draft.md
```

## Review Focus

- A caller passes one calibration sample among dev samples: training must fail before creating an artifact directory; pinned in Task 2.
- A model file is changed after training: loading must fail on SHA-256 mismatch rather than execute it; pinned in Task 1.
- scikit-learn returns classes in an unexpected order: the provider must map probabilities by `classes_`, not enum order; pinned in Task 3.
- `--provider tfidf-logreg` is used without `--model`: CLI must return a clear usage error without creating a run; pinned in Task 4.
- The default offline job does not install classifier dependencies: importing ordinary CLI/rules functionality must still succeed; pinned in Task 6.

## Spec Coverage

- 公平性与数据边界：Tasks 2、4 和 5 固定训练、校准、测试的职责并禁止跨数据集比较。
- 模型与可追溯产物：Tasks 1、2 和 3 实现训练、哈希验证和统一 Provider。
- 指标、成本与失败边界：Tasks 3、4 和 5 保留错误状态并公开建设、运行证据，不生成综合冠军。
- 标签变化实验：Task 7 预注册变更成本协议，不在样本不足时制造成绩。
- 测试与 CI：Tasks 1-6 逐项 TDD，Task 8 执行整体验证和审查。
- 网站沉淀：Task 7 生成网站文章源稿，但保留人工发布门。

---

### Task 1: Define immutable classifier artifacts

**Files:**
- Modify: `pyproject.toml`
- Create: `src/jev_lab/training/__init__.py`
- Create: `src/jev_lab/training/artifacts.py`
- Create: `tests/training/test_artifacts.py`

**Interfaces:**
- Produces: `ClassifierArtifactMetadata`, `ClassifierTrainingManifest`, `sha256_file(path: Path) -> str`, `write_artifact(*, pipeline: object, metadata: ClassifierArtifactMetadata, output_dir: Path) -> Path`, and `load_verified_pipeline(model_dir: Path) -> tuple[object, ClassifierTrainingManifest]`.
- Consumes: `RouteLabel` and `Split` from `jev_lab.contracts`.

- [ ] **Step 1: Add the optional dependency contract and failing manifest/hash tests**

Add to `pyproject.toml`:

```toml
classifier = [
  "scikit-learn>=1.5,<2",
  "joblib>=1.4,<2",
]
```

Create tests that construct a manifest with these exact fields and that mutate `model.joblib` after writing:

```python
def test_verified_load_rejects_changed_model(tmp_path: Path) -> None:
    model_dir = write_test_artifact(tmp_path)
    (model_dir / "model.joblib").write_bytes(b"changed")
    with pytest.raises(ValueError, match="model hash mismatch"):
        load_verified_pipeline(model_dir)


def test_artifact_writer_refuses_existing_directory(tmp_path: Path) -> None:
    target = tmp_path / "same-id"
    write_test_artifact(tmp_path, model_id="same-id")
    with pytest.raises(FileExistsError):
        write_test_artifact(tmp_path, model_id="same-id")
    assert target.exists()
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `.venv/bin/pytest tests/training/test_artifacts.py -v`
Expected: collection fails because `jev_lab.training.artifacts` does not exist.

- [ ] **Step 3: Implement the manifest and verified persistence**

Define `ClassifierArtifactMetadata` with all provenance supplied by training—`model_id`, timestamps, dataset fields, sample/family IDs, labels, parameters, seed, duration and commit—and define the persisted manifest by extending it with the generated file facts:

```python
class ClassifierTrainingManifest(BaseModel):
    model_id: str
    provider: Literal["tfidf-logreg"]
    created_at: datetime
    dataset_path: str
    dataset_sha256: str
    training_split: Literal["dev"]
    training_sample_ids: list[str]
    training_family_ids: list[str]
    labels: list[RouteLabel]
    sklearn_version: str
    tfidf_params: dict[str, JSONValue]
    classifier_params: dict[str, JSONValue]
    random_seed: int
    training_duration_ms: float = Field(ge=0)
    model_size_bytes: int = Field(ge=1)
    model_sha256: str
    git_commit: str
```

`write_artifact` writes to `<model-id>.tmp`, serializes with `joblib.dump`, calculates the final bytes and hash, writes JSON files, then atomically renames the directory. It deletes the temporary directory on failure but never deletes or overwrites the final target. `load_verified_pipeline` validates all required files and the hash before calling `joblib.load`.

The hash detects accidental replacement; it does not make pickle/joblib safe for untrusted downloads. `load_verified_pipeline` documentation and CLI help must say that only artifacts created locally by this lab may be loaded.

- [ ] **Step 4: Run tests and static checks**

Run:

```bash
.venv/bin/python -m pip install -e '.[dev,classifier]'
.venv/bin/pytest tests/training/test_artifacts.py -v
.venv/bin/ruff check src/jev_lab/training tests/training
.venv/bin/mypy src
```

Expected: all commands pass.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/jev_lab/training tests/training/test_artifacts.py
git commit -m "feat: add verified classifier artifacts"
```

---

### Task 2: Train and cross-validate without split leakage

**Files:**
- Create: `src/jev_lab/training/classifier.py`
- Create: `tests/training/test_classifier_training.py`

**Interfaces:**
- Consumes: `write_artifact`, `ClassifierTrainingManifest`, `RoutingSample`, `Split`, `dataset_sha256`.
- Produces: `validate_training_samples(samples: list[RoutingSample]) -> None`, `cross_validate_classifier(samples: list[RoutingSample], folds: int = 3, random_seed: int = 42) -> CrossValidationResult`, and `train_classifier(samples: list[RoutingSample], dataset_path: Path, output_dir: Path, model_id: str, git_commit: str, random_seed: int = 42) -> Path`.

- [ ] **Step 1: Write failing leakage, grouping and reproducibility tests**

Use small synthetic `RoutingSample` fixtures and assert:

```python
def test_training_rejects_non_dev_before_writing(tmp_path: Path) -> None:
    samples = balanced_dev_samples()
    samples.append(samples[0].model_copy(update={"sample_id": "cal-1", "split": Split.CALIBRATION}))
    with pytest.raises(ValueError, match="dev samples only"):
        train_classifier(samples, dataset_file(tmp_path), tmp_path / "artifacts", "m1", "abc")
    assert not (tmp_path / "artifacts").exists()


def test_cross_validation_keeps_families_in_one_fold() -> None:
    result = cross_validate_classifier(balanced_family_samples(), folds=2, random_seed=42)
    for fold in result.folds:
        assert set(fold.training_family_ids).isdisjoint(fold.validation_family_ids)


def test_same_seed_produces_same_cross_validation_scores() -> None:
    first = cross_validate_classifier(balanced_family_samples(), folds=2, random_seed=42)
    second = cross_validate_classifier(balanced_family_samples(), folds=2, random_seed=42)
    assert first == second
```

Also test empty input, missing label classes, duplicate sample IDs and too few families for the requested fold count.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `.venv/bin/pytest tests/training/test_classifier_training.py -v`
Expected: FAIL because the training functions do not exist.

- [ ] **Step 3: Implement the fixed baseline**

Use this pipeline in one factory shared by fitting and cross-validation:

```python
Pipeline(
    [
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1)),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=random_seed,
            ),
        ),
    ]
)
```

Define `CrossValidationFold` with `fold_index`, `accuracy`, training/validation sample IDs and training/validation family IDs. Define `CrossValidationResult` with `folds`, `mean_accuracy`, `standard_deviation`, `random_seed` and `fold_count`.

Use `StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=random_seed)` with `family_id` as groups. The default is 3 folds because the current dev split contains only 3 `human_review` families; reject any fold count greater than the smallest per-label family count. Store fold accuracy, training sample IDs, validation sample IDs and both family-ID sets in `CrossValidationResult`. Validate all samples before creating `output_dir`. Train the final model on all dev samples, build `ClassifierArtifactMetadata`, and pass it with the fitted pipeline to `write_artifact`.

- [ ] **Step 4: Run focused and full classifier tests**

Run:

```bash
.venv/bin/pytest tests/training -v
.venv/bin/ruff check src tests
.venv/bin/mypy src
```

Expected: all commands pass.

- [ ] **Step 5: Commit**

```bash
git add src/jev_lab/training/classifier.py tests/training/test_classifier_training.py
git commit -m "feat: train grouped offline classifier baseline"
```

---

### Task 3: Adapt the frozen classifier to DecisionProvider

**Files:**
- Create: `src/jev_lab/providers/classifier.py`
- Create: `tests/providers/test_classifier.py`

**Interfaces:**
- Consumes: `load_verified_pipeline(model_dir)` and existing `DecisionResult`, `RoutingSample`, `RunMode`.
- Produces: `ClassifierProvider.from_artifact(model_dir: Path) -> ClassifierProvider` and `decide(sample: RoutingSample) -> DecisionResult`.

- [ ] **Step 1: Write failing provider contract tests**

Build a fake pipeline whose `classes_` order is deliberately different from `RouteLabel` order:

```python
def test_provider_maps_probabilities_using_pipeline_classes() -> None:
    pipeline = FakePipeline(
        classes=["human_review", "search", "database", "code"],
        probabilities=[[0.1, 0.6, 0.2, 0.1]],
    )
    provider = ClassifierProvider(pipeline, manifest())
    result = provider.decide(sample())
    assert result.label == RouteLabel.SEARCH
    assert result.probabilities == {
        RouteLabel.HUMAN_REVIEW: 0.1,
        RouteLabel.SEARCH: 0.6,
        RouteLabel.DATABASE: 0.2,
        RouteLabel.CODE: 0.1,
    }
```

Also test missing class, duplicate class, non-finite probability, prediction exception and `run_mode == RunMode.OFFLINE_DEVELOPMENT`.

- [ ] **Step 2: Run the provider tests and verify RED**

Run: `.venv/bin/pytest tests/providers/test_classifier.py -v`
Expected: FAIL because `ClassifierProvider` does not exist.

- [ ] **Step 3: Implement the provider**

`ClassifierProvider` exposes:

```python
name = "tfidf-logreg"
run_mode = RunMode.OFFLINE_DEVELOPMENT
model_version = manifest.model_id
```

`decide` measures prediction-only latency, calls `predict_proba([sample.input])`, maps values using the fitted classifier's `classes_`, validates the exact `RouteLabel` set, selects the maximum probability, and returns cost `0.0`. Expected model/output errors become `RequestStatus.FAILED` with stable categories such as `invalid_model_output` or `prediction_error`; secrets and exception messages are not copied into results.

- [ ] **Step 4: Run focused and provider suites**

Run:

```bash
.venv/bin/pytest tests/providers -v
.venv/bin/ruff check src tests
.venv/bin/mypy src
```

Expected: all commands pass.

- [ ] **Step 5: Commit**

```bash
git add src/jev_lab/providers/classifier.py tests/providers/test_classifier.py
git commit -m "feat: add frozen classifier provider"
```

---

### Task 4: Add train and classifier run CLI paths

**Files:**
- Modify: `src/jev_lab/cli.py`
- Modify: `src/jev_lab/contracts.py`
- Modify: `tests/test_cli.py`
- Create: `tests/test_classifier_cli.py`

**Interfaces:**
- Consumes: `train_classifier`, `cross_validate_classifier`, `ClassifierProvider.from_artifact`.
- Produces: `jev-lab train --provider tfidf-logreg --split dev` and `jev-lab run --provider tfidf-logreg --model <dir> --split <split>`.

- [ ] **Step 1: Write failing CLI behavior tests**

Add tests for help exposure and conditional model validation:

```python
def test_classifier_run_requires_model_without_creating_run(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        ["run", "--provider", "tfidf-logreg", "--split", "dev", "--output-dir", str(tmp_path)],
    )
    assert result.exit_code != 0
    assert "--model is required" in result.output
    assert not list(tmp_path.iterdir())


def test_train_rejects_calibration_before_artifact_creation(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        ["train", "--provider", "tfidf-logreg", "--split", "calibration", "--output-dir", str(tmp_path)],
    )
    assert result.exit_code != 0
    assert not tmp_path.exists()
```

Add an end-to-end test that trains on the real dev split, runs on calibration, evaluates the run, and confirms `probability_sample_count == 20`.

- [ ] **Step 2: Run the new CLI tests and verify RED**

Run: `.venv/bin/pytest tests/test_classifier_cli.py tests/test_cli.py -v`
Expected: FAIL because `train` and `tfidf-logreg` are not registered.

- [ ] **Step 3: Implement CLI wiring and provenance fields**

Extend `RunManifest` with backward-compatible optional fields:

```python
model_artifact_sha256: str | None = None
training_sample_count: int | None = Field(default=None, ge=1)
training_duration_ms: float | None = Field(default=None, ge=0)
model_size_bytes: int | None = Field(default=None, ge=1)
```

Add `train` with literal provider `tfidf-logreg`, a `Split` option defaulting to `dev`, `--output-dir`, `--model-id`, `--seed`, and `--folds`. Reject a non-dev split before loading or writing. Print the artifact path and grouped cross-validation summary.

Extend `run` with provider literal `tfidf-logreg` and optional `--model`. Validate conditional requirements before creating `run_id`. Load the artifact, populate optional provenance fields from its manifest, and retain existing behavior for every other Provider.

- [ ] **Step 4: Run CLI, contract and offline regression suites**

Run:

```bash
.venv/bin/pytest tests/test_classifier_cli.py tests/test_cli.py tests/test_contracts.py tests/test_offline_acceptance.py -v
.venv/bin/ruff check src tests
.venv/bin/mypy src
```

Expected: all commands pass.

- [ ] **Step 5: Commit**

```bash
git add src/jev_lab/cli.py src/jev_lab/contracts.py tests/test_cli.py tests/test_classifier_cli.py
git commit -m "feat: expose classifier training and evaluation CLI"
```

---

### Task 5: Report local-model engineering costs

**Files:**
- Modify: `src/jev_lab/reporting.py`
- Modify: `tests/test_reporting.py`
- Create: `src/jev_lab/comparison.py`
- Create: `tests/test_comparison.py`

**Interfaces:**
- Consumes: optional classifier provenance from `RunManifest` and existing `EvaluationMetrics`.
- Produces: `ComparisonRow` and `compare_runs(run_dirs: list[Path]) -> list[ComparisonRow]` without altering predictions.

- [ ] **Step 1: Write failing report and comparison tests**

Extend the reporting fixture with classifier provenance and assert both Markdown and public JSON contain the same values. Add:

```python
def test_compare_runs_keeps_missing_costs_explicit(tmp_path: Path) -> None:
    classifier_run = build_run(tmp_path / "classifier", provider="tfidf-logreg", api_cost=0.0)
    jev_run = build_run(tmp_path / "jev", provider="jev", api_cost=None)
    rows = compare_runs([classifier_run, jev_run])
    assert rows[0].mean_known_cost_usd == 0.0
    assert rows[1].mean_known_cost_usd is None
```

Also assert comparisons reject different dataset hashes or splits, so unrelated experiments cannot be ranked together.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `.venv/bin/pytest tests/test_reporting.py tests/test_comparison.py -v`
Expected: FAIL because provenance display and comparison module are absent.

- [ ] **Step 3: Implement provenance display and strict comparisons**

Add a `Local model provenance` report section containing artifact hash, training sample count, training duration and model size. Preserve `None` as `not available`; do not convert it to zero.

Define `ComparisonRow` as a Pydantic model with `provider`, `model_version`, `run_mode`, `dataset_sha256`, `split`, `threshold`, `accuracy`, `coverage`, `selective_accuracy`, `brier_score`, `p50_latency_ms`, `p95_latency_ms`, `mean_known_cost_usd`, `training_sample_count`, `training_duration_ms` and `model_size_bytes`. Numeric evidence fields remain optional except `coverage`. `compare_runs` validates a common dataset hash and split and returns rows in caller order; it does not compute an aggregate winner.

- [ ] **Step 4: Run reporting, security and metric suites**

Run:

```bash
.venv/bin/pytest tests/test_reporting.py tests/test_comparison.py tests/test_security.py tests/test_metrics.py -v
.venv/bin/ruff check src tests
.venv/bin/mypy src
```

Expected: all commands pass.

- [ ] **Step 5: Commit**

```bash
git add src/jev_lab/reporting.py src/jev_lab/comparison.py tests/test_reporting.py tests/test_comparison.py
git commit -m "feat: compare decision providers with training provenance"
```

---

### Task 6: Add isolated classifier CI and prove default offline compatibility

**Files:**
- Modify: `.github/workflows/ci.yml`
- Create: `.coveragerc.offline`
- Create: `.coveragerc.classifier`
- Modify: `pyproject.toml`
- Modify: `tests/test_offline_acceptance.py`
- Create: `tests/test_ci_config.py`
- Modify: `tests/test_classifier_cli.py`
- Modify: `tests/training/test_classifier_training.py`
- Modify: `tests/training/test_artifacts.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: all previous tasks.
- Produces: separate `offline` and `classifier` CI evidence.

- [ ] **Step 1: Strengthen the default-install acceptance test**

Add a subprocess assertion that imports and runs rule CLI functionality while blocking classifier imports through a clean interpreter environment. It must not import `sklearn` merely by importing `jev_lab.cli`; classifier imports must remain lazy until `train` or classifier run is selected.

- [ ] **Step 2: Run the default offline suite before CI changes**

Training-module tests must call `pytest.importorskip("sklearn")` before importing optional training code; artifact-persistence tests must likewise skip on missing `joblib`. Mark those tests `classifier`. This keeps pytest collection usable in the base `[dev]` environment. Keep classifier CLI integration tests marked; also retain dependency-free fake-dispatch tests for early validation and lazy CLI wiring.

Run:

```bash
.venv/bin/pytest -m 'not live_deepseek and not live_jev and not classifier' -q
```

Expected: all selected tests pass with existing dependencies.

- [ ] **Step 3: Add the classifier CI job**

Add:

```yaml
classifier:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: "3.11"
        cache: pip
    - run: python -m pip install -e '.[dev,classifier]'
    - run: pytest -m 'not live_deepseek and not live_jev' --cov=jev_lab --cov-config=.coveragerc.classifier --cov-fail-under=90
    - run: |
        jev-lab train --provider tfidf-logreg --split dev --model-id ci-model --output-dir artifacts
        jev-lab run --provider tfidf-logreg --model artifacts/ci-model --split calibration --output-dir runs
```

Keep the existing `offline` job on `.[dev]`; do not add secrets or live markers. The offline job excludes optional classifier tests and uses `.coveragerc.offline` to measure the base-install surface without counting modules that cannot be imported without the extra. The classifier job runs the full non-live suite and measures all `jev_lab` modules using `.coveragerc.classifier`.

- [ ] **Step 4: Document the exact local workflow**

Update `README.md` with:

```bash
python -m pip install -e '.[dev,classifier]'
jev-lab train --provider tfidf-logreg --split dev --model-id routing-v1-s42
jev-lab run --provider tfidf-logreg --model artifacts/routing-v1-s42 --split calibration
```

Explain that cross-validation is development evidence, calibration selects a threshold, and held-out test may be run only after freezing the model and threshold.

- [ ] **Step 5: Run the complete local equivalent of both jobs**

Run:

```bash
.venv/bin/ruff check .
.venv/bin/mypy src
PYTHONPATH=src ../../.venv/bin/pytest -m 'not live_deepseek and not live_jev and not classifier' --cov=jev_lab --cov-config=.coveragerc.offline --cov-fail-under=90
.venv/bin/jev-lab dataset validate
.venv/bin/pytest -m 'not live_deepseek and not live_jev' --cov=jev_lab --cov-config=.coveragerc.classifier --cov-fail-under=90
```

Expected: all commands pass and coverage remains at least 90%.

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/ci.yml tests/test_offline_acceptance.py README.md
git commit -m "ci: verify classifier baseline independently"
```

---

### Task 7: Document the label-change experiment and publication boundary

**Files:**
- Create: `docs/research/classifier-change-cost-protocol.md`
- Create: `docs/research/jev-vs-classifier-draft.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: frozen experiment commands, manifests and comparison rows.
- Produces: a preregistered change-cost protocol and a website-ready draft containing no invented live results.

- [ ] **Step 1: Write the change-cost protocol**

Specify that `human_review` is hypothetically split into `security_review` and `business_review`. For each Provider, record the required label work, code/config changes, retraining, recalibration, validation commands and elapsed engineering steps. State that no accuracy result exists until a separately versioned dataset contains enough labeled examples; do not add unsupported synthetic scores.

- [ ] **Step 2: Write the article draft**

Use these sections:

```text
1. Transformer、分类器和 Jev 为什么不是同层概念
2. Jev 与动态分类器的相似之处
3. 公平比较为什么不能只看准确率
4. 数据切分与防泄漏协议
5. 建设、变更和运行成本
6. 当前已获得的离线证据
7. 尚未获得的 Jev live evidence
8. 下一次更新需要哪些运行 ID 和哈希
```

Include links to the design, implementation plan and public GitHub repository. Future result sections must display the literal status `尚无 live evidence`—never fake numbers or generic implementation markers.

- [ ] **Step 3: Add research-document links to README**

Link both documents under a `研究协议` section and state that they are source material for manual publication to Agent 工程笔记, not an automatic deployment channel.

- [ ] **Step 4: Verify documentation boundaries**

Run:

```bash
rg -n "193\.6|444\.6|live|实测|生产就绪|优于" docs/research README.md
git diff --check
```

Inspect every match and confirm it is either a sourced vendor claim, an explicit absence statement, or actual run evidence with commit/hash. Expected: no unsupported performance claim.

- [ ] **Step 5: Commit**

```bash
git add docs/research README.md
git commit -m "docs: preregister classifier comparison study"
```

---

### Task 8: Final verification and review handoff

**Files:**
- Review only: all files changed by Tasks 1-7

**Interfaces:**
- Consumes: complete branch.
- Produces: fresh verification evidence and a review-ready branch; no new features.

- [ ] **Step 1: Run all quality gates from a clean shell**

```bash
.venv/bin/python -m pip install -e '.[dev,classifier]'
.venv/bin/ruff check .
.venv/bin/mypy src
PYTHONPATH=src ../../.venv/bin/pytest -m 'not live_deepseek and not live_jev and not classifier' --cov=jev_lab --cov-config=.coveragerc.offline --cov-fail-under=90
.venv/bin/pytest -m 'not live_deepseek and not live_jev' --cov=jev_lab --cov-config=.coveragerc.classifier --cov-fail-under=90
.venv/bin/jev-lab dataset validate
```

Expected: zero failures, zero lint/type errors and coverage at least 90% in each dependency lane.

- [ ] **Step 2: Run a disposable end-to-end classifier experiment**

```bash
tmp_dir="$(mktemp -d)"
.venv/bin/jev-lab train --provider tfidf-logreg --split dev --model-id smoke --output-dir "$tmp_dir/artifacts"
.venv/bin/jev-lab run --provider tfidf-logreg --model "$tmp_dir/artifacts/smoke" --split calibration --output-dir "$tmp_dir/runs"
run_dir="$(find "$tmp_dir/runs" -mindepth 1 -maxdepth 1 -type d | head -1)"
.venv/bin/jev-lab evaluate --run "$run_dir"
.venv/bin/jev-lab report --run "$run_dir" --report-dir "$tmp_dir/reports" --public-dir "$tmp_dir/public"
```

Expected: training, inference, evaluation and report commands exit zero; the public JSON contains aggregate metrics and classifier provenance but no sample inputs.

- [ ] **Step 3: Inspect repository state and generated-file safety**

```bash
git status --short
git diff --check main...HEAD
git log --oneline main..HEAD
```

Expected: only intended source, test and documentation changes; no artifacts, runs, reports, keys, caches or raw responses tracked.

- [ ] **Step 4: Request whole-branch code review**

Review against `docs/superpowers/specs/2026-09-22-classifier-baseline-design.md`, paying special attention to split leakage, unsafe joblib loading, artifact/dataset identity, label metadata integrity, probability/class ordering, CLI side effects and unsupported public claims. Address accepted Important/Critical findings with TDD and repeat Steps 1-3; document deferred Minor findings.

- [ ] **Step 5: Commit review fixes if any**

```bash
git add src tests docs README.md .github/workflows/ci.yml pyproject.toml
git commit -m "fix: address classifier baseline review"
```

Do not create an empty commit when no fixes are required.
