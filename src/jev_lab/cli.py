import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

import typer

from jev_lab.contracts import DecisionResult, RunManifest, Split
from jev_lab.dataset import dataset_sha256, load_dataset, validate_dataset
from jev_lab.metrics import evaluate
from jev_lab.providers.base import DecisionProvider
from jev_lab.providers.deepseek import DeepSeekProvider
from jev_lab.providers.jev import JevProvider
from jev_lab.providers.recorded import RecordedProvider
from jev_lab.providers.rules import RulesProvider
from jev_lab.reporting import write_report
from jev_lab.runner import run_experiment

DATASET = Path("datasets/routing-v1.yaml")

app = typer.Typer(help="Jev Decision Lab")
dataset_app = typer.Typer(help="Validate and inspect versioned datasets")
app.add_typer(dataset_app, name="dataset")


@dataset_app.callback(invoke_without_command=True)
def dataset_root() -> None:
    """Dataset commands are added by the dataset task."""


@dataset_app.command("validate")
def validate(path: Path = Path("datasets/routing-v1.yaml")) -> None:
    samples = load_dataset(path)
    validate_dataset(samples)
    typer.echo(f"valid: {len(samples)} samples sha256={dataset_sha256(path)}")


@app.command("run")
def run(
    provider: Annotated[Literal["rules", "deepseek", "jev", "recorded"], typer.Option()],
    split: Annotated[Split, typer.Option()],
    output_dir: Annotated[Path, typer.Option()] = Path("runs"),
) -> None:
    samples = [sample for sample in load_dataset(DATASET) if sample.split == split]
    selected: DecisionProvider
    if provider == "rules":
        selected = RulesProvider()
    elif provider == "deepseek":
        selected = DeepSeekProvider.from_environment()
    elif provider == "jev":
        selected = JevProvider.from_environment()
    else:
        selected = RecordedProvider(
            Path("tests/fixtures/recorded/jev-routing.jsonl"), "jev", "synthetic-contract-fixture"
        )
    timestamp = datetime.now(UTC)
    run_id = f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}-{provider}-{split.value}"
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    manifest = RunManifest(
        run_id=run_id,
        provider=selected.name,
        model_version=selected.model_version,
        run_mode=selected.run_mode,
        split=split,
        dataset_path=str(DATASET),
        dataset_sha256=dataset_sha256(DATASET),
        git_commit=commit,
        created_at=timestamp,
    )
    artifact = run_experiment(selected, samples, output_dir, manifest)
    typer.echo(str(artifact.parent))


@app.command("evaluate")
def evaluate_run(run: Annotated[Path, typer.Option(exists=True, file_okay=False)]) -> None:
    manifest = RunManifest.model_validate_json((run / "manifest.json").read_text())
    dataset_path = Path(manifest.dataset_path)
    if dataset_sha256(dataset_path) != manifest.dataset_sha256:
        raise typer.BadParameter("dataset hash mismatch")
    samples = [sample for sample in load_dataset(dataset_path) if sample.split == manifest.split]
    results = [
        DecisionResult.model_validate_json(line)
        for line in (run / "results.jsonl").read_text().splitlines()
    ]
    metrics = evaluate(samples, results, manifest.threshold)
    (run / "metrics.json").write_text(metrics.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(
        json.dumps(
            {
                "accuracy": metrics.accuracy,
                "coverage": metrics.coverage,
                "success_rate": metrics.success_rate,
            }
        )
    )


@app.command("report")
def report_run(
    run: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    report_dir: Annotated[Path, typer.Option()] = Path("reports"),
    public_dir: Annotated[Path, typer.Option()] = Path("public"),
) -> None:
    markdown, public_json = write_report(run, report_dir, public_dir)
    typer.echo(f"report={markdown} public={public_json}")
