import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

import typer

from jev_lab.contracts import RunManifest, Split
from jev_lab.dataset import dataset_sha256, load_dataset, validate_dataset
from jev_lab.providers.rules import RulesProvider
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
    provider: Annotated[Literal["rules"], typer.Option()],
    split: Annotated[Split, typer.Option()],
    output_dir: Annotated[Path, typer.Option()] = Path("runs"),
) -> None:
    samples = [sample for sample in load_dataset(DATASET) if sample.split == split]
    selected = RulesProvider()
    timestamp = datetime.now(UTC)
    run_id = f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}-{provider}-{split.value}"
    commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True,
                            text=True).stdout.strip()
    manifest = RunManifest(run_id=run_id, provider=selected.name,
        model_version=selected.model_version, run_mode=selected.run_mode, split=split,
        dataset_path=str(DATASET), dataset_sha256=dataset_sha256(DATASET), git_commit=commit,
        created_at=timestamp)
    artifact = run_experiment(selected, samples, output_dir, manifest)
    typer.echo(str(artifact.parent))
