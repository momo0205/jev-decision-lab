from pathlib import Path

import typer

from jev_lab.dataset import dataset_sha256, load_dataset, validate_dataset

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
