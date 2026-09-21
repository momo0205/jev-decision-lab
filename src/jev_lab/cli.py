import typer

app = typer.Typer(help="Jev Decision Lab")
dataset_app = typer.Typer(help="Validate and inspect versioned datasets")
app.add_typer(dataset_app, name="dataset")


@dataset_app.callback(invoke_without_command=True)
def dataset_root() -> None:
    """Dataset commands are added by the dataset task."""
