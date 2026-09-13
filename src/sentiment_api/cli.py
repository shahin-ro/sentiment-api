"""CLI for training and serving the sentiment model."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from sentiment_api import __version__
from sentiment_api.config import DEFAULT_DATA_PATH, DEFAULT_MODEL_PATH
from sentiment_api.datasets import write_sample_csv
from sentiment_api.predict import predict_text
from sentiment_api.train import load_model, train_model

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Train and serve a text sentiment classifier.",
)
console = Console(safe_box=True)


@app.command("version")
def version() -> None:
    console.print(f"sentiment-api {__version__}")


@app.command("make-sample")
def make_sample(
    output: Path = typer.Option(DEFAULT_DATA_PATH, "--output", "-o"),
) -> None:
    """Write the bundled sample reviews CSV."""
    path = write_sample_csv(output)
    console.print(f"[green]Sample data:[/green] {path}")


@app.command("train")
def train(
    data: Path = typer.Option(DEFAULT_DATA_PATH, "--data", "-d", exists=True),
    model_path: Path = typer.Option(DEFAULT_MODEL_PATH, "--model", "-m"),
    test_size: float = typer.Option(0.25, help="Holdout fraction"),
) -> None:
    """Train TF-IDF + Logistic Regression and save the pipeline."""
    result = train_model(data, model_path=model_path, test_size=test_size)
    table = Table(title="Training summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Rows", str(result.n_rows))
    table.add_row("Accuracy", f"{result.accuracy:.3f}")
    table.add_row("Macro F1", f"{result.macro_f1:.3f}")
    table.add_row("Model", str(result.model_path))
    console.print(table)


@app.command("predict")
def predict_cmd(
    text: str = typer.Argument(..., help="Text to classify"),
    model_path: Path = typer.Option(DEFAULT_MODEL_PATH, "--model", "-m"),
) -> None:
    """Classify one text locally."""
    model = load_model(model_path)
    result = predict_text(text, model)
    console.print(f"label=[bold]{result.label}[/bold] confidence={result.confidence:.3f}")
    console.print(result.probabilities)


@app.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Start the FastAPI server."""
    import uvicorn

    if not Path(DEFAULT_MODEL_PATH).exists():
        console.print(
            "[yellow]No model found. Train first:[/yellow] sentiment-api train"
        )
    uvicorn.run("sentiment_api.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
