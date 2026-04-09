from __future__ import annotations

import typer

app = typer.Typer(name="mausoleo", no_args_is_help=True)


@app.command()
def root() -> None:
    typer.echo("not implemented")


@app.command()
def node(node_id: str) -> None:
    typer.echo("not implemented")


@app.command()
def children(node_id: str) -> None:
    typer.echo("not implemented")


@app.command()
def parent(node_id: str) -> None:
    typer.echo("not implemented")


@app.command()
def text(node_id: str) -> None:
    typer.echo("not implemented")


@app.command()
def search(
    query: str,
    mode: str = typer.Option("semantic", help="semantic or text"),
    level: str | None = typer.Option(None),
    date_from: str | None = typer.Option(None, "--from"),
    date_to: str | None = typer.Option(None, "--to"),
    limit: int = typer.Option(20),
) -> None:
    typer.echo("not implemented")


@app.command()
def stats() -> None:
    typer.echo("not implemented")
