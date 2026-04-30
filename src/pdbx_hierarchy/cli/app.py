"""Typer CLI application for pdbx_hierarchy."""

import typer

app = typer.Typer(
    name="pdbx-hierarchy",
    help="Read, write, and modify PDBx/mmCIF files with hierarchical heterogeneity extensions.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """PDBx Hierarchy CLI tool."""
    pass
