"""Typer CLI application for pdbx_hierarchy."""

import typer

from pdbx_hierarchy.cli.commands import clash, coexist, create, hierarchy, show, validate

app = typer.Typer(
    name="pdbx-hierarchy",
    help="Read, write, and modify PDBx/mmCIF files with hierarchical heterogeneity extensions.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """PDBx Hierarchy CLI tool."""


app.command("show")(show.show)
app.command("validate")(validate.validate)
app.command("infer")(create.infer)
app.command("import")(create.import_spec)
app.add_typer(hierarchy.app, name="hierarchy")
app.add_typer(coexist.app, name="coexist")
app.add_typer(clash.app, name="clash")
