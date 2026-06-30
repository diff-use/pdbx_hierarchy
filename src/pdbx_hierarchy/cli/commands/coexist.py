"""The ``coexist`` sub-app: add and remove coexistence rules."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from pdbx_hierarchy.cli.commands._utils import error_handler, load_document, resolve_output
from pdbx_hierarchy.exceptions import PdbxValidationError
from pdbx_hierarchy.io.reader import read_coexistence, read_hierarchy
from pdbx_hierarchy.io.validation import validate_file
from pdbx_hierarchy.io.writer import write_coexistence, write_mmcif
from pdbx_hierarchy.models.coexistence import CoexistenceRule, CoexistenceTable, StateCoexistence

app = typer.Typer(name="coexist", help="Add or remove coexistence rules between hierarchy states.")

_FileArg = Annotated[Path, typer.Argument(help="mmCIF file to modify.")]
_OutputOpt = Annotated[Path | None, typer.Option("--output", "-o", help="Output path (default: <name>_pdbx_N).")]
_YesOpt = Annotated[bool, typer.Option("--yes", "-y", help="Skip the overwrite prompt.")]


@app.command("add")
def add(
    file: _FileArg,
    rule: Annotated[CoexistenceRule, typer.Option("--rule", help="Coexistence rule type.")],
    source: Annotated[str, typer.Option("--source", help="Source hierarchy state id (heterogeneity_id).")],
    related: Annotated[str, typer.Option("--related", help="Comma-separated related state ids.")],
    description: Annotated[str | None, typer.Option("--description", help="Optional description.")] = None,
    output: _OutputOpt = None,
    yes: _YesOpt = False,
) -> None:
    """Add a coexistence rule, validating its references against the hierarchy."""
    with error_handler():
        doc, block = load_document(file)
        tree = read_hierarchy(block)
        validate_file(block, raise_on_error=True)
        table = read_coexistence(block) or CoexistenceTable()

        related_ids = [part.strip() for part in related.split(",") if part.strip()]
        if not related_ids:
            raise PdbxValidationError("--related must list at least one state id")
        # De-duplicate (preserving order) and reject a self-reference: both are
        # degenerate rules the remap path would otherwise silently rewrite/drop,
        # so reject them at the source instead of writing them out.
        related_ids = list(dict.fromkeys(related_ids))
        if source in related_ids:
            raise PdbxValidationError(f"--related must not contain the source state {source!r} (self-reference)")

        next_id = max((existing.id for existing in table.rules), default=0) + 1
        table.add_rule(
            StateCoexistence(
                id=next_id,
                rule=rule,
                heterogeneity_id=source,
                heterogeneity_ids=related_ids,
                description=description,
            )
        )
        table.validate_against_hierarchy(tree, raise_on_error=True)

        out = resolve_output(file, output, yes=yes)
        write_coexistence(block, table, overwrite=True)
        write_mmcif(doc, out)
        typer.echo(f"Added coexistence rule {next_id} ({rule.value}); wrote {out}")


@app.command("remove")
def remove(
    file: _FileArg,
    rule_id: Annotated[int, typer.Option("--id", help="Id of the coexistence rule to remove.")],
    output: _OutputOpt = None,
    yes: _YesOpt = False,
) -> None:
    """Remove a coexistence rule by id."""
    with error_handler():
        doc, block = load_document(file)
        table = read_coexistence(block)
        if table is None:
            raise PdbxValidationError("no coexistence table found")
        table.remove_rule(rule_id)
        out = resolve_output(file, output, yes=yes)
        write_coexistence(block, table, overwrite=True)
        write_mmcif(doc, out)
        typer.echo(f"Removed coexistence rule {rule_id}; wrote {out}")
