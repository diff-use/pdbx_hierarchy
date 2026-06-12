"""The ``validate`` command: check hierarchy, coexistence, and atom references."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from pdbx_hierarchy.cli.commands._utils import error_handler, fail
from pdbx_hierarchy.io.reader import has_hierarchy, read_mmcif
from pdbx_hierarchy.io.validation import validate_file


def validate(
    file: Annotated[Path, typer.Argument(help="mmCIF file to validate.")],
    strict: Annotated[bool, typer.Option("--strict", "-s", help="Report only the first error instead of all.")] = False,
) -> None:
    """Validate hierarchy structure, coexistence references, and atom assignments."""
    with error_handler():
        block = read_mmcif(file)
        if not has_hierarchy(block):
            typer.echo("No hierarchy table found; nothing to validate.")
            return

        # Always collect every error so the output format is identical in both
        # modes; --strict only trims the displayed list to the first error.
        errors = validate_file(block, raise_on_error=False)
        if errors:
            for message in errors[:1] if strict else errors:
                typer.secho(message, fg=typer.colors.RED, err=True)
            fail(f"Validation failed with {len(errors)} error(s).")
        typer.echo("Validation passed.")
