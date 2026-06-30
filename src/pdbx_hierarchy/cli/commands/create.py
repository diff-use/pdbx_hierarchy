"""The ``infer`` and ``import`` commands: add a hierarchy to a plain file."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from pdbx_hierarchy.assignment import assign_from_alt_ids
from pdbx_hierarchy.cli.commands._utils import error_handler, load_document, resolve_output, warn
from pdbx_hierarchy.io.reader import has_hierarchy
from pdbx_hierarchy.io.validation import validate_file
from pdbx_hierarchy.io.writer import write_atom_site_heterogeneity_ids, write_hierarchy, write_mmcif
from pdbx_hierarchy.models.hierarchy import HierarchyTree


def infer(
    input_file: Annotated[Path, typer.Argument(help="Plain mmCIF file to infer a hierarchy from.")],
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Output path (default: <name>_pdbx_N).")] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the overwrite prompt.")] = False,
) -> None:
    """Infer a hierarchy from _atom_site.label_alt_id and write it with assignments."""
    with error_handler():
        doc, block = load_document(input_file)
        if has_hierarchy(block):
            warn("input already contains a hierarchy; it will be replaced by the inferred one")
        tree, ids = assign_from_alt_ids(block)
        out = resolve_output(input_file, output, yes=yes)
        write_hierarchy(block, tree, overwrite=True)
        write_atom_site_heterogeneity_ids(block, ids, overwrite=True)
        write_mmcif(doc, out)
        typer.echo(f"Inferred {len(tree)} state(s); wrote {out}")


def import_spec(
    input_file: Annotated[Path, typer.Argument(help="mmCIF file to write the hierarchy into.")],
    spec: Annotated[Path, typer.Option("--spec", help="JSON file describing a HierarchyTree.")],
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Output path (default: <name>_pdbx_N).")] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the overwrite prompt.")] = False,
) -> None:
    """Apply a hierarchy described by a JSON spec to a file (no atom assignment)."""
    with error_handler():
        doc, block = load_document(input_file)
        if has_hierarchy(block):
            warn("input already contains a hierarchy; it will be replaced by the imported one")
        tree = HierarchyTree.model_validate_json(spec.read_text())
        out = resolve_output(input_file, output, yes=yes)
        write_hierarchy(block, tree, overwrite=True)
        # The spec doesn't touch _atom_site, so existing assignments may now point
        # at states the imported tree doesn't define; surface those rather than
        # writing a silently-invalid file.
        for message in validate_file(block, raise_on_error=False):
            warn(message)
        write_mmcif(doc, out)
        typer.echo(f"Imported {len(tree)} state(s); wrote {out}")
