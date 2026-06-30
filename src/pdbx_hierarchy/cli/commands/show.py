"""The ``show`` command: inspect hierarchy, coexistence, and atom assignments."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

from pdbx_hierarchy.cli.commands._utils import error_handler, render_tree, warn
from pdbx_hierarchy.exceptions import HierarchyNotFoundError, InvalidHierarchyError
from pdbx_hierarchy.io.reader import read_atom_site_heterogeneity_ids, read_coexistence, read_hierarchy, read_mmcif
from pdbx_hierarchy.io.validation import validate_file
from pdbx_hierarchy.models.coexistence import CoexistenceTable
from pdbx_hierarchy.models.hierarchy import HierarchyTree


def show(
    file: Annotated[Path, typer.Argument(help="mmCIF file to inspect.")],
    tree_view: Annotated[bool, typer.Option("--tree", "-t", help="Render the hierarchy as a tree.")] = False,
    json_out: Annotated[bool, typer.Option("--json", "-j", help="Emit machine-readable JSON.")] = False,
    hierarchy: Annotated[bool, typer.Option("--hierarchy", help="Show only the hierarchy.")] = False,
    coexistence: Annotated[bool, typer.Option("--coexistence", help="Show only coexistence rules.")] = False,
    assignments: Annotated[bool, typer.Option("--assignments", help="Show only atom assignments.")] = False,
) -> None:
    """Show hierarchy, coexistence, and atom-assignment data present in a file."""
    with error_handler():
        block = read_mmcif(file)

        tree = _read_hierarchy_lenient(block)
        coex = read_coexistence(block)
        atom_ids = _read_assignments_lenient(block)

        # Implicit validation: warn but never abort, so broken files stay inspectable.
        try:
            for message in validate_file(block, raise_on_error=False):
                warn(message)
        except HierarchyNotFoundError:
            pass

        show_all = not (hierarchy or coexistence or assignments)
        want_hierarchy = show_all or hierarchy
        want_coexistence = show_all or coexistence
        want_assignments = show_all or assignments

        if json_out:
            _emit_json(tree, coex, atom_ids, want_hierarchy, want_coexistence, want_assignments)
            return

        if want_hierarchy:
            _print_hierarchy(tree, tree_view=tree_view)
        if want_coexistence:
            _print_coexistence(coex)
        if want_assignments:
            _print_assignments(atom_ids)


def _read_hierarchy_lenient(block: object) -> HierarchyTree | None:
    """Read the hierarchy, returning None (with a warning on structural errors)."""
    try:
        return read_hierarchy(block)  # type: ignore[arg-type]
    except HierarchyNotFoundError:
        return None
    except InvalidHierarchyError as exc:
        warn(f"hierarchy is structurally invalid and cannot be shown: {exc}")
        return None


def _read_assignments_lenient(block: object) -> list[str] | None:
    """Read the atom-assignment column, returning None if absent."""
    try:
        return read_atom_site_heterogeneity_ids(block)  # type: ignore[arg-type]
    except HierarchyNotFoundError:
        return None


def _emit_json(
    tree: HierarchyTree | None,
    coex: CoexistenceTable | None,
    atom_ids: list[str] | None,
    want_hierarchy: bool,
    want_coexistence: bool,
    want_assignments: bool,
) -> None:
    payload: dict[str, object] = {}
    if want_hierarchy:
        payload["hierarchy"] = json.loads(tree.model_dump_json()) if tree is not None else None
    if want_coexistence:
        payload["coexistence"] = json.loads(coex.model_dump_json()) if coex is not None else None
    if want_assignments:
        payload["assignments"] = atom_ids
    typer.echo(json.dumps(payload, indent=2))


def _print_hierarchy(tree: HierarchyTree | None, *, tree_view: bool) -> None:
    if tree is None:
        typer.echo("No hierarchy table found.")
        return
    if tree_view:
        typer.echo(render_tree(tree))
        return
    typer.echo(f"{'id':<10} {'name':<20} {'parent':<10} details")
    for state in tree.states:
        parent = state.parent if state.parent is not None else "."
        details = state.details if state.details is not None else "?"
        typer.echo(f"{state.id:<10} {state.name:<20} {parent:<10} {details}")


def _print_coexistence(coex: CoexistenceTable | None) -> None:
    if coex is None or len(coex) == 0:
        typer.echo("No coexistence table found.")
        return
    typer.echo(f"{'id':<5} {'rule':<5} {'state':<10} {'related':<20} description")
    for rule in coex.rules:
        related = ",".join(rule.heterogeneity_ids)
        description = rule.description if rule.description is not None else "?"
        typer.echo(f"{rule.id:<5} {rule.rule.value:<5} {rule.heterogeneity_id:<10} {related:<20} {description}")


def _print_assignments(atom_ids: list[str] | None) -> None:
    if atom_ids is None:
        typer.echo("No atom_site assignments found.")
        return
    counts = Counter(atom_ids)
    summary = ", ".join(f"{state_id}: {count}" for state_id, count in sorted(counts.items()))
    typer.echo(f"Atom assignments ({len(atom_ids)} atoms): {summary}")
