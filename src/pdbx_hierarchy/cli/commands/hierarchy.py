"""The ``hierarchy`` sub-app: add, remove, rename, reparent, merge, split, and reassign hierarchy states."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from pdbx_hierarchy.cli.commands._utils import (
    error_handler,
    load_document,
    merge_states,
    parse_selection,
    read_atom_residue_keys,
    reassign_ids,
    remap_coexistence,
    resolve_output,
    warn,
)
from pdbx_hierarchy.exceptions import HierarchyNotFoundError, PdbxValidationError
from pdbx_hierarchy.io.reader import read_atom_site_heterogeneity_ids, read_coexistence, read_hierarchy
from pdbx_hierarchy.io.validation import validate_file
from pdbx_hierarchy.io.writer import (
    write_atom_site_heterogeneity_ids,
    write_coexistence,
    write_hierarchy,
    write_mmcif,
)
from pdbx_hierarchy.models.hierarchy import HierarchyState
from pdbx_hierarchy.models.id_generator import HierarchyIdGenerator

app = typer.Typer(name="hierarchy", help="Add, remove, rename, reparent, merge, split, or reassign hierarchy states.")

_FileArg = Annotated[Path, typer.Argument(help="mmCIF file to modify.")]
_OutputOpt = Annotated[Path | None, typer.Option("--output", "-o", help="Output path (default: <name>_pdbx_N).")]
_YesOpt = Annotated[bool, typer.Option("--yes", "-y", help="Skip the overwrite prompt.")]


def _read_assignments_optional(block: object) -> list[str] | None:
    """Return the atom-assignment column, or None if the file has none."""
    try:
        return read_atom_site_heterogeneity_ids(block)  # type: ignore[arg-type]
    except HierarchyNotFoundError:
        return None


def _remap_coexistence_in_block(block: object, id_map: dict[str, str]) -> None:
    """Rewrite coexistence references through ``id_map`` and write them back.

    No-op when the file has no coexistence table. Warns for any self-reference
    or rule dropped as a result of the remap.
    """
    table = read_coexistence(block)  # type: ignore[arg-type]
    if table is None:
        return
    new_table, notes = remap_coexistence(table, id_map)
    for note in notes:
        warn(note)
    write_coexistence(block, new_table, overwrite=True)  # type: ignore[arg-type]


@app.command("add")
def add(
    file: _FileArg,
    hier_id: Annotated[str, typer.Option("--id", help="Id for the new state.")],
    name: Annotated[str, typer.Option("--name", help="Name for the new state.")],
    parent: Annotated[str, typer.Option("--parent", help="Parent state id.")],
    details: Annotated[str | None, typer.Option("--details", help="Optional description.")] = None,
    output: _OutputOpt = None,
    yes: _YesOpt = False,
) -> None:
    """Add a new state to the hierarchy."""
    with error_handler():
        doc, block = load_document(file)
        tree = read_hierarchy(block)
        validate_file(block, raise_on_error=True)
        tree.add_state(HierarchyState(id=hier_id, name=name, parent=parent, details=details))
        out = resolve_output(file, output, yes=yes)
        write_hierarchy(block, tree, overwrite=True)
        write_mmcif(doc, out)
        typer.echo(f"Added state {hier_id!r}; wrote {out}")


@app.command("remove")
def remove(
    file: _FileArg,
    hier_id: Annotated[str, typer.Option("--id", help="Id of the state to remove.")],
    output: _OutputOpt = None,
    yes: _YesOpt = False,
) -> None:
    """Remove a state, folding its atoms and children into its parent."""
    with error_handler():
        doc, block = load_document(file)
        tree = read_hierarchy(block)
        validate_file(block, raise_on_error=True)

        parent = tree.get_state(hier_id).parent
        if parent is None:
            raise PdbxValidationError(f"cannot remove root state {hier_id!r}")

        # Re-parent children to the grandparent, then remove (now childless) state.
        for child in tree.get_children(hier_id):
            tree.update_state(child.id, parent=parent)
        tree.remove_state(hier_id)

        # Fold the removed state's atoms up into its parent so nothing dangles.
        assignments = _read_assignments_optional(block)
        if assignments is not None:
            moved = sum(1 for atom_id in assignments if atom_id == hier_id)
            assignments = [parent if atom_id == hier_id else atom_id for atom_id in assignments]
            if moved:
                typer.echo(f"Reassigned {moved} atom(s) from {hier_id!r} to parent {parent!r}")

        out = resolve_output(file, output, yes=yes)
        write_hierarchy(block, tree, overwrite=True)
        if assignments is not None:
            write_atom_site_heterogeneity_ids(block, assignments, overwrite=True)
        _remap_coexistence_in_block(block, {hier_id: parent})
        write_mmcif(doc, out)
        typer.echo(f"Removed state {hier_id!r}; wrote {out}")


@app.command("rename")
def rename(
    file: _FileArg,
    hier_id: Annotated[str, typer.Option("--id", help="Id of the state to rename.")],
    name: Annotated[str, typer.Option("--name", help="New name for the state.")],
    output: _OutputOpt = None,
    yes: _YesOpt = False,
) -> None:
    """Change a state's name."""
    with error_handler():
        doc, block = load_document(file)
        tree = read_hierarchy(block)
        validate_file(block, raise_on_error=True)
        tree.update_state(hier_id, name=name)
        out = resolve_output(file, output, yes=yes)
        write_hierarchy(block, tree, overwrite=True)
        write_mmcif(doc, out)
        typer.echo(f"Renamed state {hier_id!r} to {name!r}; wrote {out}")


@app.command("reparent")
def reparent(
    file: _FileArg,
    hier_id: Annotated[str, typer.Option("--id", help="Id of the state to move.")],
    parent: Annotated[str, typer.Option("--parent", help="Id of the new parent state.")],
    output: _OutputOpt = None,
    yes: _YesOpt = False,
) -> None:
    """Move a state (with its whole subtree) under a new parent.

    Ids and atom assignments are unchanged, so coexistence rules and assignments
    are left as-is. Moving a state beneath one of its own descendants is rejected
    as a cycle.
    """
    with error_handler():
        doc, block = load_document(file)
        tree = read_hierarchy(block)
        validate_file(block, raise_on_error=True)

        if tree.get_state(hier_id).parent is None:
            raise PdbxValidationError(f"cannot reparent the root state {hier_id!r}")
        if not tree.contains(parent):
            raise PdbxValidationError(f"unknown parent state {parent!r}")

        # update_state re-validates the tree, so a move that would create a cycle
        # (new parent is the state itself or one of its descendants) raises here.
        tree.update_state(hier_id, parent=parent)

        out = resolve_output(file, output, yes=yes)
        write_hierarchy(block, tree, overwrite=True)
        write_mmcif(doc, out)
        typer.echo(f"Reparented state {hier_id!r} under {parent!r}; wrote {out}")


@app.command("merge")
def merge(
    file: _FileArg,
    ids: Annotated[str, typer.Option("--ids", help="Comma-separated state ids; the first absorbs the rest.")],
    reassign: Annotated[bool, typer.Option("--reassign-ids", help="Canonicalize all ids after merging.")] = False,
    output: _OutputOpt = None,
    yes: _YesOpt = False,
) -> None:
    """Merge states: the first id absorbs the others' atoms and children."""
    with error_handler():
        doc, block = load_document(file)
        tree = read_hierarchy(block)
        validate_file(block, raise_on_error=True)
        assignments = _read_assignments_optional(block)

        # De-duplicate (order-preserving) so the target can't absorb itself: a
        # repeated id would otherwise remove the target state while atoms still
        # reference it, silently producing an invalid file.
        id_list = list(dict.fromkeys(part.strip() for part in ids.split(",") if part.strip()))
        if len(id_list) < 2:
            raise PdbxValidationError("merge requires at least two distinct ids, e.g. --ids A,B")
        target = id_list[0]
        assignments, merge_map = merge_states(tree, assignments, target, id_list[1:])

        if reassign:
            mapping = reassign_ids(tree)
            if assignments is not None:
                assignments = [mapping.get(atom_id, atom_id) for atom_id in assignments]

        out = resolve_output(file, output, yes=yes)
        write_hierarchy(block, tree, overwrite=True)
        if assignments is not None:
            write_atom_site_heterogeneity_ids(block, assignments, overwrite=True)
        _remap_coexistence_in_block(block, merge_map)
        if reassign:
            _remap_coexistence_in_block(block, mapping)
        write_mmcif(doc, out)
        typer.echo(f"Merged {id_list[1:]} into {target!r}; wrote {out}")


@app.command("split")
def split(
    file: _FileArg,
    hier_id: Annotated[str, typer.Option("--id", help="Id of the state to split.")],
    select_a: Annotated[str, typer.Option("--select-a", help="Residue selection for the first child.")],
    select_b: Annotated[str, typer.Option("--select-b", help="Residue selection for the second child.")],
    name_a: Annotated[str | None, typer.Option("--name-a", help="Name for the first child (default: its id).")] = None,
    name_b: Annotated[str | None, typer.Option("--name-b", help="Name for the second child (default: its id).")] = None,
    auth: Annotated[bool, typer.Option("--auth", help="Select on auth_* numbering instead of label_*.")] = False,
    output: _OutputOpt = None,
    yes: _YesOpt = False,
) -> None:
    """Split a state's atoms into two child states by residue selection.

    Residues matched by neither ``--select-a`` nor ``--select-b`` are left on the
    state being split (which then keeps direct atoms alongside its two new
    children); a warning lists them.
    """
    with error_handler():
        doc, block = load_document(file)
        tree = read_hierarchy(block)
        validate_file(block, raise_on_error=True)
        assignments = read_atom_site_heterogeneity_ids(block)
        residue_keys = read_atom_residue_keys(block, use_auth=auth)
        if len(residue_keys) != len(assignments):
            raise PdbxValidationError("atom_site residue columns and assignment column have different lengths")

        rows_in_state = [i for i, atom_id in enumerate(assignments) if atom_id == hier_id]
        if not rows_in_state:
            raise PdbxValidationError(f"state {hier_id!r} has no atoms to split")

        chains = {residue_keys[i][0] for i in rows_in_state}
        sel_a = parse_selection(select_a, chains)
        sel_b = parse_selection(select_b, chains)

        non_numeric = sorted({residue_keys[i] for i in rows_in_state if _residue_key_int(residue_keys[i]) is None})
        if non_numeric:
            warn(
                f"{len(non_numeric)} residue(s) in {hier_id!r} have non-numeric ids and cannot be selected; "
                f"they keep their assignment to {hier_id!r}: {non_numeric}"
            )

        present = _present_residues(residue_keys, rows_in_state)
        _warn_missing(sel_a - present, "--select-a")
        _warn_missing(sel_b - present, "--select-b")
        if not (sel_a & present) and not (sel_b & present):
            raise PdbxValidationError(f"neither --select-a nor --select-b matched any atom in state {hier_id!r}")
        overlap = sel_a & sel_b & present
        if overlap:
            warn(f"residues in both selections are assigned to --select-a: {sorted(overlap)}")
        unselected = present - sel_a - sel_b
        if unselected:
            warn(f"residues matched by neither selection keep their assignment to {hier_id!r}: {sorted(unselected)}")

        generator = HierarchyIdGenerator(tree)
        child_a, child_b = next(generator), next(generator)
        tree.add_state(HierarchyState(id=child_a, name=name_a or child_a, parent=hier_id, details=None))
        tree.add_state(HierarchyState(id=child_b, name=name_b or child_b, parent=hier_id, details=None))

        for i in rows_in_state:
            key = _residue_key_int(residue_keys[i])
            if key is None:
                continue
            if key in sel_a:
                assignments[i] = child_a
            elif key in sel_b:
                assignments[i] = child_b

        out = resolve_output(file, output, yes=yes)
        write_hierarchy(block, tree, overwrite=True)
        write_atom_site_heterogeneity_ids(block, assignments, overwrite=True)
        write_mmcif(doc, out)
        typer.echo(f"Split {hier_id!r} into {child_a!r} and {child_b!r}; wrote {out}")


@app.command("reassign")
def reassign(
    file: _FileArg,
    preserve_named: Annotated[
        bool, typer.Option("--preserve-named", help="Keep non-canonical (hand-given) state ids.")
    ] = False,
    output: _OutputOpt = None,
    yes: _YesOpt = False,
) -> None:
    """Canonicalize state ids (Base stays Base; others become A, B, C, ...)."""
    with error_handler():
        doc, block = load_document(file)
        tree = read_hierarchy(block)
        validate_file(block, raise_on_error=True)
        assignments = _read_assignments_optional(block)

        mapping = reassign_ids(tree, preserve_named=preserve_named)
        out = resolve_output(file, output, yes=yes)
        write_hierarchy(block, tree, overwrite=True)
        if assignments is not None:
            write_atom_site_heterogeneity_ids(
                block, [mapping.get(atom_id, atom_id) for atom_id in assignments], overwrite=True
            )
        _remap_coexistence_in_block(block, mapping)
        write_mmcif(doc, out)
        typer.echo(f"Reassigned ids; wrote {out}")


def _present_residues(residue_keys: list[tuple[str, str]], rows: list[int]) -> set[tuple[str, int]]:
    """Return the ``(chain, seq)`` pairs (numeric seq only) present in the given rows."""
    present: set[tuple[str, int]] = set()
    for i in rows:
        key = _residue_key_int(residue_keys[i])
        if key is not None:
            present.add(key)
    return present


def _residue_key_int(key: tuple[str, str]) -> tuple[str, int] | None:
    """Convert a ``(chain, seq_str)`` key to ``(chain, int)``, or None if seq isn't numeric."""
    chain, seq_str = key
    try:
        return chain, int(seq_str)
    except ValueError:
        return None


def _warn_missing(missing: set[tuple[str, int]], flag: str) -> None:
    if missing:
        warn(f"{flag}: residues not present in the state were ignored: {sorted(missing)}")
