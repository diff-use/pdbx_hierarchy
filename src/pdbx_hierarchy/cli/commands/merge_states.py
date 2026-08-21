"""The ``merge-states`` command: merge a ground and a changed model into one file.

Registered at the top level rather than inside the ``hierarchy`` sub-app: it
operates on two files and produces a third, which is not what that sub-app's
within-one-file commands do.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from pdbx_hierarchy import merge, numbering
from pdbx_hierarchy.cli.commands._utils import error_handler, render_tree, warn

if TYPE_CHECKING:
    from collections.abc import Mapping


def _validate_occ(value: float) -> float:
    """Reject an occupancy the output format cannot express.

    Args:
        value: The ``--occ`` value as parsed by typer.

    Returns:
        The value, unchanged.

    Raises:
        typer.BadParameter: If the value is outside ``0 < occ < 1`` or carries
            more than two decimal places. Both are usage errors rather than
            domain exceptions: the fault is in the command line, not the data.
    """
    if not 0 < value < 1:
        raise typer.BadParameter(f"must satisfy 0 < occ < 1, got {value}")
    if round(value, 2) != value:
        raise typer.BadParameter(f"must carry at most two decimal places, got {value}")
    return value


def _render_map(mapping: Mapping[str, str]) -> str:
    """Render an old -> new remap as one traceable line.

    Sorted by the old value so a reader looking up what became of a particular
    state id or altloc label can scan for it, rather than having to know which
    order the merge happened to assign in.

    Args:
        mapping: Old -> new, of state ids or of altloc labels.

    Returns:
        The remap as ``old->new`` pairs, comma-separated.
    """
    return ", ".join(f"{old}->{new}" for old, new in sorted(mapping.items()))


def _render_shift(shift: numbering.NonPolymerShift) -> str:
    """Render the non-polymer renumbering as one traceable line.

    The offset is printed even when it is zero, and the ranges alongside it: the
    merged file has nowhere to record the number a non-polymer used to carry, so
    this line is the only route from a merged water or ligand back to its
    deposited identity, and "nothing moved" is as much a part of that as a shift
    is.

    Args:
        shift: The shift the merge applied.

    Returns:
        The line to print.
    """
    if shift.residue_count == 0:
        return "Changed non-polymer residues: none to renumber"
    return (
        f"Changed non-polymer auth_seq_id: {shift.old_min}-{shift.old_max} -> "
        f"{shift.new_min}-{shift.new_max} (offset +{shift.offset}, {shift.residue_count} residue(s))"
    )


def _resolve_merged_path(ground: Path, changed: Path, output: Path | None, *, yes: bool) -> Path:
    """Decide where the merged file goes.

    The default name puts the changed state first, because that is the state a
    reader cares most about; ground comes first everywhere *inside* the file,
    where it is the interpretive baseline.

    Args:
        ground: The ground input path.
        changed: The changed input path.
        output: The requested output path, or None for the default name in the
            working directory.
        yes: If True, skip the overwrite confirmation.

    Returns:
        The path to write the merged file to.
    """
    target = output if output is not None else Path.cwd() / f"{changed.stem}_{ground.stem}_hierarchy.cif"
    if target.exists() and not yes:
        typer.confirm(f"{target} already exists. Overwrite?", abort=True)
    return target


def merge_states(
    ground: Annotated[Path, typer.Option("--ground", help="Ground-state mmCIF file.")],
    changed: Annotated[Path, typer.Option("--changed", help="Changed-state mmCIF file.")],
    occ: Annotated[
        float,
        typer.Option(
            "--occ",
            callback=_validate_occ,
            help="The changed state's fraction of the crystal (0 < occ < 1, at most two decimals).",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Merged file path (default: <changed>_<ground>_hierarchy.cif)."),
    ] = None,
    keep_intermediates: Annotated[
        bool, typer.Option("--keep-intermediates", help="Also write each input as a standalone state file.")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the overwrite prompt.")] = False,
) -> None:
    """Merge a ground and a changed model into one hierarchical file.

    The merged file's tree is Base -> {Ground, Changed}, with every atom from
    both inputs assigned to a state under the appropriate branch.
    """
    with error_handler():
        out = _resolve_merged_path(ground, changed, output, yes=yes)
        if keep_intermediates:
            warn("--keep-intermediates is not implemented yet; only the merged file will be written")
        report = merge.merge_states(ground_path=ground, changed_path=changed, occ=occ, output_path=out)

        for note in report.notes:
            warn(note)
        typer.echo(f"Merged {ground.name} (ground) and {changed.name} (changed) at occ={occ:.2f}")
        typer.echo(render_tree(report.tree))
        # Printed in full rather than summarised: these mappings are the only
        # trace from an atom in the merged file back to its row in an input.
        if report.changed_id_map:
            typer.echo(f"Changed state ids: {_render_map(report.changed_id_map)}")
        for altloc_map in report.altloc_maps:
            typer.echo(f"Altloc labels in {altloc_map.source_name}: {_render_map(altloc_map.mapping)}")
        typer.echo(_render_shift(report.nonpolymer_shift))
        typer.echo(f"Wrote {out}")
