"""The ``merge-states`` command: merge a ground and a changed model into one file.

Registered at the top level rather than inside the ``hierarchy`` sub-app: it
operates on two files and produces a third, which is not what that sub-app's
within-one-file commands do.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, NamedTuple

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


class _OutputPlan(NamedTuple):
    """Where a run's files go.

    Attributes:
        merged: The merged file's path.
        intermediate_dir: Where the intermediates go, or None when none were
            asked for. Held as the directory rather than the two paths because
            that is what the merge takes, and naming them is
            :func:`merge.intermediate_path`'s job in both places.
    """

    merged: Path
    intermediate_dir: Path | None


def _plan_outputs(
    ground: Path,
    changed: Path,
    output: Path | None,
    *,
    keep_intermediates: bool,
    yes: bool,
) -> _OutputPlan:
    """Decide where every file this run writes goes, and confirm the lot at once.

    The merged file's default name puts the changed state first, because that is
    the state a reader cares most about; ground comes first everywhere *inside*
    the file, where it is the interpretive baseline. The intermediates are named
    after their own input and always land in the working directory, so a run
    started here leaves its side files here even when ``-o`` sends the merged file
    elsewhere.

    Args:
        ground: The ground input path.
        changed: The changed input path.
        output: The requested merged-file path, or None for the default name in
            the working directory.
        keep_intermediates: Whether the run will also write the two intermediates.
        yes: If True, skip the overwrite confirmation.

    Returns:
        The plan for this run's output files.

    Raises:
        typer.BadParameter: If two of the run's outputs resolve to one path,
            which would silently overwrite one model with another.
        typer.Abort: If a file would be overwritten and the user declines.
    """
    working = Path.cwd()
    merged = output if output is not None else working / f"{changed.stem}_{ground.stem}_hierarchy.cif"
    intermediate_dir = working if keep_intermediates else None

    targets = [merged]
    if intermediate_dir is not None:
        targets.extend(merge.intermediate_path(source, intermediate_dir) for source in (ground, changed))

    resolved = [target.resolve() for target in targets]
    for target, resolved_target in zip(targets, resolved, strict=True):
        if resolved.count(resolved_target) > 1:
            raise typer.BadParameter(f"more than one output file would be written to {target}")

    _confirm_overwrite(targets, yes=yes)
    return _OutputPlan(merged=merged, intermediate_dir=intermediate_dir)


def _confirm_overwrite(targets: list[Path], *, yes: bool) -> None:
    """Ask once about every output file that already exists.

    One confirmation rather than one per file: a run writing three files should
    not stop three times, and answering for some of them but not others would
    leave the outputs inconsistent with each other anyway.

    Args:
        targets: Every file the run will write.
        yes: If True, skip the confirmation entirely.

    Raises:
        typer.Abort: If the user declines.
    """
    if yes:
        return
    existing = [target for target in targets if target.exists()]
    if not existing:
        return
    if len(existing) == 1:
        typer.confirm(f"{existing[0]} already exists. Overwrite?", abort=True)
        return
    listing = "\n".join(f"  {target}" for target in existing)
    typer.confirm(f"{len(existing)} output files already exist:\n{listing}\nOverwrite all?", abort=True)


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
        plan = _plan_outputs(ground, changed, output, keep_intermediates=keep_intermediates, yes=yes)
        report = merge.merge_states(
            ground_path=ground,
            changed_path=changed,
            occ=occ,
            output_path=plan.merged,
            intermediate_dir=plan.intermediate_dir,
        )

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
        typer.echo(f"Wrote {plan.merged}")
        for path in report.intermediate_paths:
            typer.echo(f"Wrote {path} (standalone, unscaled occupancies)")
