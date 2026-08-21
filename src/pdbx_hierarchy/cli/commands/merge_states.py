"""The ``merge-states`` command: merge a ground and a changed model into one file.

Registered at the top level rather than inside the ``hierarchy`` sub-app: it
operates on two files and produces a third, which is not what that sub-app's
within-one-file commands do.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from pdbx_hierarchy import merge
from pdbx_hierarchy.cli.commands._utils import error_handler, render_tree, warn


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
        if report.changed_id_map:
            remap = ", ".join(f"{old}->{new}" for old, new in sorted(report.changed_id_map.items()))
            typer.echo(f"Changed state ids: {remap}")
        typer.echo(f"Wrote {out}")
