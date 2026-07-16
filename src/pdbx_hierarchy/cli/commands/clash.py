"""The ``clash`` sub-app: detect steric clashes and apply proposed mitigations."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from pdbx_hierarchy.clash import (
    CATEGORY_ACTIONABLE,
    CATEGORY_ANCESTOR,
    CATEGORY_BENIGN,
    CATEGORY_SAME_STATE,
    ClashReport,
    MergeProposal,
    NotProposal,
    classify_clashes,
    detect_clashes,
)
from pdbx_hierarchy.cli.commands._utils import (
    error_handler,
    load_document,
    merge_states,
    remap_coexistence,
    resolve_output,
    warn,
)
from pdbx_hierarchy.exceptions import ClashAnalysisError
from pdbx_hierarchy.io.reader import read_atom_site_heterogeneity_ids, read_coexistence, read_hierarchy
from pdbx_hierarchy.io.validation import validate_file
from pdbx_hierarchy.io.writer import (
    write_atom_site_heterogeneity_ids,
    write_coexistence,
    write_hierarchy,
    write_mmcif,
)
from pdbx_hierarchy.models.coexistence import CoexistenceRule, CoexistenceTable, StateCoexistence
from pdbx_hierarchy.models.hierarchy import HierarchyTree

app = typer.Typer(name="clash", help="Detect steric clashes and apply merge / coexistence mitigations.")

_FileArg = Annotated[Path, typer.Argument(help="mmCIF file to analyze.")]
_OutputOpt = Annotated[Path | None, typer.Option("--output", "-o", help="Output path (default: <name>_pdbx_N).")]
_YesOpt = Annotated[bool, typer.Option("--yes", "-y", help="Skip the overwrite prompt.")]

# Number of most-severe warning clashes to list in the human summary.
_TOP_WARNINGS = 5


def _print_summary(report: ClashReport) -> None:
    """Print a human-readable summary of a clash report to stdout."""
    actionable = report.clashes_in(CATEGORY_ACTIONABLE)
    same_state = report.clashes_in(CATEGORY_SAME_STATE)
    ancestor = report.clashes_in(CATEGORY_ANCESTOR)
    benign = report.clashes_in(CATEGORY_BENIGN)

    typer.echo(f"Detected {len(report.clashes)} clash(es):")
    typer.echo(f"  actionable (cross-state):   {len(actionable)}")
    typer.echo(f"  same-state (warning):       {len(same_state)}")
    typer.echo(f"  Base/ancestor (warning):    {len(ancestor)}")
    typer.echo(f"  benign (alternatives/NOT):  {len(benign)}")

    if report.merges:
        typer.echo("\nProposed merges:")
        for m in report.merges:
            typer.echo(f"  merge {m.states} (target {m.states[0]!r})")
    if report.not_rules:
        typer.echo("\nProposed NOT rules:")
        for n in report.not_rules:
            typer.echo(f"  NOT {n.states[0]} <-> {n.states[1]}")
    if report.conflicts:
        typer.echo("\nConflicts (resolve manually):")
        for msg in report.conflicts:
            typer.echo(f"  {msg}")

    warnings = sorted(same_state + ancestor, key=lambda c: c.overlap, reverse=True)
    if warnings:
        typer.echo(f"\nMost severe warnings (top {min(_TOP_WARNINGS, len(warnings))} of {len(warnings)}):")
        for c in warnings[:_TOP_WARNINGS]:
            a1, a2 = c.atom1, c.atom2
            typer.echo(
                f"  {a1.het_id}/{a1.chain}{a1.seq}/{a1.atom_name} <-> "
                f"{a2.het_id}/{a2.chain}{a2.seq}/{a2.atom_name}  overlap={c.overlap} Å"
            )


@app.command("detect")
def detect(
    file: _FileArg,
    tolerance: Annotated[float, typer.Option("--tolerance", help="Minimum vdW overlap (Å) to count as a clash.")] = 0.4,
    report: Annotated[Path | None, typer.Option("--report", help="Write the JSON clash report to this path.")] = None,
    not_only: Annotated[bool, typer.Option("--not-only", help="Propose NOT rules only (no merges).")] = False,
    symmetry: Annotated[bool, typer.Option("--symmetry", help="Include crystal symmetry-mate clashes.")] = False,
    auth: Annotated[bool, typer.Option("--auth", help="Use auth_* residue numbering instead of label_*.")] = False,
) -> None:
    """Detect clashes and propose mitigations; optionally write a JSON report."""
    with error_handler():
        _, block = load_document(file)
        tree = read_hierarchy(block)
        validate_file(block, raise_on_error=True)
        coexistence = read_coexistence(block)

        clashes = detect_clashes(block, tree, tolerance=tolerance, symmetry=symmetry, use_auth=auth)
        clash_report = classify_clashes(clashes, block, tree, coexistence, not_only=not_only, use_auth=auth)

        _print_summary(clash_report)

        if report is not None:
            report.write_text(clash_report.to_json())
            typer.echo(f"\nWrote report to {report}")
        else:
            typer.echo("\n(no --report given; nothing written)")


def _validate_report_refs(report: ClashReport, tree: HierarchyTree) -> None:
    """Raise if any enabled proposal references a state absent from the hierarchy."""
    proposals: list[MergeProposal | NotProposal] = [*report.merges, *report.not_rules]
    for proposal in proposals:
        if not proposal.enabled:
            continue
        for state_id in proposal.states:
            if not tree.contains(state_id):
                raise ClashAnalysisError(
                    f"report references state {state_id!r}, which is not in the hierarchy; "
                    "the file may have changed since the report was generated"
                )


@app.command("apply")
def apply(
    file: _FileArg,
    report_path: Annotated[Path, typer.Option("--report", help="JSON clash report produced by 'clash detect'.")],
    output: _OutputOpt = None,
    yes: _YesOpt = False,
) -> None:
    """Apply the enabled merge / NOT proposals from a clash report to a file."""
    with error_handler():
        clash_report = ClashReport.from_json(report_path.read_text())
        doc, block = load_document(file)
        tree = read_hierarchy(block)
        validate_file(block, raise_on_error=True)
        assignments = read_atom_site_heterogeneity_ids(block)
        coexistence = read_coexistence(block) or CoexistenceTable()

        _validate_report_refs(clash_report, tree)

        # Merges first, so NOT proposals can be remapped onto the surviving ids.
        merge_map: dict[str, str] = {}
        applied_merges = 0
        for merge_prop in clash_report.merges:
            if not merge_prop.enabled:
                continue
            updated, mm = merge_states(tree, assignments, merge_prop.states[0], merge_prop.states[1:])
            assert updated is not None  # assignments was read from the file, never None
            assignments = updated
            merge_map.update(mm)
            applied_merges += 1

        if merge_map:
            coexistence, notes = remap_coexistence(coexistence, merge_map)
            for note in notes:
                warn(note)

        applied_nots = 0
        for not_prop in clash_report.not_rules:
            if not not_prop.enabled:
                continue
            source = merge_map.get(not_prop.states[0], not_prop.states[0])
            related = merge_map.get(not_prop.states[1], not_prop.states[1])
            if source == related:
                warn(f"NOT rule {not_prop.states} collapsed to a self-reference after merging; skipped")
                continue
            next_id = max((r.id for r in coexistence.rules), default=0) + 1
            coexistence.add_rule(
                StateCoexistence(
                    id=next_id,
                    rule=CoexistenceRule.NOT,
                    heterogeneity_id=source,
                    heterogeneity_ids=[related],
                    description="added by clash apply",
                )
            )
            applied_nots += 1

        coexistence.validate_against_hierarchy(tree, raise_on_error=True)

        out = resolve_output(file, output, yes=yes)
        write_hierarchy(block, tree, overwrite=True)
        write_atom_site_heterogeneity_ids(block, assignments, overwrite=True)
        if len(coexistence):
            write_coexistence(block, coexistence, overwrite=True)
        write_mmcif(doc, out)
        typer.echo(f"Applied {applied_merges} merge(s) and {applied_nots} NOT rule(s); wrote {out}")
