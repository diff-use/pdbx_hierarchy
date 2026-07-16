"""Tests for the clash-detection and mitigation module."""

from __future__ import annotations

from pathlib import Path

import pytest

from pdbx_hierarchy.clash import (
    CATEGORY_ACTIONABLE,
    CATEGORY_ANCESTOR,
    CATEGORY_BENIGN,
    CATEGORY_SAME_STATE,
    Clash,
    ClashAtom,
    ClashReport,
    MergeProposal,
    NotProposal,
    classify_clashes,
    detect_clashes,
)
from pdbx_hierarchy.exceptions import ClashAnalysisError
from pdbx_hierarchy.io.reader import read_coexistence, read_hierarchy


@pytest.fixture
def clashing_cif(fixtures_dir: Path) -> Path:
    return fixtures_dir / "clashing_states.cif"


@pytest.fixture
def symmetry_cif(fixtures_dir: Path) -> Path:
    return fixtures_dir / "clashing_symmetry.cif"


def _report_for(path: Path, *, not_only: bool = False) -> tuple[list[Clash], ClashReport]:
    tree = read_hierarchy(path)
    clashes = detect_clashes(path, tree)
    report = classify_clashes(clashes, path, tree, read_coexistence(path), not_only=not_only)
    return clashes, report


def _pair(clash: Clash) -> frozenset[str]:
    return frozenset((clash.atom1.het_id, clash.atom2.het_id))


# --- detection ---------------------------------------------------------------


def test_detects_expected_clashes(clashing_cif: Path) -> None:
    clashes, _ = _report_for(clashing_cif)
    pairs = {_pair(c) for c in clashes}
    assert frozenset({"A", "C"}) in pairs
    assert frozenset({"E", "H"}) in pairs
    # 5 clusters produce clashes; the two bonded clusters (C-N, S-S) do not.
    assert len(clashes) == 5


def test_covalent_bonds_are_excluded(clashing_cif: Path) -> None:
    clashes, _ = _report_for(clashing_cif)
    # The Base C-N (1.33 A) and disulfide S-S (2.05 A) pairs are bonds, not clashes.
    residues_involved = {(c.atom1.seq, c.atom2.seq) for c in clashes}
    assert ("50", "51") not in residues_involved and ("51", "50") not in residues_involved
    assert ("60", "61") not in residues_involved and ("61", "60") not in residues_involved


def test_extreme_overlap_is_not_mistaken_for_a_bond(fixtures_dir: Path) -> None:
    # A@res1 and C@res5 overlap at 2.5 A; that is well outside any covalent window,
    # so it is reported rather than silently swallowed as a bond.
    clashes, _ = _report_for(fixtures_dir / "clashing_states.cif")
    ac = [c for c in clashes if _pair(c) == frozenset({"A", "C"})]
    assert len(ac) == 1
    assert ac[0].overlap > 0.4


# --- classification ----------------------------------------------------------


def test_binary_complement_merge(clashing_cif: Path) -> None:
    _, report = _report_for(clashing_cif)
    merged = sorted(sorted(m.states) for m in report.merges)
    assert merged == [["A", "D"], ["B", "C"]]
    assert not report.not_rules or frozenset({"E", "H"}) in {frozenset(n.states) for n in report.not_rules}


def test_three_alternatives_give_not_rule(clashing_cif: Path) -> None:
    _, report = _report_for(clashing_cif)
    not_pairs = {frozenset(n.states) for n in report.not_rules}
    assert frozenset({"E", "H"}) in not_pairs


def test_alternative_conformation_is_benign(clashing_cif: Path) -> None:
    clashes, _ = _report_for(clashing_cif)
    kl = [c for c in clashes if _pair(c) == frozenset({"K", "L"})]
    assert len(kl) == 1
    assert kl[0].category == CATEGORY_BENIGN


def test_base_clash_is_ancestor_warning(clashing_cif: Path) -> None:
    clashes, _ = _report_for(clashing_cif)
    ancestor = [c for c in clashes if c.category == CATEGORY_ANCESTOR]
    assert len(ancestor) == 1
    assert "Base" in (ancestor[0].atom1.het_id, ancestor[0].atom2.het_id)


def test_same_state_clash_is_warning(clashing_cif: Path) -> None:
    clashes, _ = _report_for(clashing_cif)
    same = [c for c in clashes if c.category == CATEGORY_SAME_STATE]
    assert len(same) == 1
    assert same[0].atom1.het_id == same[0].atom2.het_id == "Base"


def test_actionable_count(clashing_cif: Path) -> None:
    clashes, _ = _report_for(clashing_cif)
    actionable = [c for c in clashes if c.category == CATEGORY_ACTIONABLE]
    assert len(actionable) == 2


def test_not_only_suppresses_merges(clashing_cif: Path) -> None:
    _, report = _report_for(clashing_cif, not_only=True)
    assert report.merges == []
    not_pairs = {frozenset(n.states) for n in report.not_rules}
    assert frozenset({"A", "C"}) in not_pairs
    assert frozenset({"E", "H"}) in not_pairs


def test_no_conflicts_on_clean_fixture(clashing_cif: Path) -> None:
    _, report = _report_for(clashing_cif)
    assert report.conflicts == []


# --- symmetry ----------------------------------------------------------------


def test_symmetry_flag_controls_crystal_contacts(symmetry_cif: Path) -> None:
    tree = read_hierarchy(symmetry_cif)
    assert detect_clashes(symmetry_cif, tree, symmetry=False) == []
    assert len(detect_clashes(symmetry_cif, tree, symmetry=True)) == 1


# --- report serialization ----------------------------------------------------


def test_report_json_round_trip(clashing_cif: Path) -> None:
    _, report = _report_for(clashing_cif)
    restored = ClashReport.from_json(report.to_json())
    assert [m.states for m in restored.merges] == [m.states for m in report.merges]
    assert [n.states for n in restored.not_rules] == [n.states for n in report.not_rules]
    assert len(restored.clashes) == len(report.clashes)


def test_report_preserves_enabled_flag() -> None:
    report = ClashReport(
        merges=[MergeProposal(states=["A", "D"], enabled=False)],
        not_rules=[NotProposal(states=["E", "H"], enabled=True)],
    )
    restored = ClashReport.from_json(report.to_json())
    assert restored.merges[0].enabled is False
    assert restored.not_rules[0].enabled is True


def test_report_rejects_wrong_version() -> None:
    text = '{"version": 999, "clashes": [], "merges": [], "not_rules": [], "conflicts": []}'
    with pytest.raises(ClashAnalysisError, match="version"):
        ClashReport.from_json(text)


def test_report_rejects_malformed_json() -> None:
    with pytest.raises(ClashAnalysisError, match="not valid JSON"):
        ClashReport.from_json("{not json")


def test_clash_atom_round_trip() -> None:
    atom = ClashAtom(serial=7, chain="A", seq="12", atom_name="CB", het_id="C")
    assert ClashAtom.from_dict(atom.to_dict()) == atom
