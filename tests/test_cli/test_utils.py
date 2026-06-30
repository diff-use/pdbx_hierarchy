"""Unit tests for src/pdbx_hierarchy/cli/commands/_utils.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from pdbx_hierarchy.cli.commands._utils import (
    parse_residue_ranges,
    parse_selection,
    reassign_ids,
    remap_coexistence,
    render_tree,
    resolve_output,
)
from pdbx_hierarchy.exceptions import PdbxValidationError
from pdbx_hierarchy.models.coexistence import CoexistenceRule, CoexistenceTable, StateCoexistence
from pdbx_hierarchy.models.hierarchy import HierarchyState, HierarchyTree


class TestParseResidueRanges:
    def test_ranges_and_singletons(self) -> None:
        assert parse_residue_ranges("10-12,14") == {10, 11, 12, 14}

    def test_whitespace_and_empty_tokens(self) -> None:
        assert parse_residue_ranges(" 1 , 3-4 , ") == {1, 3, 4}

    def test_invalid_token_raises(self) -> None:
        with pytest.raises(PdbxValidationError):
            parse_residue_ranges("abc")

    def test_reversed_range_raises(self) -> None:
        with pytest.raises(PdbxValidationError):
            parse_residue_ranges("5-3")


class TestParseSelection:
    def test_single_chain_bare(self) -> None:
        assert parse_selection("10-12,14", {"A"}) == {("A", 10), ("A", 11), ("A", 12), ("A", 14)}

    def test_chain_qualified(self) -> None:
        assert parse_selection("A/1,B/2", {"A", "B"}) == {("A", 1), ("B", 2)}

    def test_multichain_bare_is_ambiguous(self) -> None:
        with pytest.raises(PdbxValidationError, match="Ambiguous"):
            parse_selection("1", {"A", "B"})


def _tree(*states: HierarchyState) -> HierarchyTree:
    return HierarchyTree(states=list(states))


class TestReassignIds:
    def test_canonical_renumbering(self) -> None:
        tree = _tree(
            HierarchyState(id="Base", name="base_state", parent=None),
            HierarchyState(id="X1", name="named", parent="Base"),
            HierarchyState(id="B", name="state_b", parent="Base"),
        )
        mapping = reassign_ids(tree)
        assert mapping["Base"] == "Base"
        assert mapping["X1"] == "A"
        assert mapping["B"] == "B"
        assert {s.id for s in tree.states} == {"Base", "A", "B"}

    def test_preserve_named(self) -> None:
        tree = _tree(
            HierarchyState(id="Base", name="base_state", parent=None),
            HierarchyState(id="X1", name="named", parent="Base"),
            HierarchyState(id="B", name="state_b", parent="Base"),
        )
        mapping = reassign_ids(tree, preserve_named=True)
        assert mapping["X1"] == "X1"  # non-canonical id kept
        assert mapping["B"] == "A"  # canonical id renumbered, skipping X1


class TestRemapCoexistence:
    def test_self_reference_dropped(self) -> None:
        table = CoexistenceTable(
            rules=[
                StateCoexistence(id=1, rule=CoexistenceRule.NOT, heterogeneity_id="A", heterogeneity_ids=["B", "C"]),
            ]
        )
        new_table, notes = remap_coexistence(table, {"B": "A"})
        assert new_table.get_rule(1).heterogeneity_ids == ["C"]
        assert any("self-reference" in note for note in notes)

    def test_empty_rule_dropped(self) -> None:
        table = CoexistenceTable(
            rules=[
                StateCoexistence(id=1, rule=CoexistenceRule.NOT, heterogeneity_id="A", heterogeneity_ids=["B"]),
            ]
        )
        new_table, notes = remap_coexistence(table, {"B": "A"})
        assert len(new_table) == 0
        assert any("dropped" in note for note in notes)


class TestResolveOutput:
    def test_auto_suffix_increments(self, tmp_path: Path) -> None:
        src = tmp_path / "in.cif"
        src.write_text("x")
        first = resolve_output(src, None, yes=False)
        assert first.name == "in_pdbx_1.cif"
        first.write_text("y")
        second = resolve_output(src, None, yes=False)
        assert second.name == "in_pdbx_2.cif"

    def test_explicit_in_place_with_yes(self, tmp_path: Path) -> None:
        src = tmp_path / "in.cif"
        src.write_text("x")
        assert resolve_output(src, src, yes=True) == src


def test_render_tree() -> None:
    tree = _tree(
        HierarchyState(id="Base", name="base_state", parent=None),
        HierarchyState(id="A", name="state_a", parent="Base"),
    )
    rendered = render_tree(tree)
    assert rendered == "Base (base_state)\n  A (state_a)"
