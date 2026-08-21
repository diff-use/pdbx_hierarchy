"""Tests for src/pdbx_hierarchy/io/reader.py."""

from pathlib import Path

import gemmi
import pytest

from pdbx_hierarchy.exceptions import HierarchyNotFoundError, PdbxParseError
from pdbx_hierarchy.io.reader import (
    _resolve_block,
    count_coexistence_rules,
    has_hierarchy,
    read_atom_site_heterogeneity_ids,
    read_coexistence,
    read_hierarchy,
    read_mmcif,
)
from pdbx_hierarchy.models.coexistence import CoexistenceRule
from pdbx_hierarchy.models.hierarchy import HierarchyTree


class TestResolveBlock:
    def test_with_path(self, hierarchy_only_cif: Path) -> None:
        block = _resolve_block(hierarchy_only_cif)
        assert isinstance(block, gemmi.cif.Block)
        assert block.name == "HIERARCHY_ONLY"

    def test_with_block(self, minimal_block: gemmi.cif.Block) -> None:
        result = _resolve_block(minimal_block)
        assert result is minimal_block

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _resolve_block(tmp_path / "nonexistent.cif")

    def test_bad_syntax(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.cif"
        bad.write_text("this is !! not valid cif content\n")
        with pytest.raises(PdbxParseError):
            _resolve_block(bad)

    def test_multiple_blocks(self, tmp_path: Path) -> None:
        multi = tmp_path / "multi.cif"
        multi.write_text("data_A\n_entry.id A\n\ndata_B\n_entry.id B\n")
        with pytest.raises(PdbxParseError):
            _resolve_block(multi)


class TestReadMmcif:
    def test_returns_block(self, hierarchy_only_cif: Path) -> None:
        block = read_mmcif(hierarchy_only_cif)
        assert isinstance(block, gemmi.cif.Block)

    def test_block_name(self, full_hierarchy_cif: Path) -> None:
        block = read_mmcif(full_hierarchy_cif)
        assert block.name == "FULL_HIERARCHY"

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_mmcif(tmp_path / "missing.cif")


class TestHasHierarchy:
    def test_true_hierarchy_only(self, hierarchy_only_cif: Path) -> None:
        assert has_hierarchy(hierarchy_only_cif) is True

    def test_true_full(self, full_hierarchy_cif: Path) -> None:
        assert has_hierarchy(full_hierarchy_cif) is True

    def test_false(self, atom_site_no_hierarchy_cif: Path) -> None:
        assert has_hierarchy(atom_site_no_hierarchy_cif) is False

    def test_accepts_block(self, minimal_block: gemmi.cif.Block) -> None:
        assert has_hierarchy(minimal_block) is False


class TestReadHierarchy:
    def test_from_path(self, hierarchy_only_cif: Path) -> None:
        tree = read_hierarchy(hierarchy_only_cif)
        assert isinstance(tree, HierarchyTree)
        assert len(tree) == 3

    def test_root(self, hierarchy_only_cif: Path) -> None:
        tree = read_hierarchy(hierarchy_only_cif)
        root = tree.get_root()
        assert root.id == "Base"
        assert root.parent is None

    def test_children(self, hierarchy_only_cif: Path) -> None:
        tree = read_hierarchy(hierarchy_only_cif)
        children = tree.get_children("Base")
        child_ids = {c.id for c in children}
        assert child_ids == {"A", "B"}

    def test_from_block(self, hierarchy_only_cif: Path) -> None:
        block = read_mmcif(hierarchy_only_cif)
        tree = read_hierarchy(block)
        assert len(tree) == 3

    def test_not_found(self, atom_site_no_hierarchy_cif: Path) -> None:
        with pytest.raises(HierarchyNotFoundError):
            read_hierarchy(atom_site_no_hierarchy_cif)

    def test_full_fixture(self, full_hierarchy_cif: Path) -> None:
        tree = read_hierarchy(full_hierarchy_cif)
        assert len(tree) == 2

    def test_details_none(self, hierarchy_only_cif: Path) -> None:
        tree = read_hierarchy(hierarchy_only_cif)
        # All states in hierarchy_only.cif have details="?"
        for state in tree.states:
            assert state.details is None

    def test_parent_dot_to_none(self, hierarchy_only_cif: Path) -> None:
        tree = read_hierarchy(hierarchy_only_cif)
        root = tree.get_root()
        assert root.parent is None


class TestReadCoexistence:
    def test_from_path(self, full_hierarchy_cif: Path) -> None:
        table = read_coexistence(full_hierarchy_cif)
        assert table is not None
        assert len(table) == 1

    def test_from_block(self, full_hierarchy_cif: Path) -> None:
        block = read_mmcif(full_hierarchy_cif)
        table = read_coexistence(block)
        assert table is not None

    def test_none_when_absent(self, hierarchy_only_cif: Path) -> None:
        assert read_coexistence(hierarchy_only_cif) is None

    def test_rule_value(self, full_hierarchy_cif: Path) -> None:
        table = read_coexistence(full_hierarchy_cif)
        assert table is not None
        assert table.rules[0].rule is CoexistenceRule.NOT

    def test_description_none(self, full_hierarchy_cif: Path) -> None:
        table = read_coexistence(full_hierarchy_cif)
        assert table is not None
        assert table.rules[0].description is None

    def test_heterogeneity_ids_list(self, full_hierarchy_cif: Path) -> None:
        table = read_coexistence(full_hierarchy_cif)
        assert table is not None
        assert table.rules[0].heterogeneity_ids == ["A"]


class TestCountCoexistenceRules:
    def test_a_loop_is_counted(self, full_hierarchy_cif: Path) -> None:
        assert count_coexistence_rules(full_hierarchy_cif) == 1

    def test_zero_when_absent(self, hierarchy_only_cif: Path) -> None:
        assert count_coexistence_rules(hierarchy_only_cif) == 0

    def test_a_single_rule_written_as_pairs_is_counted(self) -> None:
        # One row is conventionally written as _tag value pairs rather than a
        # one-row loop_, and a count that missed those would report a file's only
        # rule as no rules at all.
        block = gemmi.cif.read_string(
            "data_X\n"
            "_pdbx_state_coexistence.id 1\n"
            "_pdbx_state_coexistence.rule NOT\n"
            "_pdbx_state_coexistence.heterogeneity_id A\n"
            "_pdbx_state_coexistence.heterogeneity_ids B\n"
            "_pdbx_state_coexistence.description ?\n"
        ).sole_block()
        assert count_coexistence_rules(block) == 1


class TestReadAtomSiteHeterogeneityIds:
    def test_from_path(self, full_hierarchy_cif: Path) -> None:
        ids = read_atom_site_heterogeneity_ids(full_hierarchy_cif)
        assert len(ids) == 2

    def test_values(self, full_hierarchy_cif: Path) -> None:
        ids = read_atom_site_heterogeneity_ids(full_hierarchy_cif)
        assert ids == ["Base", "A"]

    def test_from_block(self, full_hierarchy_cif: Path) -> None:
        block = read_mmcif(full_hierarchy_cif)
        ids = read_atom_site_heterogeneity_ids(block)
        assert ids == ["Base", "A"]

    def test_not_found_no_column(self, atom_site_no_hierarchy_cif: Path) -> None:
        with pytest.raises(HierarchyNotFoundError):
            read_atom_site_heterogeneity_ids(atom_site_no_hierarchy_cif)

    def test_not_found_no_atom_site(self, hierarchy_only_cif: Path) -> None:
        with pytest.raises(HierarchyNotFoundError):
            read_atom_site_heterogeneity_ids(hierarchy_only_cif)
