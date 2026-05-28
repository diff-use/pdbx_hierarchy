"""Tests for src/pdbx_hierarchy/assignment.py."""

from __future__ import annotations

from pathlib import Path

import gemmi
import pytest

from pdbx_hierarchy import assign_from_alt_ids, validate_file
from pdbx_hierarchy.io.writer import write_atom_site_heterogeneity_ids, write_hierarchy, write_mmcif
from pdbx_hierarchy.models.hierarchy import HierarchyTree

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sequence_gap_result() -> tuple[HierarchyTree, list[str]]:
    return assign_from_alt_ids(FIXTURES / "assign_sequence_gap.cif")


@pytest.fixture
def no_alt_result() -> tuple[HierarchyTree, list[str]]:
    return assign_from_alt_ids(FIXTURES / "assign_no_alt.cif")


@pytest.fixture
def no_alt_column_result() -> tuple[HierarchyTree, list[str]]:
    return assign_from_alt_ids(FIXTURES / "assign_no_alt_column.cif")


@pytest.fixture
def sidechain_result() -> tuple[HierarchyTree, list[str]]:
    return assign_from_alt_ids(FIXTURES / "assign_single_residue_sidechain.cif")


@pytest.fixture
def backbone_merge_result() -> tuple[HierarchyTree, list[str]]:
    return assign_from_alt_ids(FIXTURES / "assign_backbone_merge.cif")


@pytest.fixture
def h_on_n_result() -> tuple[HierarchyTree, list[str]]:
    return assign_from_alt_ids(FIXTURES / "assign_h_on_n_merge.cif")


@pytest.fixture
def multi_chain_result() -> tuple[HierarchyTree, list[str]]:
    return assign_from_alt_ids(FIXTURES / "assign_multiple_chains.cif")


class TestNoAltConformations:
    def test_all_dot_returns_base_only(self, no_alt_result: tuple[HierarchyTree, list[str]]) -> None:
        tree, assignments = no_alt_result
        assert len(tree) == 1
        assert tree.get_root().id == "Base"
        assert all(a == "Base" for a in assignments)

    def test_missing_alt_column_returns_base_only(self, no_alt_column_result: tuple[HierarchyTree, list[str]]) -> None:
        tree, assignments = no_alt_column_result
        assert len(tree) == 1
        assert all(a == "Base" for a in assignments)


class TestReturnContract:
    def test_returns_tuple_of_tree_and_list(self, sidechain_result: tuple[HierarchyTree, list[str]]) -> None:
        result = sidechain_result
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], HierarchyTree)
        assert isinstance(result[1], list)

    def test_tree_has_base_root(self, sidechain_result: tuple[HierarchyTree, list[str]]) -> None:
        tree, _ = sidechain_result
        assert tree.get_root().id == "Base"

    def test_all_non_root_states_under_base(self, sidechain_result: tuple[HierarchyTree, list[str]]) -> None:
        tree, _ = sidechain_result
        for state in tree.states:
            if not state.is_root:
                assert state.parent == "Base"

    def test_assignment_length_matches_atom_count(self, sidechain_result: tuple[HierarchyTree, list[str]]) -> None:
        _, assignments = sidechain_result
        assert len(assignments) == 4  # assign_single_residue_sidechain.cif has 4 rows

    def test_all_assignments_are_valid_state_ids(self, sidechain_result: tuple[HierarchyTree, list[str]]) -> None:
        tree, assignments = sidechain_result
        valid_ids = {s.id for s in tree.states}
        for a in assignments:
            assert a in valid_ids


class TestSingleResidueSidechain:
    def test_two_alt_states_created(self, sidechain_result: tuple[HierarchyTree, list[str]]) -> None:
        tree, _ = sidechain_result
        assert len(tree) == 3  # Base + A + B

    def test_backbone_atoms_assigned_base(self, sidechain_result: tuple[HierarchyTree, list[str]]) -> None:
        _, assignments = sidechain_result
        assert assignments[0] == "Base"  # N (row 0)
        assert assignments[1] == "Base"  # CA (row 1)

    def test_sidechain_alt_a_atoms_assigned_same_state(self, sidechain_result: tuple[HierarchyTree, list[str]]) -> None:
        _, assignments = sidechain_result
        assert assignments[2] != "Base"  # CB alt A (row 2)

    def test_sidechain_alt_b_atoms_assigned_same_state(self, sidechain_result: tuple[HierarchyTree, list[str]]) -> None:
        _, assignments = sidechain_result
        assert assignments[3] != "Base"  # CB alt B (row 3)

    def test_alt_a_and_alt_b_are_different_states(self, sidechain_result: tuple[HierarchyTree, list[str]]) -> None:
        _, assignments = sidechain_result
        assert assignments[2] != assignments[3]


class TestBackboneMerge:
    def test_both_residues_alt_a_in_same_state(self, backbone_merge_result: tuple[HierarchyTree, list[str]]) -> None:
        _, assignments = backbone_merge_result
        # Res1-N(A) = row 0, Res2-N(A) = row 8
        assert assignments[0] == assignments[8]

    def test_both_residues_alt_b_in_same_state(self, backbone_merge_result: tuple[HierarchyTree, list[str]]) -> None:
        _, assignments = backbone_merge_result
        # Res1-N(B) = row 1, Res2-N(B) = row 9
        assert assignments[1] == assignments[9]

    def test_alt_a_and_alt_b_states_differ(self, backbone_merge_result: tuple[HierarchyTree, list[str]]) -> None:
        _, assignments = backbone_merge_result
        assert assignments[0] != assignments[1]

    def test_exactly_two_alt_states(self, backbone_merge_result: tuple[HierarchyTree, list[str]]) -> None:
        tree, _ = backbone_merge_result
        assert len(tree) == 3  # Base + A + B


class TestHOnNMerge:
    def test_h_of_res2_same_state_as_res1_alt_a(self, h_on_n_result: tuple[HierarchyTree, list[str]]) -> None:
        _, assignments = h_on_n_result
        # Res1-N(A) = row 0, Res2-H(A) = row 9
        assert assignments[9] == assignments[0]

    def test_cb_of_res2_different_state_from_res1(self, h_on_n_result: tuple[HierarchyTree, list[str]]) -> None:
        _, assignments = h_on_n_result
        # Res1-N(A) = row 0, Res2-CB(A) = row 12
        assert assignments[12] != assignments[0]

    def test_n_side_and_ca_side_of_res2_are_different_states(
        self, h_on_n_result: tuple[HierarchyTree, list[str]]
    ) -> None:
        _, assignments = h_on_n_result
        # Res2-H(A) = row 9 (N-side), Res2-CB(A) = row 12 (CA-side)
        assert assignments[9] != assignments[12]

    def test_four_alt_states_total(self, h_on_n_result: tuple[HierarchyTree, list[str]]) -> None:
        tree, _ = h_on_n_result
        assert len(tree) == 5  # Base + A (Res1-A+H-A) + B (Res1-B+H-B) + C (CB-A) + D (CB-B)


class TestMultipleChains:
    def test_same_alt_labels_in_different_chains_are_independent_states(
        self, multi_chain_result: tuple[HierarchyTree, list[str]]
    ) -> None:
        _, assignments = multi_chain_result
        # ChainA-N(A) = row 0, ChainB-N(A) = row 6
        assert assignments[0] != assignments[6]


class TestSequenceGap:
    def test_consecutive_residues_merge(self, sequence_gap_result: tuple[HierarchyTree, list[str]]) -> None:
        _, assignments = sequence_gap_result
        # Res1-N(A) = row 0, Res2-N(A) = row 6 — seq 1→2, consecutive, should merge
        assert assignments[0] == assignments[6]

    def test_gap_residues_do_not_merge(self, sequence_gap_result: tuple[HierarchyTree, list[str]]) -> None:
        _, assignments = sequence_gap_result
        # Res2-N(A) = row 6, Res5-N(A) = row 12 — seq 2→5, gap of 3, should not merge
        assert assignments[6] != assignments[12]

    def test_gap_creates_independent_states(self, sequence_gap_result: tuple[HierarchyTree, list[str]]) -> None:
        tree, _ = sequence_gap_result
        # Res1+2 merged → 2 states (A, B); Res5 isolated → 2 more states (C, D)
        assert len(tree) == 5  # Base + A + B + C + D


class TestEdgeCases:
    def test_accepts_block_input(self) -> None:
        block = gemmi.cif.read(str(FIXTURES / "assign_backbone_merge.cif")).sole_block()
        tree, assignments = assign_from_alt_ids(block)
        assert isinstance(tree, HierarchyTree)
        assert len(assignments) > 0

    def test_nonstd_residue_without_backbone_is_isolated_group(self) -> None:
        doc = gemmi.cif.Document()
        block = doc.add_new_block("LIGAND")
        loop = block.init_loop("_atom_site.", ["id", "label_asym_id", "label_seq_id", "label_atom_id", "label_alt_id"])
        loop.add_row(["1", "A", ".", "C1", "A"])
        loop.add_row(["2", "A", ".", "C1", "B"])
        loop.add_row(["3", "A", ".", "C2", "A"])
        loop.add_row(["4", "A", ".", "C2", "B"])
        tree, assignments = assign_from_alt_ids(block)
        # C1(A) and C2(A) are in the same (residue_key, alt_id) group → same state
        assert assignments[0] == assignments[2]
        # alt A and alt B are different states
        assert assignments[0] != assignments[1]

    def test_output_passes_validate_file(self, tmp_path: Path) -> None:
        fixture = FIXTURES / "assign_backbone_merge.cif"
        doc = gemmi.cif.read(str(fixture))
        block = doc.sole_block()
        tree, assignments = assign_from_alt_ids(block)
        write_hierarchy(block, tree)
        write_atom_site_heterogeneity_ids(block, assignments)
        out_path = tmp_path / "out.cif"
        write_mmcif(doc, out_path)
        errors = validate_file(out_path, raise_on_error=False)
        assert errors == []
