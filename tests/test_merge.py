"""Unit tests for the merge helpers whose bugs a whole-file assertion would hide."""

from __future__ import annotations

from pathlib import Path

import gemmi
import pytest

from pdbx_hierarchy.exceptions import PdbxValidationError
from pdbx_hierarchy.merge import SourceModel, _next_free_ids, _union_chem_comp
from pdbx_hierarchy.models.hierarchy import HierarchyState, HierarchyTree

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestNextFreeIds:
    def test_ids_continue_above_the_highest_taken(self) -> None:
        assert _next_free_ids({"A", "B"}, 2) == ["C", "D"]

    def test_a_gap_is_skipped_past_not_filled(self) -> None:
        # The ground tree keeps A and C; the changed tree must land above C, not
        # reuse the free B, so that reading the ids in order reads ground first.
        assert _next_free_ids({"A", "C"}, 2) == ["D", "E"]

    def test_named_ids_are_avoided_but_do_not_set_the_floor(self) -> None:
        assert _next_free_ids({"Base", "Ground", "Changed"}, 3) == ["A", "B", "C"]

    def test_the_sequence_widens_past_z(self) -> None:
        alphabet = {chr(code) for code in range(ord("A"), ord("Z") + 1)}
        assert _next_free_ids(alphabet, 2) == ["AA", "AB"]

    def test_no_ids_requested(self) -> None:
        assert _next_free_ids({"A"}, 0) == []

    def test_a_rejected_candidate_is_skipped(self) -> None:
        # An id whose generated name would collide with a name the ground tree
        # already uses is no more usable than an id that collides outright.
        assert _next_free_ids({"A"}, 2, reject=lambda candidate: candidate in {"B", "C"}) == ["D", "E"]


def _source(name: str, cif: str, tmp_path: Path) -> SourceModel:
    """Build a SourceModel carrying just enough for _union_chem_comp."""
    path = tmp_path / name
    path.write_text(cif)
    block = gemmi.cif.read(str(path)).sole_block()
    return SourceModel(
        path=path,
        block=block,
        structure=gemmi.Structure(),
        tree=HierarchyTree(states=[HierarchyState(id="Base", name="base_state", parent=None)]),
        atom_state_ids=[],
        reused_hierarchy=False,
    )


_TWO_COMPONENTS = """data_A
loop_
_chem_comp.id
_chem_comp.formula
ALA 'C3 H7 N O2'
EDO 'C2 H6 O2'
"""

_OVERLAPPING = """data_B
loop_
_chem_comp.id
_chem_comp.formula
ALA 'C3 H7 N O2'
W2S 'C4 H6 N2 O'
"""

_CONFLICTING = """data_B
loop_
_chem_comp.id
_chem_comp.formula
ALA 'C9 H9 N O9'
"""

_EXTRA_COLUMN = """data_B
loop_
_chem_comp.id
_chem_comp.formula
_chem_comp.type
W2S 'C4 H6 N2 O' non-polymer
"""


class TestUnionChemComp:
    def test_components_from_both_inputs_survive(self, tmp_path: Path) -> None:
        union = _union_chem_comp(
            [_source("a.cif", _TWO_COMPONENTS, tmp_path), _source("b.cif", _OVERLAPPING, tmp_path)]
        )
        assert union["id"] == ["ALA", "EDO", "W2S"]

    def test_a_differing_formula_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(PdbxValidationError, match="ALA"):
            _union_chem_comp([_source("a.cif", _TWO_COMPONENTS, tmp_path), _source("b.cif", _CONFLICTING, tmp_path)])

    def test_a_column_only_one_input_has_is_filled_in(self, tmp_path: Path) -> None:
        union = _union_chem_comp(
            [_source("a.cif", _TWO_COMPONENTS, tmp_path), _source("b.cif", _EXTRA_COLUMN, tmp_path)]
        )
        assert union["id"] == ["ALA", "EDO", "W2S"]
        assert union["type"] == ["?", "?", "non-polymer"]

    def test_no_chem_comp_in_either_input(self, tmp_path: Path) -> None:
        empty = "data_A\n_entry.id A\n"
        assert _union_chem_comp([_source("a.cif", empty, tmp_path), _source("b.cif", empty, tmp_path)]) == {}
