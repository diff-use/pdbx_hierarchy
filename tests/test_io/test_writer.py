"""Tests for src/pdbx_hierarchy/io/writer.py."""

from pathlib import Path

import gemmi
import pytest

from pdbx_hierarchy.exceptions import PdbxValidationError
from pdbx_hierarchy.io.reader import read_coexistence, read_hierarchy
from pdbx_hierarchy.io.writer import (
    write_atom_site_heterogeneity_ids,
    write_coexistence,
    write_hierarchy,
    write_mmcif,
)
from pdbx_hierarchy.models.coexistence import CoexistenceRule, CoexistenceTable, StateCoexistence
from pdbx_hierarchy.models.hierarchy import HierarchyState, HierarchyTree


def _fresh_doc_with_block(name: str = "TEST") -> tuple[gemmi.cif.Document, gemmi.cif.Block]:
    """Create a new Document with one named Block."""
    doc = gemmi.cif.Document()
    block = doc.add_new_block(name)
    return doc, block


def _block_with_atom_site(rows: int = 3, name: str = "TEST") -> tuple[gemmi.cif.Document, gemmi.cif.Block]:
    """Create a Document/Block with an _atom_site loop of the given row count."""
    doc, block = _fresh_doc_with_block(name)
    loop = block.init_loop("_atom_site.", ["id", "type_symbol"])
    for i in range(1, rows + 1):
        loop.add_row([str(i), "C"])
    return doc, block


class TestWriteHierarchy:
    def test_creates_loop(self, simple_hierarchy: HierarchyTree) -> None:
        _, block = _fresh_doc_with_block()
        write_hierarchy(block, simple_hierarchy)
        assert bool(block.find_loop("_pdbx_heterogeneity_hierarchy.id"))

    def test_row_count(self, simple_hierarchy: HierarchyTree) -> None:
        _, block = _fresh_doc_with_block()
        write_hierarchy(block, simple_hierarchy)
        col = block.find_loop("_pdbx_heterogeneity_hierarchy.id")
        assert len(col) == len(simple_hierarchy)

    def test_root_parent_dot(self, simple_hierarchy: HierarchyTree) -> None:
        _, block = _fresh_doc_with_block()
        write_hierarchy(block, simple_hierarchy)
        tbl = block.find("_pdbx_heterogeneity_hierarchy.", ["id", "parent"])
        parent_by_id = {row[0]: row[1] for row in tbl}
        assert parent_by_id["Base"] == "."

    def test_details_question_mark(self, simple_hierarchy: HierarchyTree) -> None:
        _, block = _fresh_doc_with_block()
        write_hierarchy(block, simple_hierarchy)
        tbl = block.find("_pdbx_heterogeneity_hierarchy.", ["id", "details"])
        # All states in simple_hierarchy have details=None -> should be "?"
        for row in tbl:
            assert row[1] == "?"

    def test_overwrite_raises(self, simple_hierarchy: HierarchyTree) -> None:
        _, block = _fresh_doc_with_block()
        write_hierarchy(block, simple_hierarchy)
        with pytest.raises(PdbxValidationError):
            write_hierarchy(block, simple_hierarchy)

    def test_overwrite_allowed(self, simple_hierarchy: HierarchyTree) -> None:
        _, block = _fresh_doc_with_block()
        write_hierarchy(block, simple_hierarchy)
        write_hierarchy(block, simple_hierarchy, overwrite=True)
        col = block.find_loop("_pdbx_heterogeneity_hierarchy.id")
        assert len(col) == len(simple_hierarchy)

    def test_round_trip(self, simple_hierarchy: HierarchyTree) -> None:
        _, block = _fresh_doc_with_block()
        write_hierarchy(block, simple_hierarchy)
        result = read_hierarchy(block)
        assert len(result) == len(simple_hierarchy)
        assert {s.id for s in result.states} == {s.id for s in simple_hierarchy.states}

    def test_details_with_spaces_round_trip_through_file(self, tmp_path: Path) -> None:
        # Regression: values with whitespace must be quoted, or the loop corrupts.
        tree = HierarchyTree(
            states=[
                HierarchyState(id="Base", name="base_state", parent=None),
                HierarchyState(id="A", name="state_a", parent="Base", details="an alternate loop conformation"),
            ]
        )
        doc, block = _fresh_doc_with_block()
        write_hierarchy(block, tree)
        path = tmp_path / "spaced.cif"
        write_mmcif(doc, path)

        result = read_hierarchy(path)  # reparses the serialized text
        assert result.get_state("A").details == "an alternate loop conformation"

    @pytest.mark.parametrize(
        "details",
        [
            "simple",
            "two words",
            "leading _underscore",
            "_starts_with_underscore",
            "loop_",
            "data_thing",
            "has 'single' quotes",
            'has "double" quotes',
            "has 'both' \"kinds\"",
            "trailing space ",
            "tab\tseparated",
        ],
    )
    def test_arbitrary_details_round_trip_through_file(self, tmp_path: Path, details: str) -> None:
        tree = HierarchyTree(
            states=[
                HierarchyState(id="Base", name="base_state", parent=None),
                HierarchyState(id="A", name="state_a", parent="Base", details=details),
            ]
        )
        doc, block = _fresh_doc_with_block()
        write_hierarchy(block, tree)
        path = tmp_path / "detail.cif"
        write_mmcif(doc, path)
        assert read_hierarchy(path).get_state("A").details == details


class TestWriteCoexistence:
    def test_creates_loop(self, simple_coexistence: CoexistenceTable) -> None:
        _, block = _fresh_doc_with_block()
        write_coexistence(block, simple_coexistence)
        assert bool(block.find_loop("_pdbx_state_coexistence.id"))

    def test_row_count(self, simple_coexistence: CoexistenceTable) -> None:
        _, block = _fresh_doc_with_block()
        write_coexistence(block, simple_coexistence)
        col = block.find_loop("_pdbx_state_coexistence.id")
        assert len(col) == len(simple_coexistence)

    def test_description_question_mark(self, simple_coexistence: CoexistenceTable) -> None:
        _, block = _fresh_doc_with_block()
        write_coexistence(block, simple_coexistence)
        tbl = block.find("_pdbx_state_coexistence.", ["id", "description"])
        for row in tbl:
            assert row[1] == "?"

    def test_heterogeneity_ids_comma(self, simple_coexistence: CoexistenceTable) -> None:
        _, block = _fresh_doc_with_block()
        write_coexistence(block, simple_coexistence)
        tbl = block.find("_pdbx_state_coexistence.", ["heterogeneity_ids"])
        # simple_coexistence has heterogeneity_ids=["A"] -> "A"
        assert tbl[0][0] == "A"

    def test_description_with_spaces_round_trip_through_file(self, tmp_path: Path) -> None:
        # Regression: a spaced description must be quoted, or the loop corrupts.
        table = CoexistenceTable(
            rules=[
                StateCoexistence(
                    id=1,
                    rule=CoexistenceRule.NOT,
                    heterogeneity_id="A",
                    heterogeneity_ids=["B"],
                    description="mutually exclusive per clash analysis",
                )
            ]
        )
        doc, block = _fresh_doc_with_block()
        write_coexistence(block, table)
        path = tmp_path / "spaced.cif"
        write_mmcif(doc, path)

        result = read_coexistence(path)  # reparses the serialized text
        assert result is not None
        assert result.rules[0].description == "mutually exclusive per clash analysis"

    def test_overwrite_raises(self, simple_coexistence: CoexistenceTable) -> None:
        _, block = _fresh_doc_with_block()
        write_coexistence(block, simple_coexistence)
        with pytest.raises(PdbxValidationError):
            write_coexistence(block, simple_coexistence)

    def test_overwrite_allowed(self, simple_coexistence: CoexistenceTable) -> None:
        _, block = _fresh_doc_with_block()
        write_coexistence(block, simple_coexistence)
        write_coexistence(block, simple_coexistence, overwrite=True)
        col = block.find_loop("_pdbx_state_coexistence.id")
        assert len(col) == len(simple_coexistence)

    def test_round_trip(self, simple_coexistence: CoexistenceTable) -> None:
        _, block = _fresh_doc_with_block()
        write_coexistence(block, simple_coexistence)
        result = read_coexistence(block)
        assert result is not None
        assert len(result) == len(simple_coexistence)
        assert result.rules[0].rule == simple_coexistence.rules[0].rule
        assert result.rules[0].heterogeneity_ids == simple_coexistence.rules[0].heterogeneity_ids


class TestWriteAtomSiteHeterogeneityIds:
    def test_adds_column(self) -> None:
        _, block = _block_with_atom_site(2)
        write_atom_site_heterogeneity_ids(block, ["Base", "A"])
        assert bool(block.find_loop("_atom_site.pdbx_heterogeneity_id"))

    def test_values(self) -> None:
        _, block = _block_with_atom_site(2)
        write_atom_site_heterogeneity_ids(block, ["Base", "A"])
        col = block.find_loop("_atom_site.pdbx_heterogeneity_id")
        assert list(col) == ["Base", "A"]

    def test_length_mismatch(self) -> None:
        _, block = _block_with_atom_site(3)
        with pytest.raises(PdbxValidationError, match="does not match atom_site row count"):
            write_atom_site_heterogeneity_ids(block, ["Base", "A"])

    def test_no_atom_site(self) -> None:
        _, block = _fresh_doc_with_block()
        with pytest.raises(PdbxValidationError, match="No _atom_site loop"):
            write_atom_site_heterogeneity_ids(block, ["Base"])

    def test_overwrite_raises(self) -> None:
        _, block = _block_with_atom_site(2)
        write_atom_site_heterogeneity_ids(block, ["Base", "A"])
        with pytest.raises(PdbxValidationError):
            write_atom_site_heterogeneity_ids(block, ["Base", "A"])

    def test_overwrite_allowed(self) -> None:
        _, block = _block_with_atom_site(2)
        write_atom_site_heterogeneity_ids(block, ["Base", "A"])
        write_atom_site_heterogeneity_ids(block, ["A", "Base"], overwrite=True)
        col = block.find_loop("_atom_site.pdbx_heterogeneity_id")
        assert list(col) == ["A", "Base"]

    def test_preserves_existing_columns(self) -> None:
        _, block = _block_with_atom_site(2)
        write_atom_site_heterogeneity_ids(block, ["Base", "A"])
        # The original id and type_symbol columns must still be there
        assert bool(block.find_loop("_atom_site.id"))
        assert bool(block.find_loop("_atom_site.type_symbol"))


class TestWriteMmcif:
    def test_creates_file(self, simple_hierarchy: HierarchyTree, tmp_cif_path: Path) -> None:
        doc, block = _fresh_doc_with_block()
        write_hierarchy(block, simple_hierarchy)
        write_mmcif(doc, tmp_cif_path)
        assert tmp_cif_path.exists()

    def test_readable(self, simple_hierarchy: HierarchyTree, tmp_cif_path: Path) -> None:
        doc, block = _fresh_doc_with_block("ROUNDTRIP")
        write_hierarchy(block, simple_hierarchy)
        write_mmcif(doc, tmp_cif_path)
        reloaded = gemmi.cif.read(str(tmp_cif_path))
        assert len(reloaded) == 1

    def test_preserves_hierarchy(self, simple_hierarchy: HierarchyTree, tmp_cif_path: Path) -> None:
        doc, block = _fresh_doc_with_block("ROUNDTRIP")
        write_hierarchy(block, simple_hierarchy)
        write_mmcif(doc, tmp_cif_path)
        result = read_hierarchy(tmp_cif_path)
        assert {s.id for s in result.states} == {s.id for s in simple_hierarchy.states}

    def test_pdbx_style(self, simple_hierarchy: HierarchyTree, tmp_cif_path: Path) -> None:
        doc, block = _fresh_doc_with_block()
        write_hierarchy(block, simple_hierarchy)
        write_mmcif(doc, tmp_cif_path)
        content = tmp_cif_path.read_text()
        assert "loop_" in content
        assert "#" in content
