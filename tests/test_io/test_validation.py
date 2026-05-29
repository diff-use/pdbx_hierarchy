"""Tests for src/pdbx_hierarchy/io/validation.py."""

from pathlib import Path

import gemmi
import pytest

from pdbx_hierarchy.exceptions import AtomSiteReferenceError
from pdbx_hierarchy.io.reader import read_hierarchy, read_mmcif
from pdbx_hierarchy.io.validation import validate_atom_site_references, validate_file
from pdbx_hierarchy.models.hierarchy import HierarchyTree


class TestValidateAtomSiteReferences:
    def test_valid(self, full_hierarchy_cif: Path) -> None:
        hierarchy = read_hierarchy(full_hierarchy_cif)
        errors = validate_atom_site_references(full_hierarchy_cif, hierarchy, raise_on_error=False)
        assert errors == []

    def test_bad_ref(self, bad_atom_site_refs_cif: Path) -> None:
        hierarchy = read_hierarchy(bad_atom_site_refs_cif)
        errors = validate_atom_site_references(bad_atom_site_refs_cif, hierarchy, raise_on_error=False)
        assert len(errors) == 1

    def test_bad_ref_message(self, bad_atom_site_refs_cif: Path) -> None:
        hierarchy = read_hierarchy(bad_atom_site_refs_cif)
        errors = validate_atom_site_references(bad_atom_site_refs_cif, hierarchy, raise_on_error=False)
        assert "NONEXISTENT" in errors[0]

    def test_raises(self, bad_atom_site_refs_cif: Path) -> None:
        hierarchy = read_hierarchy(bad_atom_site_refs_cif)
        with pytest.raises(AtomSiteReferenceError):
            validate_atom_site_references(bad_atom_site_refs_cif, hierarchy, raise_on_error=True)

    def test_no_column(self, atom_site_no_hierarchy_cif: Path, simple_hierarchy: HierarchyTree) -> None:
        errors = validate_atom_site_references(atom_site_no_hierarchy_cif, simple_hierarchy, raise_on_error=False)
        assert errors == []

    def test_from_block(self, full_hierarchy_cif: Path) -> None:
        block = read_mmcif(full_hierarchy_cif)
        hierarchy = read_hierarchy(block)
        errors = validate_atom_site_references(block, hierarchy, raise_on_error=False)
        assert errors == []

    def test_all_bad(self, tmp_path: Path) -> None:
        cif = tmp_path / "all_bad.cif"
        cif.write_text(
            "data_ALL_BAD\n#\n"
            "loop_\n_pdbx_heterogeneity_hierarchy.id\n_pdbx_heterogeneity_hierarchy.name\n"
            "_pdbx_heterogeneity_hierarchy.parent\n_pdbx_heterogeneity_hierarchy.details\n"
            "Base base_state . ?\n#\n"
            "loop_\n_atom_site.id\n_atom_site.pdbx_heterogeneity_id\n"
            "1 BAD1\n2 BAD2\n#\n"
        )
        hierarchy = read_hierarchy(cif)
        errors = validate_atom_site_references(cif, hierarchy, raise_on_error=False)
        assert len(errors) == 2


class TestValidateFile:
    def test_valid_full(self, full_hierarchy_cif: Path) -> None:
        errors = validate_file(full_hierarchy_cif, raise_on_error=False)
        assert errors == []

    def test_no_hierarchy(self, atom_site_no_hierarchy_cif: Path) -> None:
        errors = validate_file(atom_site_no_hierarchy_cif, raise_on_error=False)
        assert errors == []

    def test_bad_atom_refs(self, bad_atom_site_refs_cif: Path) -> None:
        errors = validate_file(bad_atom_site_refs_cif, raise_on_error=False)
        assert len(errors) >= 1

    def test_bad_atom_refs_raises(self, bad_atom_site_refs_cif: Path) -> None:
        with pytest.raises(AtomSiteReferenceError):
            validate_file(bad_atom_site_refs_cif, raise_on_error=True)

    def test_hierarchy_only_no_errors(self, hierarchy_only_cif: Path) -> None:
        errors = validate_file(hierarchy_only_cif, raise_on_error=False)
        assert errors == []

    def test_bad_coexistence(self) -> None:
        doc = gemmi.cif.Document()
        block = doc.add_new_block("BAD_COX")
        h_loop = block.init_loop(
            "_pdbx_heterogeneity_hierarchy.",
            ["id", "name", "parent", "details"],
        )
        h_loop.add_row(["Base", "base_state", ".", "?"])
        c_loop = block.init_loop(
            "_pdbx_state_coexistence.",
            ["id", "rule", "heterogeneity_id", "heterogeneity_ids", "description"],
        )
        c_loop.add_row(["1", "NOT", "Base", "GHOST", "?"])
        errors = validate_file(block, raise_on_error=False)
        assert any("GHOST" in e for e in errors)

    def test_from_block(self, full_hierarchy_cif: Path) -> None:
        block = read_mmcif(full_hierarchy_cif)
        errors = validate_file(block, raise_on_error=False)
        assert errors == []
