"""Shared pytest fixtures for pdbx_hierarchy tests."""

from pathlib import Path

import gemmi
import pytest

from pdbx_hierarchy.models.coexistence import CoexistenceRule, CoexistenceTable, StateCoexistence
from pdbx_hierarchy.models.hierarchy import HierarchyState, HierarchyTree

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def hierarchy_only_cif() -> Path:
    return FIXTURES_DIR / "hierarchy_only.cif"


@pytest.fixture
def full_hierarchy_cif() -> Path:
    return FIXTURES_DIR / "full_hierarchy.cif"


@pytest.fixture
def atom_site_no_hierarchy_cif() -> Path:
    return FIXTURES_DIR / "atom_site_no_hierarchy.cif"


@pytest.fixture
def bad_atom_site_refs_cif() -> Path:
    return FIXTURES_DIR / "bad_atom_site_refs.cif"


@pytest.fixture
def simple_hierarchy() -> HierarchyTree:
    """HierarchyTree with Base -> A, B."""
    return HierarchyTree(
        states=[
            HierarchyState(id="Base", name="base_state", parent=None),
            HierarchyState(id="A", name="state_a", parent="Base"),
            HierarchyState(id="B", name="state_b", parent="Base"),
        ]
    )


@pytest.fixture
def simple_coexistence() -> CoexistenceTable:
    """CoexistenceTable with one NOT rule: Base -> [A]."""
    return CoexistenceTable(
        rules=[
            StateCoexistence(
                id=1,
                rule=CoexistenceRule.NOT,
                heterogeneity_id="Base",
                heterogeneity_ids=["A"],
            )
        ]
    )


@pytest.fixture
def minimal_block() -> gemmi.cif.Block:
    """Fresh gemmi Block with a 2-row atom_site loop, no hierarchy tables."""
    doc = gemmi.cif.Document()
    block = doc.add_new_block("MINIMAL")
    loop = block.init_loop("_atom_site.", ["id", "type_symbol"])
    loop.add_row(["1", "C"])
    loop.add_row(["2", "N"])
    return block


@pytest.fixture
def tmp_cif_path(tmp_path: Path) -> Path:
    return tmp_path / "output.cif"
