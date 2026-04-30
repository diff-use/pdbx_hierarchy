"""Tests for HierarchyState and HierarchyTree models."""

import pytest

from pdbx_hierarchy.exceptions import InvalidHierarchyError
from pdbx_hierarchy.models import HierarchyState, HierarchyTree


class TestHierarchyState:
    """Tests for HierarchyState model."""

    def test_valid_construction(self) -> None:
        """HierarchyState can be constructed with valid data."""
        state = HierarchyState(id="A", name="state_a", parent="Base")
        assert state.id == "A"
        assert state.name == "state_a"
        assert state.parent == "Base"
        assert state.details is None

    def test_root_state(self) -> None:
        """Root state has no parent and is_root is True."""
        state = HierarchyState(id="Base", name="base_state", parent=None)
        assert state.is_root is True
        assert state.parent is None

    def test_non_root_state(self) -> None:
        """Non-root state has parent and is_root is False."""
        state = HierarchyState(id="A", name="state_a", parent="Base")
        assert state.is_root is False

    def test_id_no_whitespace(self) -> None:
        """ID cannot contain whitespace."""
        with pytest.raises(ValueError, match="whitespace"):
            HierarchyState(id="A B", name="state", parent=None)

    def test_id_not_empty(self) -> None:
        """ID cannot be empty."""
        with pytest.raises(ValueError, match="empty"):
            HierarchyState(id="", name="state", parent=None)

    def test_name_no_whitespace(self) -> None:
        """Name cannot contain whitespace."""
        with pytest.raises(ValueError, match="whitespace"):
            HierarchyState(id="A", name="state a", parent=None)

    def test_name_not_empty(self) -> None:
        """Name cannot be empty."""
        with pytest.raises(ValueError, match="empty"):
            HierarchyState(id="A", name="", parent=None)

    def test_to_mmcif_row(self) -> None:
        """to_mmcif_row converts to dict with mmCIF format."""
        state = HierarchyState(id="A", name="state_a", parent="Base", details="Some details")
        row = state.to_mmcif_row()

        assert row == {
            "id": "A",
            "name": "state_a",
            "parent": "Base",
            "details": "Some details",
        }

    def test_to_mmcif_row_none_values(self) -> None:
        """to_mmcif_row converts None to '.' for parent and '?' for details."""
        state = HierarchyState(id="Base", name="base_state", parent=None, details=None)
        row = state.to_mmcif_row()

        assert row["parent"] == "."
        assert row["details"] == "?"

    def test_from_mmcif_row(self) -> None:
        """from_mmcif_row parses mmCIF dict correctly."""
        row = {"id": "A", "name": "state_a", "parent": "Base", "details": "info"}
        state = HierarchyState.from_mmcif_row(row)

        assert state.id == "A"
        assert state.name == "state_a"
        assert state.parent == "Base"
        assert state.details == "info"

    def test_from_mmcif_row_none_values(self) -> None:
        """from_mmcif_row converts '.' and '?' to None."""
        row = {"id": "Base", "name": "base_state", "parent": ".", "details": "?"}
        state = HierarchyState.from_mmcif_row(row)

        assert state.parent is None
        assert state.details is None

    def test_mmcif_roundtrip(self) -> None:
        """State survives to_mmcif_row/from_mmcif_row roundtrip."""
        original = HierarchyState(id="A", name="state_a", parent="Base", details="test")
        row = original.to_mmcif_row()
        restored = HierarchyState.from_mmcif_row(row)

        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.parent == original.parent
        assert restored.details == original.details

    def test_mutable_fields(self) -> None:
        """Fields can be modified after construction."""
        state = HierarchyState(id="A", name="state_a", parent="Base")
        state.name = "new_name"
        assert state.name == "new_name"

    def test_validation_on_assignment(self) -> None:
        """Validation runs when fields are assigned."""
        state = HierarchyState(id="A", name="state_a", parent="Base")
        with pytest.raises(ValueError, match="whitespace"):
            state.name = "invalid name"


class TestHierarchyTree:
    """Tests for HierarchyTree model."""

    @pytest.fixture
    def simple_tree(self) -> HierarchyTree:
        """A simple tree with Base and two children."""
        return HierarchyTree(
            states=[
                HierarchyState(id="Base", name="base_state", parent=None),
                HierarchyState(id="A", name="state_a", parent="Base"),
                HierarchyState(id="B", name="state_b", parent="Base"),
            ]
        )

    @pytest.fixture
    def nested_tree(self) -> HierarchyTree:
        """A tree with nested children: Base -> A -> AA, AB."""
        return HierarchyTree(
            states=[
                HierarchyState(id="Base", name="base_state", parent=None),
                HierarchyState(id="A", name="state_a", parent="Base"),
                HierarchyState(id="AA", name="state_aa", parent="A"),
                HierarchyState(id="AB", name="state_ab", parent="A"),
            ]
        )

    def test_valid_tree_construction(self, simple_tree: HierarchyTree) -> None:
        """Tree can be constructed with valid states."""
        assert len(simple_tree) == 3
        assert simple_tree.get_root().id == "Base"

    def test_empty_tree_fails(self) -> None:
        """Tree with no states fails validation."""
        with pytest.raises(InvalidHierarchyError, match="exactly one root"):
            HierarchyTree(states=[])

    def test_no_root_fails(self) -> None:
        """Tree without a root state fails."""
        with pytest.raises(InvalidHierarchyError, match="exactly one root"):
            HierarchyTree(
                states=[
                    HierarchyState(id="A", name="state_a", parent="Base"),
                ]
            )

    def test_multiple_roots_fails(self) -> None:
        """Tree with multiple roots fails."""
        with pytest.raises(InvalidHierarchyError, match="exactly one root"):
            HierarchyTree(
                states=[
                    HierarchyState(id="Base", name="base_state", parent=None),
                    HierarchyState(id="Root2", name="another_root", parent=None),
                ]
            )

    def test_duplicate_id_fails(self) -> None:
        """Tree with duplicate IDs fails."""
        with pytest.raises(InvalidHierarchyError, match="Duplicate state ID"):
            HierarchyTree(
                states=[
                    HierarchyState(id="Base", name="base_state", parent=None),
                    HierarchyState(id="A", name="state_a", parent="Base"),
                    HierarchyState(id="A", name="state_a2", parent="Base"),
                ]
            )

    def test_duplicate_name_fails(self) -> None:
        """Tree with duplicate names fails."""
        with pytest.raises(InvalidHierarchyError, match="Duplicate state name"):
            HierarchyTree(
                states=[
                    HierarchyState(id="Base", name="base_state", parent=None),
                    HierarchyState(id="A", name="same_name", parent="Base"),
                    HierarchyState(id="B", name="same_name", parent="Base"),
                ]
            )

    def test_invalid_parent_ref_fails(self) -> None:
        """Tree with invalid parent reference fails."""
        with pytest.raises(InvalidHierarchyError, match="non-existent parent"):
            HierarchyTree(
                states=[
                    HierarchyState(id="Base", name="base_state", parent=None),
                    HierarchyState(id="A", name="state_a", parent="NonExistent"),
                ]
            )

    def test_orphan_detection(self) -> None:
        """Tree with orphaned states fails."""
        with pytest.raises(InvalidHierarchyError, match="Orphaned states"):
            HierarchyTree(
                states=[
                    HierarchyState(id="Base", name="base_state", parent=None),
                    HierarchyState(id="A", name="state_a", parent="B"),  # A points to B
                    HierarchyState(id="B", name="state_b", parent="A"),  # B points to A (cycle)
                ]
            )

    def test_get_root(self, simple_tree: HierarchyTree) -> None:
        """get_root returns the root state."""
        root = simple_tree.get_root()
        assert root.id == "Base"
        assert root.is_root is True

    def test_get_state(self, simple_tree: HierarchyTree) -> None:
        """get_state returns state by ID."""
        state = simple_tree.get_state("A")
        assert state.id == "A"
        assert state.name == "state_a"

    def test_get_state_not_found(self, simple_tree: HierarchyTree) -> None:
        """get_state raises KeyError for unknown ID."""
        with pytest.raises(KeyError, match="No state with ID"):
            simple_tree.get_state("NonExistent")

    def test_get_state_by_name(self, simple_tree: HierarchyTree) -> None:
        """get_state_by_name returns state by name."""
        state = simple_tree.get_state_by_name("state_a")
        assert state.id == "A"

    def test_get_state_by_name_not_found(self, simple_tree: HierarchyTree) -> None:
        """get_state_by_name raises KeyError for unknown name."""
        with pytest.raises(KeyError, match="No state with name"):
            simple_tree.get_state_by_name("nonexistent")

    def test_get_children(self, simple_tree: HierarchyTree) -> None:
        """get_children returns direct children."""
        children = simple_tree.get_children("Base")
        child_ids = [c.id for c in children]
        assert set(child_ids) == {"A", "B"}

    def test_get_children_leaf(self, simple_tree: HierarchyTree) -> None:
        """get_children returns empty list for leaf nodes."""
        children = simple_tree.get_children("A")
        assert children == []

    def test_get_children_not_found(self, simple_tree: HierarchyTree) -> None:
        """get_children raises KeyError for unknown ID."""
        with pytest.raises(KeyError):
            simple_tree.get_children("NonExistent")

    def test_get_ancestors(self, nested_tree: HierarchyTree) -> None:
        """get_ancestors returns parent chain to root."""
        ancestors = nested_tree.get_ancestors("AA")
        ancestor_ids = [a.id for a in ancestors]
        assert ancestor_ids == ["A", "Base"]

    def test_get_ancestors_root(self, nested_tree: HierarchyTree) -> None:
        """get_ancestors returns empty list for root."""
        ancestors = nested_tree.get_ancestors("Base")
        assert ancestors == []

    def test_get_descendants(self, nested_tree: HierarchyTree) -> None:
        """get_descendants returns all descendants in breadth-first order."""
        descendants = nested_tree.get_descendants("Base")
        desc_ids = [d.id for d in descendants]
        # A first, then its children AA and AB
        assert desc_ids[0] == "A"
        assert set(desc_ids[1:]) == {"AA", "AB"}

    def test_get_descendants_leaf(self, nested_tree: HierarchyTree) -> None:
        """get_descendants returns empty list for leaf nodes."""
        descendants = nested_tree.get_descendants("AA")
        assert descendants == []

    def test_contains(self, simple_tree: HierarchyTree) -> None:
        """contains checks if ID exists."""
        assert simple_tree.contains("A") is True
        assert simple_tree.contains("NonExistent") is False

    def test_len(self, simple_tree: HierarchyTree) -> None:
        """len returns number of states."""
        assert len(simple_tree) == 3

    def test_iter_via_states(self, simple_tree: HierarchyTree) -> None:
        """Can iterate via .states attribute."""
        states = list(simple_tree.states)
        assert len(states) == 3
        assert all(isinstance(s, HierarchyState) for s in states)

    def test_add_state(self, simple_tree: HierarchyTree) -> None:
        """add_state adds a new state and validates."""
        new_state = HierarchyState(id="C", name="state_c", parent="Base")
        simple_tree.add_state(new_state)

        assert len(simple_tree) == 4
        assert simple_tree.contains("C")

    def test_add_state_invalid(self, simple_tree: HierarchyTree) -> None:
        """add_state fails if it would create invalid tree."""
        # Adding second root
        with pytest.raises(InvalidHierarchyError):
            simple_tree.add_state(HierarchyState(id="Root2", name="root2", parent=None))

    def test_remove_state(self, simple_tree: HierarchyTree) -> None:
        """remove_state removes a state and validates."""
        removed = simple_tree.remove_state("A")

        assert removed.id == "A"
        assert len(simple_tree) == 2
        assert not simple_tree.contains("A")

    def test_remove_state_not_found(self, simple_tree: HierarchyTree) -> None:
        """remove_state raises KeyError for unknown ID."""
        with pytest.raises(KeyError):
            simple_tree.remove_state("NonExistent")

    def test_remove_state_with_children_fails(self, nested_tree: HierarchyTree) -> None:
        """remove_state fails if state has children (creates orphans)."""
        with pytest.raises(InvalidHierarchyError, match="non-existent parent"):
            nested_tree.remove_state("A")

    def test_update_state(self, simple_tree: HierarchyTree) -> None:
        """update_state modifies a state and validates."""
        updated = simple_tree.update_state("A", name="new_name")

        assert updated.name == "new_name"
        assert simple_tree.get_state("A").name == "new_name"

    def test_update_state_invalid(self, simple_tree: HierarchyTree) -> None:
        """update_state fails if it would create invalid tree."""
        # Creating duplicate name
        with pytest.raises(InvalidHierarchyError, match="Duplicate state name"):
            simple_tree.update_state("A", name="state_b")

    def test_to_mmcif_rows(self, simple_tree: HierarchyTree) -> None:
        """to_mmcif_rows converts all states to dicts."""
        rows = simple_tree.to_mmcif_rows()

        assert len(rows) == 3
        ids = [r["id"] for r in rows]
        assert set(ids) == {"Base", "A", "B"}

    def test_from_mmcif_rows(self) -> None:
        """from_mmcif_rows creates tree from dicts."""
        rows = [
            {"id": "Base", "name": "base_state", "parent": ".", "details": "?"},
            {"id": "A", "name": "state_a", "parent": "Base", "details": "?"},
        ]
        tree = HierarchyTree.from_mmcif_rows(rows)

        assert len(tree) == 2
        assert tree.get_root().id == "Base"

    def test_mmcif_roundtrip(self, simple_tree: HierarchyTree) -> None:
        """Tree survives mmCIF roundtrip."""
        rows = simple_tree.to_mmcif_rows()
        restored = HierarchyTree.from_mmcif_rows(rows)

        assert len(restored) == len(simple_tree)
        for state in simple_tree.states:
            restored_state = restored.get_state(state.id)
            assert restored_state.name == state.name
            assert restored_state.parent == state.parent
