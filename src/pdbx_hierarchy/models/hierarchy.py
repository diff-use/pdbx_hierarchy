"""Data models for the _pdbx_heterogeneity_hierarchy table.

This module provides Pydantic models representing the hierarchical structure
of heterogeneity states in a PDBx/mmCIF file.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr, field_validator

from pdbx_hierarchy.exceptions import InvalidHierarchyError


class HierarchyState(BaseModel):
    """A single row in the _pdbx_heterogeneity_hierarchy table.

    Represents a structural state in the heterogeneity hierarchy tree.

    Attributes:
        id: Unique state identifier. Can be any non-whitespace string.
            Generator produces "Base", "A"-"Z", "AA"-"ZZ", etc.
        name: Human-readable name without whitespace (e.g., "base_state").
        parent: Parent state's id, or None for the root state.
        details: Optional description of this state.

    Examples:
        >>> root = HierarchyState(id="Base", name="base_state", parent=None)
        >>> child = HierarchyState(id="A", name="state_A", parent="Base")
        >>> root.is_root
        True
        >>> child.to_mmcif_row()
        {'id': 'A', 'name': 'state_A', 'parent': 'Base', 'details': '?'}
    """

    model_config = {"validate_assignment": True}

    id: str = Field(..., description="Unique state identifier")
    name: str = Field(..., description="Human-readable name (no whitespace)")
    parent: str | None = Field(None, description="Parent state's id, None for root")
    details: str | None = Field(None, description="Optional description")

    @field_validator("id")
    @classmethod
    def validate_id_no_whitespace(cls, v: str) -> str:
        """Validate that id contains no whitespace."""
        if not v:
            raise ValueError("ID must not be empty")
        if re.search(r"\s", v):
            raise ValueError(f"ID must not contain whitespace, got: {v!r}")
        return v

    @field_validator("name")
    @classmethod
    def validate_name_no_whitespace(cls, v: str) -> str:
        """Validate that name contains no whitespace."""
        if not v:
            raise ValueError("Name must not be empty")
        if re.search(r"\s", v):
            raise ValueError(f"Name must not contain whitespace, got: {v!r}")
        return v

    @property
    def is_root(self) -> bool:
        """Return True if this is the root state (no parent)."""
        return self.parent is None

    def to_mmcif_row(self) -> dict[str, str]:
        """Convert to a dictionary suitable for mmCIF output.

        Returns:
            Dictionary with mmCIF-formatted values. None becomes "." for parent,
            "?" for details.
        """
        return {
            "id": self.id,
            "name": self.name,
            "parent": self.parent if self.parent is not None else ".",
            "details": self.details if self.details is not None else "?",
        }

    @classmethod
    def from_mmcif_row(cls, row: dict[str, str]) -> HierarchyState:
        """Create a HierarchyState from an mmCIF row dictionary.

        Args:
            row: Dictionary with keys 'id', 'name', 'parent', 'details'.
                 "." in parent is converted to None.
                 "?" in details is converted to None.

        Returns:
            A new HierarchyState instance.
        """
        return cls(
            id=row["id"],
            name=row["name"],
            parent=None if row.get("parent") in (".", None) else row["parent"],
            details=None if row.get("details") in ("?", None) else row.get("details"),
        )


class HierarchyTree(BaseModel):
    """Collection of HierarchyState objects forming a valid tree.

    Validates tree invariants on construction and provides methods for
    tree traversal and modification. All modifications re-validate the tree.

    Invariants enforced:
        - Exactly one root state (parent is None)
        - All IDs are unique
        - All names are unique
        - All parent references point to existing states
        - No cycles in the parent chain
        - No orphaned subtrees (all states reachable from root)

    Attributes:
        states: List of HierarchyState objects forming the tree.

    Examples:
        >>> root = HierarchyState(id="Base", name="base_state", parent=None)
        >>> child_a = HierarchyState(id="A", name="state_A", parent="Base")
        >>> tree = HierarchyTree(states=[root, child_a])
        >>> tree.get_root().id
        'Base'
        >>> [s.id for s in tree.get_children("Base")]
        ['A']
    """

    model_config = {"validate_assignment": True}

    states: list[HierarchyState] = Field(default_factory=list)

    # Private lookup tables rebuilt on validation
    _by_id: dict[str, HierarchyState] = PrivateAttr(default_factory=dict)
    _by_name: dict[str, HierarchyState] = PrivateAttr(default_factory=dict)
    _children: dict[str, list[str]] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """Validate tree structure and build lookup tables after init."""
        self._validate_and_rebuild_indices()

    def __setattr__(self, name: str, value: Any) -> None:
        """Override to rebuild indices when states is assigned."""
        super().__setattr__(name, value)
        if name == "states":
            self._validate_and_rebuild_indices()

    def _validate_and_rebuild_indices(self) -> None:
        """Validate tree invariants and rebuild internal lookup tables.

        Raises:
            InvalidHierarchyError: If any tree invariant is violated.
        """
        # Reset indices
        self._by_id = {}
        self._by_name = {}
        self._children = {}

        roots: list[HierarchyState] = []

        for state in self.states:
            # Check unique IDs
            if state.id in self._by_id:
                raise InvalidHierarchyError(f"Duplicate state ID: {state.id!r}")
            self._by_id[state.id] = state

            # Check unique names
            if state.name in self._by_name:
                raise InvalidHierarchyError(
                    f"Duplicate state name: {state.name!r} "
                    f"(used by IDs {self._by_name[state.name].id!r} and {state.id!r})"
                )
            self._by_name[state.name] = state

            # Track roots
            if state.is_root:
                roots.append(state)

        # Validate exactly one root
        if len(roots) == 0:
            raise InvalidHierarchyError("Hierarchy must have exactly one root state (found 0)")
        if len(roots) > 1:
            root_ids = [r.id for r in roots]
            raise InvalidHierarchyError(f"Hierarchy must have exactly one root state, found {len(roots)}: {root_ids}")

        # Validate parent references and build children index
        for state in self.states:
            if state.parent is not None:
                if state.parent not in self._by_id:
                    raise InvalidHierarchyError(f"State {state.id!r} references non-existent parent {state.parent!r}")
                self._children.setdefault(state.parent, []).append(state.id)

        # Check for cycles and orphans using DFS from root
        self._check_tree_connectivity()

    def _check_tree_connectivity(self) -> None:
        """Verify all states are reachable from root (no cycles, no orphans).

        Raises:
            InvalidHierarchyError: If a cycle is detected or states are orphaned.
        """
        visited: set[str] = set()
        path: set[str] = set()

        def dfs(state_id: str) -> None:
            if state_id in path:
                raise InvalidHierarchyError(f"Cycle detected involving state {state_id!r}")
            if state_id in visited:
                return

            visited.add(state_id)
            path.add(state_id)

            for child_id in self._children.get(state_id, []):
                dfs(child_id)

            path.remove(state_id)

        root = self.get_root()
        dfs(root.id)

        # Check all nodes were visited (no orphaned subtrees)
        if len(visited) != len(self.states):
            unvisited = {s.id for s in self.states} - visited
            raise InvalidHierarchyError(f"Orphaned states not connected to root: {unvisited}")

    # === Query Methods ===

    def get_root(self) -> HierarchyState:
        """Return the root state of the hierarchy.

        Returns:
            The unique root HierarchyState.
        """
        for state in self.states:
            if state.is_root:
                return state
        raise InvalidHierarchyError("No root state found")

    def get_state(self, state_id: str) -> HierarchyState:
        """Get a state by its ID.

        Args:
            state_id: The ID of the state to retrieve.

        Returns:
            The HierarchyState with the given ID.

        Raises:
            KeyError: If no state with that ID exists.
        """
        if state_id not in self._by_id:
            raise KeyError(f"No state with ID {state_id!r}")
        return self._by_id[state_id]

    def get_state_by_name(self, name: str) -> HierarchyState:
        """Get a state by its name.

        Args:
            name: The name of the state to retrieve.

        Returns:
            The HierarchyState with the given name.

        Raises:
            KeyError: If no state with that name exists.
        """
        if name not in self._by_name:
            raise KeyError(f"No state with name {name!r}")
        return self._by_name[name]

    def get_children(self, state_id: str) -> list[HierarchyState]:
        """Get all direct children of a state.

        Args:
            state_id: The ID of the parent state.

        Returns:
            List of child HierarchyState objects (may be empty).

        Raises:
            KeyError: If no state with that ID exists.
        """
        if state_id not in self._by_id:
            raise KeyError(f"No state with ID {state_id!r}")
        child_ids = self._children.get(state_id, [])
        return [self._by_id[cid] for cid in child_ids]

    def get_ancestors(self, state_id: str) -> list[HierarchyState]:
        """Get all ancestors of a state, from parent to root.

        Args:
            state_id: The ID of the state.

        Returns:
            List of ancestor HierarchyState objects, starting with the
            immediate parent and ending with the root. Empty for root state.

        Raises:
            KeyError: If no state with that ID exists.
        """
        state = self.get_state(state_id)
        ancestors: list[HierarchyState] = []
        current = state
        while current.parent is not None:
            parent = self._by_id[current.parent]
            ancestors.append(parent)
            current = parent
        return ancestors

    def get_descendants(self, state_id: str) -> list[HierarchyState]:
        """Get all descendants of a state (children, grandchildren, etc.).

        Args:
            state_id: The ID of the state.

        Returns:
            List of descendant HierarchyState objects in breadth-first order.

        Raises:
            KeyError: If no state with that ID exists.
        """
        if state_id not in self._by_id:
            raise KeyError(f"No state with ID {state_id!r}")

        descendants: list[HierarchyState] = []
        queue = list(self._children.get(state_id, []))

        while queue:
            child_id = queue.pop(0)
            child = self._by_id[child_id]
            descendants.append(child)
            queue.extend(self._children.get(child_id, []))

        return descendants

    def contains(self, state_id: str) -> bool:
        """Check if a state ID exists in the tree.

        Args:
            state_id: The ID to check.

        Returns:
            True if a state with this ID exists.
        """
        return state_id in self._by_id

    def __len__(self) -> int:
        """Return the number of states in the tree."""
        return len(self.states)

    # === Modification Methods ===

    def add_state(self, state: HierarchyState) -> None:
        """Add a new state to the tree.

        Args:
            state: The HierarchyState to add.

        Raises:
            InvalidHierarchyError: If adding this state would violate tree invariants.
        """
        new_states = list(self.states) + [state]
        self.states = new_states

    def remove_state(self, state_id: str) -> HierarchyState:
        """Remove a state from the tree.

        Args:
            state_id: The ID of the state to remove.

        Returns:
            The removed HierarchyState.

        Raises:
            KeyError: If no state with that ID exists.
            InvalidHierarchyError: If removal would violate tree invariants
                (e.g., removing root, leaving orphans).
        """
        if state_id not in self._by_id:
            raise KeyError(f"No state with ID {state_id!r}")

        removed = self._by_id[state_id]
        new_states = [s for s in self.states if s.id != state_id]
        self.states = new_states
        return removed

    def update_state(self, state_id: str, **kwargs: Any) -> HierarchyState:
        """Update fields of an existing state.

        Args:
            state_id: The ID of the state to update.
            **kwargs: Fields to update (name, parent, details).

        Returns:
            The updated HierarchyState.

        Raises:
            KeyError: If no state with that ID exists.
            InvalidHierarchyError: If the update would violate tree invariants.
            ValueError: If updating with invalid field values.
        """
        if state_id not in self._by_id:
            raise KeyError(f"No state with ID {state_id!r}")

        old_state = self._by_id[state_id]
        new_state = old_state.model_copy(update=kwargs)

        new_states = [new_state if s.id == state_id else s for s in self.states]
        self.states = new_states
        return self._by_id[state_id]

    # === Serialization ===

    def to_mmcif_rows(self) -> list[dict[str, str]]:
        """Convert all states to mmCIF row dictionaries.

        Returns:
            List of dictionaries suitable for writing to mmCIF.
        """
        return [state.to_mmcif_row() for state in self.states]

    @classmethod
    def from_mmcif_rows(cls, rows: list[dict[str, str]]) -> HierarchyTree:
        """Create a HierarchyTree from mmCIF row dictionaries.

        Args:
            rows: List of dictionaries from mmCIF parsing.

        Returns:
            A validated HierarchyTree.

        Raises:
            InvalidHierarchyError: If the data forms an invalid tree.
        """
        states = [HierarchyState.from_mmcif_row(row) for row in rows]
        return cls(states=states)
