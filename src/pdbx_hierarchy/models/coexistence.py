"""Data models for the _pdbx_state_coexistence table.

This module provides Pydantic models representing coexistence rules
between heterogeneity states in a PDBx/mmCIF file.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, PrivateAttr, field_validator

from pdbx_hierarchy.exceptions import InvalidCoexistenceError

if TYPE_CHECKING:
    from pdbx_hierarchy.models.hierarchy import HierarchyTree


class CoexistenceRule(str, Enum):
    """Valid coexistence rule types.

    Attributes:
        AND: All referenced states must coexist.
        OR: At least one referenced state must be present.
        NOT: The referenced states are mutually exclusive with the source.
    """

    AND = "AND"
    OR = "OR"
    NOT = "NOT"


class StateCoexistence(BaseModel):
    """A single row in the _pdbx_state_coexistence table.

    Defines a coexistence relationship between structural states.

    Attributes:
        id: Unique integer identifier for this rule (must be >= 1).
        rule: The coexistence rule type (AND, OR, NOT).
        heterogeneity_id: Source state ID this rule applies to.
        heterogeneity_ids: List of related state IDs.
        description: Optional description of this rule.

    Examples:
        >>> rule = StateCoexistence(id=1, rule=CoexistenceRule.AND, heterogeneity_id="A", heterogeneity_ids=["B", "C"])
        >>> rule.to_mmcif_row()
        {'id': '1', 'rule': 'AND', 'heterogeneity_id': 'A',
         'heterogeneity_ids': 'B,C', 'description': '?'}
    """

    model_config = {"validate_assignment": True}

    id: int = Field(..., ge=1, description="Unique integer identifier (>= 1)")
    rule: CoexistenceRule = Field(..., description="Coexistence rule type")
    heterogeneity_id: str = Field(..., description="Source state ID")
    heterogeneity_ids: list[str] = Field(..., description="Related state IDs")
    description: str | None = Field(None, description="Optional description")

    @field_validator("heterogeneity_ids")
    @classmethod
    def validate_heterogeneity_ids_not_empty(cls, v: list[str]) -> list[str]:
        """Validate that heterogeneity_ids is not empty."""
        if not v:
            raise ValueError("heterogeneity_ids must contain at least one state ID")
        return v

    def to_mmcif_row(self) -> dict[str, str]:
        """Convert to a dictionary suitable for mmCIF output.

        Returns:
            Dictionary with mmCIF-formatted values.
        """
        return {
            "id": str(self.id),
            "rule": self.rule.value,
            "heterogeneity_id": self.heterogeneity_id,
            "heterogeneity_ids": ",".join(self.heterogeneity_ids),
            "description": self.description if self.description is not None else "?",
        }

    @classmethod
    def from_mmcif_row(cls, row: dict[str, str]) -> StateCoexistence:
        """Create a StateCoexistence from an mmCIF row dictionary.

        Args:
            row: Dictionary with keys 'id', 'rule', 'heterogeneity_id',
                 'heterogeneity_ids', 'description'.

        Returns:
            A new StateCoexistence instance.
        """
        heterogeneity_ids_str = row.get("heterogeneity_ids", "")
        heterogeneity_ids = [s.strip() for s in heterogeneity_ids_str.split(",") if s.strip()]

        return cls(
            id=int(row["id"]),
            rule=CoexistenceRule(row["rule"]),
            heterogeneity_id=row["heterogeneity_id"],
            heterogeneity_ids=heterogeneity_ids,
            description=None if row.get("description") in ("?", None) else row.get("description"),
        )


class CoexistenceTable(BaseModel):
    """Collection of StateCoexistence rules.

    Stores coexistence rules independently from the hierarchy. Cross-validation
    against a HierarchyTree is available via validate_against_hierarchy().

    Attributes:
        rules: List of StateCoexistence objects.

    Examples:
        >>> table = CoexistenceTable(
        ...     rules=[StateCoexistence(id=1, rule=CoexistenceRule.AND, heterogeneity_id="A", heterogeneity_ids=["B"])]
        ... )
        >>> errors = table.validate_against_hierarchy(tree, raise_on_error=False)
    """

    model_config = {"validate_assignment": True}

    rules: list[StateCoexistence] = Field(default_factory=list)

    # Internal lookup by ID
    _by_id: dict[int, StateCoexistence] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """Validate unique IDs and build lookup table after init."""
        self._validate_and_rebuild_index()

    def __setattr__(self, name: str, value: Any) -> None:
        """Override to rebuild index when rules is assigned."""
        super().__setattr__(name, value)
        if name == "rules":
            self._validate_and_rebuild_index()

    def _validate_and_rebuild_index(self) -> None:
        """Validate rule IDs are unique and rebuild lookup table.

        Raises:
            InvalidCoexistenceError: If duplicate rule IDs exist.
        """
        self._by_id = {}
        for rule in self.rules:
            if rule.id in self._by_id:
                raise InvalidCoexistenceError(f"Duplicate coexistence rule ID: {rule.id}")
            self._by_id[rule.id] = rule

    # === Query Methods ===

    def get_rule(self, rule_id: int) -> StateCoexistence:
        """Get a rule by its ID.

        Args:
            rule_id: The ID of the rule to retrieve.

        Returns:
            The StateCoexistence with the given ID.

        Raises:
            KeyError: If no rule with that ID exists.
        """
        if rule_id not in self._by_id:
            raise KeyError(f"No coexistence rule with ID {rule_id}")
        return self._by_id[rule_id]

    def get_rules_for_state(self, state_id: str) -> list[StateCoexistence]:
        """Get all rules where the given state is the source.

        Args:
            state_id: The heterogeneity_id to search for.

        Returns:
            List of StateCoexistence rules with matching heterogeneity_id.
        """
        return [r for r in self.rules if r.heterogeneity_id == state_id]

    def __len__(self) -> int:
        """Return the number of rules in the table."""
        return len(self.rules)

    # === Modification Methods ===

    def add_rule(self, rule: StateCoexistence) -> None:
        """Add a new rule to the table.

        Args:
            rule: The StateCoexistence to add.

        Raises:
            InvalidCoexistenceError: If a rule with this ID already exists.
        """
        new_rules = list(self.rules) + [rule]
        self.rules = new_rules

    def remove_rule(self, rule_id: int) -> StateCoexistence:
        """Remove a rule from the table.

        Args:
            rule_id: The ID of the rule to remove.

        Returns:
            The removed StateCoexistence.

        Raises:
            KeyError: If no rule with that ID exists.
        """
        if rule_id not in self._by_id:
            raise KeyError(f"No coexistence rule with ID {rule_id}")

        removed = self._by_id[rule_id]
        new_rules = [r for r in self.rules if r.id != rule_id]
        self.rules = new_rules
        return removed

    # === Cross-Validation ===

    def validate_against_hierarchy(self, hierarchy: HierarchyTree, *, raise_on_error: bool = True) -> list[str]:
        """Validate that all state references exist in the hierarchy.

        Args:
            hierarchy: The HierarchyTree to validate against.
            raise_on_error: If True, raise on first error. If False,
                collect and return all errors.

        Returns:
            List of error messages (empty if valid).

        Raises:
            InvalidCoexistenceError: If raise_on_error is True and
                validation fails.
        """
        errors: list[str] = []

        for rule in self.rules:
            # Check heterogeneity_id exists
            if not hierarchy.contains(rule.heterogeneity_id):
                msg = f"Rule {rule.id}: heterogeneity_id {rule.heterogeneity_id!r} not in hierarchy"
                if raise_on_error:
                    raise InvalidCoexistenceError(msg)
                errors.append(msg)

            # Check all heterogeneity_ids exist
            for ref_id in rule.heterogeneity_ids:
                if not hierarchy.contains(ref_id):
                    msg = f"Rule {rule.id}: heterogeneity_ids contains {ref_id!r} not in hierarchy"
                    if raise_on_error:
                        raise InvalidCoexistenceError(msg)
                    errors.append(msg)

        return errors

    # === Serialization ===

    def to_mmcif_rows(self) -> list[dict[str, str]]:
        """Convert all rules to mmCIF row dictionaries.

        Returns:
            List of dictionaries suitable for writing to mmCIF.
        """
        return [rule.to_mmcif_row() for rule in self.rules]

    @classmethod
    def from_mmcif_rows(cls, rows: list[dict[str, str]]) -> CoexistenceTable:
        """Create a CoexistenceTable from mmCIF row dictionaries.

        Args:
            rows: List of dictionaries from mmCIF parsing.

        Returns:
            A CoexistenceTable instance.

        Raises:
            InvalidCoexistenceError: If rule IDs are not unique.
        """
        rules = [StateCoexistence.from_mmcif_row(row) for row in rows]
        return cls(rules=rules)
