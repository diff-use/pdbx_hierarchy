"""Tests for StateCoexistence and CoexistenceTable models."""

import pytest

from pdbx_hierarchy.exceptions import InvalidCoexistenceError
from pdbx_hierarchy.models import (
    CoexistenceRule,
    CoexistenceTable,
    HierarchyState,
    HierarchyTree,
    StateCoexistence,
)


class TestCoexistenceRule:
    """Tests for CoexistenceRule enum."""

    def test_valid_values(self) -> None:
        """All expected rule values exist."""
        assert CoexistenceRule.AND.value == "AND"
        assert CoexistenceRule.OR.value == "OR"
        assert CoexistenceRule.NOT.value == "NOT"

    def test_from_string(self) -> None:
        """Rules can be created from string values."""
        assert CoexistenceRule("AND") == CoexistenceRule.AND
        assert CoexistenceRule("OR") == CoexistenceRule.OR
        assert CoexistenceRule("NOT") == CoexistenceRule.NOT

    def test_invalid_value(self) -> None:
        """Invalid string raises ValueError."""
        with pytest.raises(ValueError):
            CoexistenceRule("INVALID")


class TestStateCoexistence:
    """Tests for StateCoexistence model."""

    def test_valid_construction(self) -> None:
        """StateCoexistence can be constructed with valid data."""
        rule = StateCoexistence(
            id=1,
            rule=CoexistenceRule.AND,
            heterogeneity_id="A",
            heterogeneity_ids=["B", "C"],
        )
        assert rule.id == 1
        assert rule.rule == CoexistenceRule.AND
        assert rule.heterogeneity_id == "A"
        assert rule.heterogeneity_ids == ["B", "C"]
        assert rule.description is None

    def test_id_must_be_positive(self) -> None:
        """ID must be >= 1."""
        with pytest.raises(ValueError):
            StateCoexistence(
                id=0,
                rule=CoexistenceRule.OR,
                heterogeneity_id="A",
                heterogeneity_ids=["B"],
            )

        with pytest.raises(ValueError):
            StateCoexistence(
                id=-1,
                rule=CoexistenceRule.OR,
                heterogeneity_id="A",
                heterogeneity_ids=["B"],
            )

    def test_heterogeneity_ids_not_empty(self) -> None:
        """heterogeneity_ids must contain at least one ID."""
        with pytest.raises(ValueError, match="at least one"):
            StateCoexistence(
                id=1,
                rule=CoexistenceRule.OR,
                heterogeneity_id="A",
                heterogeneity_ids=[],
            )

    def test_to_mmcif_row(self) -> None:
        """to_mmcif_row converts to dict with mmCIF format."""
        rule = StateCoexistence(
            id=1,
            rule=CoexistenceRule.AND,
            heterogeneity_id="A",
            heterogeneity_ids=["B", "C"],
            description="Test rule",
        )
        row = rule.to_mmcif_row()

        assert row == {
            "id": "1",
            "rule": "AND",
            "heterogeneity_id": "A",
            "heterogeneity_ids": "B,C",
            "description": "Test rule",
        }

    def test_to_mmcif_row_none_description(self) -> None:
        """to_mmcif_row converts None description to '?'."""
        rule = StateCoexistence(
            id=1,
            rule=CoexistenceRule.OR,
            heterogeneity_id="A",
            heterogeneity_ids=["B"],
            description=None,
        )
        row = rule.to_mmcif_row()

        assert row["description"] == "?"

    def test_from_mmcif_row(self) -> None:
        """from_mmcif_row parses mmCIF dict correctly."""
        row = {
            "id": "1",
            "rule": "NOT",
            "heterogeneity_id": "A",
            "heterogeneity_ids": "B,C,D",
            "description": "Mutually exclusive",
        }
        rule = StateCoexistence.from_mmcif_row(row)

        assert rule.id == 1
        assert rule.rule == CoexistenceRule.NOT
        assert rule.heterogeneity_id == "A"
        assert rule.heterogeneity_ids == ["B", "C", "D"]
        assert rule.description == "Mutually exclusive"

    def test_from_mmcif_row_none_description(self) -> None:
        """from_mmcif_row converts '?' to None."""
        row = {
            "id": "1",
            "rule": "OR",
            "heterogeneity_id": "A",
            "heterogeneity_ids": "B",
            "description": "?",
        }
        rule = StateCoexistence.from_mmcif_row(row)

        assert rule.description is None

    def test_mmcif_roundtrip(self) -> None:
        """Rule survives to_mmcif_row/from_mmcif_row roundtrip."""
        original = StateCoexistence(
            id=1,
            rule=CoexistenceRule.AND,
            heterogeneity_id="A",
            heterogeneity_ids=["B", "C"],
            description="test",
        )
        row = original.to_mmcif_row()
        restored = StateCoexistence.from_mmcif_row(row)

        assert restored.id == original.id
        assert restored.rule == original.rule
        assert restored.heterogeneity_id == original.heterogeneity_id
        assert restored.heterogeneity_ids == original.heterogeneity_ids
        assert restored.description == original.description

    def test_mutable_fields(self) -> None:
        """Fields can be modified after construction."""
        rule = StateCoexistence(
            id=1,
            rule=CoexistenceRule.OR,
            heterogeneity_id="A",
            heterogeneity_ids=["B"],
        )
        rule.description = "Updated"
        assert rule.description == "Updated"

    def test_validation_on_assignment(self) -> None:
        """Validation runs when fields are assigned."""
        rule = StateCoexistence(
            id=1,
            rule=CoexistenceRule.OR,
            heterogeneity_id="A",
            heterogeneity_ids=["B"],
        )
        with pytest.raises(ValueError):
            rule.heterogeneity_ids = []


class TestCoexistenceTable:
    """Tests for CoexistenceTable model."""

    @pytest.fixture
    def simple_hierarchy(self) -> HierarchyTree:
        """A simple hierarchy for cross-validation tests."""
        return HierarchyTree(
            states=[
                HierarchyState(id="Base", name="base_state", parent=None),
                HierarchyState(id="A", name="state_a", parent="Base"),
                HierarchyState(id="B", name="state_b", parent="Base"),
                HierarchyState(id="C", name="state_c", parent="Base"),
            ]
        )

    @pytest.fixture
    def simple_table(self) -> CoexistenceTable:
        """A simple coexistence table with two rules."""
        return CoexistenceTable(
            rules=[
                StateCoexistence(
                    id=1,
                    rule=CoexistenceRule.AND,
                    heterogeneity_id="A",
                    heterogeneity_ids=["B"],
                ),
                StateCoexistence(
                    id=2,
                    rule=CoexistenceRule.NOT,
                    heterogeneity_id="A",
                    heterogeneity_ids=["C"],
                ),
            ]
        )

    def test_valid_construction(self, simple_table: CoexistenceTable) -> None:
        """CoexistenceTable can be constructed with valid rules."""
        assert len(simple_table) == 2

    def test_empty_table(self) -> None:
        """Empty CoexistenceTable is valid."""
        table = CoexistenceTable(rules=[])
        assert len(table) == 0

    def test_duplicate_rule_id_fails(self) -> None:
        """Table with duplicate rule IDs fails."""
        with pytest.raises(InvalidCoexistenceError, match="Duplicate"):
            CoexistenceTable(
                rules=[
                    StateCoexistence(
                        id=1,
                        rule=CoexistenceRule.OR,
                        heterogeneity_id="A",
                        heterogeneity_ids=["B"],
                    ),
                    StateCoexistence(
                        id=1,
                        rule=CoexistenceRule.AND,
                        heterogeneity_id="B",
                        heterogeneity_ids=["C"],
                    ),
                ]
            )

    def test_get_rule(self, simple_table: CoexistenceTable) -> None:
        """get_rule returns rule by ID."""
        rule = simple_table.get_rule(1)
        assert rule.id == 1
        assert rule.heterogeneity_id == "A"

    def test_get_rule_not_found(self, simple_table: CoexistenceTable) -> None:
        """get_rule raises KeyError for unknown ID."""
        with pytest.raises(KeyError, match="No coexistence rule"):
            simple_table.get_rule(99)

    def test_get_rules_for_state(self, simple_table: CoexistenceTable) -> None:
        """get_rules_for_state returns rules for a state."""
        rules = simple_table.get_rules_for_state("A")
        assert len(rules) == 2
        assert all(r.heterogeneity_id == "A" for r in rules)

    def test_get_rules_for_state_none(self, simple_table: CoexistenceTable) -> None:
        """get_rules_for_state returns empty list for unknown state."""
        rules = simple_table.get_rules_for_state("Z")
        assert rules == []

    def test_len(self, simple_table: CoexistenceTable) -> None:
        """len returns number of rules."""
        assert len(simple_table) == 2

    def test_iter_via_rules(self, simple_table: CoexistenceTable) -> None:
        """Can iterate via .rules attribute."""
        rules = list(simple_table.rules)
        assert len(rules) == 2
        assert all(isinstance(r, StateCoexistence) for r in rules)

    def test_add_rule(self, simple_table: CoexistenceTable) -> None:
        """add_rule adds a new rule and validates."""
        new_rule = StateCoexistence(
            id=3,
            rule=CoexistenceRule.OR,
            heterogeneity_id="B",
            heterogeneity_ids=["C"],
        )
        simple_table.add_rule(new_rule)

        assert len(simple_table) == 3
        assert simple_table.get_rule(3).heterogeneity_id == "B"

    def test_add_rule_duplicate_id(self, simple_table: CoexistenceTable) -> None:
        """add_rule fails if ID already exists."""
        with pytest.raises(InvalidCoexistenceError, match="Duplicate"):
            simple_table.add_rule(
                StateCoexistence(
                    id=1,
                    rule=CoexistenceRule.OR,
                    heterogeneity_id="X",
                    heterogeneity_ids=["Y"],
                )
            )

    def test_remove_rule(self, simple_table: CoexistenceTable) -> None:
        """remove_rule removes a rule."""
        removed = simple_table.remove_rule(1)

        assert removed.id == 1
        assert len(simple_table) == 1
        with pytest.raises(KeyError):
            simple_table.get_rule(1)

    def test_remove_rule_not_found(self, simple_table: CoexistenceTable) -> None:
        """remove_rule raises KeyError for unknown ID."""
        with pytest.raises(KeyError):
            simple_table.remove_rule(99)

    def test_validate_against_hierarchy_valid(
        self, simple_table: CoexistenceTable, simple_hierarchy: HierarchyTree
    ) -> None:
        """validate_against_hierarchy passes for valid references."""
        errors = simple_table.validate_against_hierarchy(simple_hierarchy, raise_on_error=False)
        assert errors == []

    def test_validate_against_hierarchy_invalid_heterogeneity_id(self, simple_hierarchy: HierarchyTree) -> None:
        """validate_against_hierarchy catches invalid heterogeneity_id."""
        table = CoexistenceTable(
            rules=[
                StateCoexistence(
                    id=1,
                    rule=CoexistenceRule.OR,
                    heterogeneity_id="NonExistent",
                    heterogeneity_ids=["A"],
                )
            ]
        )

        errors = table.validate_against_hierarchy(simple_hierarchy, raise_on_error=False)
        assert len(errors) == 1
        assert "heterogeneity_id" in errors[0]
        assert "NonExistent" in errors[0]

    def test_validate_against_hierarchy_invalid_heterogeneity_ids(self, simple_hierarchy: HierarchyTree) -> None:
        """validate_against_hierarchy catches invalid heterogeneity_ids."""
        table = CoexistenceTable(
            rules=[
                StateCoexistence(
                    id=1,
                    rule=CoexistenceRule.OR,
                    heterogeneity_id="A",
                    heterogeneity_ids=["B", "NonExistent"],
                )
            ]
        )

        errors = table.validate_against_hierarchy(simple_hierarchy, raise_on_error=False)
        assert len(errors) == 1
        assert "heterogeneity_ids" in errors[0]
        assert "NonExistent" in errors[0]

    def test_validate_against_hierarchy_raises(self, simple_hierarchy: HierarchyTree) -> None:
        """validate_against_hierarchy raises when raise_on_error=True."""
        table = CoexistenceTable(
            rules=[
                StateCoexistence(
                    id=1,
                    rule=CoexistenceRule.OR,
                    heterogeneity_id="NonExistent",
                    heterogeneity_ids=["A"],
                )
            ]
        )

        with pytest.raises(InvalidCoexistenceError, match="NonExistent"):
            table.validate_against_hierarchy(simple_hierarchy, raise_on_error=True)

    def test_to_mmcif_rows(self, simple_table: CoexistenceTable) -> None:
        """to_mmcif_rows converts all rules to dicts."""
        rows = simple_table.to_mmcif_rows()

        assert len(rows) == 2
        ids = [r["id"] for r in rows]
        assert set(ids) == {"1", "2"}

    def test_from_mmcif_rows(self) -> None:
        """from_mmcif_rows creates table from dicts."""
        rows = [
            {
                "id": "1",
                "rule": "AND",
                "heterogeneity_id": "A",
                "heterogeneity_ids": "B,C",
                "description": "?",
            },
            {
                "id": "2",
                "rule": "NOT",
                "heterogeneity_id": "B",
                "heterogeneity_ids": "C",
                "description": "?",
            },
        ]
        table = CoexistenceTable.from_mmcif_rows(rows)

        assert len(table) == 2
        assert table.get_rule(1).rule == CoexistenceRule.AND

    def test_mmcif_roundtrip(self, simple_table: CoexistenceTable) -> None:
        """Table survives mmCIF roundtrip."""
        rows = simple_table.to_mmcif_rows()
        restored = CoexistenceTable.from_mmcif_rows(rows)

        assert len(restored) == len(simple_table)
        for rule in simple_table.rules:
            restored_rule = restored.get_rule(rule.id)
            assert restored_rule.rule == rule.rule
            assert restored_rule.heterogeneity_id == rule.heterogeneity_id
            assert restored_rule.heterogeneity_ids == rule.heterogeneity_ids
