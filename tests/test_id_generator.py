"""Tests for the HierarchyIdGenerator class."""

from pdbx_hierarchy.models import HierarchyIdGenerator, HierarchyState, HierarchyTree


class TestHierarchyIdGenerator:
    """Tests for HierarchyIdGenerator."""

    def test_generates_single_letters(self) -> None:
        """Generator produces A-Z for first 26 IDs."""
        gen = HierarchyIdGenerator()
        ids = [next(gen) for _ in range(26)]

        assert ids[0] == "A"
        assert ids[25] == "Z"
        assert ids == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def test_generates_double_letters(self) -> None:
        """Generator produces AA-ZZ after Z."""
        gen = HierarchyIdGenerator()
        # Skip first 26 (A-Z)
        for _ in range(26):
            next(gen)

        # Next should be AA
        assert next(gen) == "AA"
        assert next(gen) == "AB"

        # Skip to AZ (index 51 = 26 + 25)
        for _ in range(23):  # Already at AC (index 28), need to get to AZ (index 51)
            next(gen)
        assert next(gen) == "AZ"
        assert next(gen) == "BA"

    def test_transition_z_to_aa(self) -> None:
        """Z is followed by AA."""
        gen = HierarchyIdGenerator()
        for _ in range(25):  # Get to Z
            next(gen)
        assert next(gen) == "Z"
        assert next(gen) == "AA"

    def test_transition_zz_to_aaa(self) -> None:
        """ZZ is followed by AAA."""
        gen = HierarchyIdGenerator()
        # Skip to ZZ: 26 (A-Z) + 676 (AA-ZZ) - 1 = 701
        for _ in range(701):
            next(gen)
        assert next(gen) == "ZZ"
        assert next(gen) == "AAA"

    def test_generates_triple_letters(self) -> None:
        """Generator produces AAA onwards after ZZ."""
        gen = HierarchyIdGenerator()
        # Index 702 should be AAA
        for _ in range(702):
            next(gen)
        assert next(gen) == "AAA"
        assert next(gen) == "AAB"

    def test_skips_existing_ids_in_tree(self) -> None:
        """Generator skips IDs that exist in the provided tree."""
        # Create a tree with Base, A, and C
        tree = HierarchyTree(
            states=[
                HierarchyState(id="Base", name="base", parent=None),
                HierarchyState(id="A", name="state_a", parent="Base"),
                HierarchyState(id="C", name="state_c", parent="Base"),
            ]
        )

        gen = HierarchyIdGenerator(tree)
        # Should skip A and C
        assert next(gen) == "B"
        assert next(gen) == "D"
        assert next(gen) == "E"

    def test_skips_base_if_in_tree(self) -> None:
        """Generator does not produce 'Base' - it's not in the A-Z sequence."""
        gen = HierarchyIdGenerator()
        # Generate many IDs - none should be "Base"
        ids = [next(gen) for _ in range(100)]
        assert "Base" not in ids

    def test_works_without_tree(self) -> None:
        """Generator works without a tree (no skipping)."""
        gen = HierarchyIdGenerator()
        assert next(gen) == "A"
        assert next(gen) == "B"

    def test_peek_does_not_advance(self) -> None:
        """Peek returns next ID without advancing."""
        gen = HierarchyIdGenerator()
        assert gen.peek() == "A"
        assert gen.peek() == "A"
        assert next(gen) == "A"
        assert gen.peek() == "B"

    def test_reset(self) -> None:
        """Reset returns generator to start."""
        gen = HierarchyIdGenerator()
        next(gen)
        next(gen)
        next(gen)
        gen.reset()
        assert next(gen) == "A"

    def test_iterator_protocol(self) -> None:
        """Generator works with iter() and for loops."""
        gen = HierarchyIdGenerator()
        assert iter(gen) is gen

        # Can use in for loop with break
        ids = []
        for id_ in gen:
            ids.append(id_)
            if len(ids) >= 3:
                break
        assert ids == ["A", "B", "C"]
