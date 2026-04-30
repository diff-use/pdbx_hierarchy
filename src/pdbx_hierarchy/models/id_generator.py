"""ID generation utility for hierarchy states.

Provides an iterator that generates IDs following the pattern:
A, B, ..., Z, AA, AB, ..., ZZ, AAA, AAB, ..., ZZZ, ...

This is a bijective base-26 numeral system where A=1, B=2, ..., Z=26.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from pdbx_hierarchy.models.hierarchy import HierarchyTree


def _index_to_id(index: int) -> str:
    """Convert a zero-based index to a hierarchy ID.

    Args:
        index: Zero-based index (0 returns "A", 25 returns "Z", 26 returns "AA").

    Returns:
        The generated ID string.

    Examples:
        >>> _index_to_id(0)
        'A'
        >>> _index_to_id(25)
        'Z'
        >>> _index_to_id(26)
        'AA'
        >>> _index_to_id(701)
        'ZZ'
        >>> _index_to_id(702)
        'AAA'
    """
    if index < 0:
        raise ValueError(f"Index must be non-negative, got {index}")

    result = []
    n = index + 1  # Convert to 1-based for bijective base-26

    while n > 0:
        n, remainder = divmod(n - 1, 26)
        result.append(chr(ord("A") + remainder))

    return "".join(reversed(result))


class HierarchyIdGenerator:
    """Generates hierarchy IDs, optionally skipping IDs already in a tree.

    Produces IDs in the sequence: A, B, ..., Z, AA, AB, ..., ZZ, AAA, ...
    When a tree is provided, skips any IDs that already exist in the tree.

    Attributes:
        tree: Optional HierarchyTree to check for existing IDs.

    Examples:
        >>> gen = HierarchyIdGenerator()
        >>> [next(gen) for _ in range(5)]
        ['A', 'B', 'C', 'D', 'E']

        >>> # With a tree that has A and C
        >>> gen = HierarchyIdGenerator(tree)
        >>> [next(gen) for _ in range(3)]
        ['B', 'D', 'E']
    """

    def __init__(self, tree: HierarchyTree | None = None) -> None:
        """Initialize the ID generator.

        Args:
            tree: Optional HierarchyTree. If provided, generated IDs will skip
                any that already exist in the tree.
        """
        self._index = 0
        self._tree = tree

    def __iter__(self) -> Self:
        """Return self as iterator."""
        return self

    def __next__(self) -> str:
        """Return the next available ID.

        Returns:
            The next ID in the sequence that is not present in the tree
            (if a tree was provided).

        Raises:
            StopIteration: Never raised; sequence is unlimited.
        """
        while True:
            candidate = _index_to_id(self._index)
            self._index += 1

            if self._tree is None or not self._tree.contains(candidate):
                return candidate

    def peek(self) -> str:
        """Return the next ID without advancing the generator.

        Returns:
            The next ID that would be returned by __next__.
        """
        saved_index = self._index
        result = next(self)
        self._index = saved_index
        return result

    def reset(self) -> None:
        """Reset the generator to start from 'A' again."""
        self._index = 0
