"""Unit tests for the altloc alphabet partition.

The partition is a small amount of arithmetic guarding a hard ceiling — 62
single-character labels, past which gemmi cannot store the result at all — so the
ordering, the disjointness, and the exhaustion message are tested here directly
rather than only through a merged file.
"""

from __future__ import annotations

import gemmi
import pytest

from pdbx_hierarchy import altloc
from pdbx_hierarchy.exceptions import PdbxValidationError


def _atoms(labels: list[str]) -> list[gemmi.Atom]:
    """Build atoms carrying the given altloc labels, blanks left unset as ``.``."""
    built: list[gemmi.Atom] = []
    for index, label in enumerate(labels):
        atom = gemmi.Atom()
        atom.name = f"C{index}"
        if label != altloc.BLANK_ALT_ID:
            atom.altloc = label
        built.append(atom)
    return built


class TestAlphabet:
    def test_the_alphabet_is_sixty_two_single_characters(self) -> None:
        assert len(altloc.ALPHABET) == 62
        assert all(len(label) == 1 for label in altloc.ALPHABET)

    def test_the_alphabet_runs_uppercase_then_lowercase_then_digits(self) -> None:
        assert altloc.ALPHABET[:26] == "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        assert altloc.ALPHABET[26:52] == "abcdefghijklmnopqrstuvwxyz"
        assert altloc.ALPHABET[52:] == "0123456789"


class TestReadAltloc:
    def test_an_unset_altloc_reads_as_blank(self) -> None:
        # gemmi stores "no altloc" as a NUL char, which is truthy in Python.
        atom = gemmi.Atom()
        assert altloc.read_altloc(atom) == altloc.BLANK_ALT_ID

    def test_a_set_altloc_reads_back(self) -> None:
        atom = gemmi.Atom()
        atom.altloc = "B"
        assert altloc.read_altloc(atom) == "B"


class TestOriginalLabels:
    def test_labels_are_distinct_and_sorted_with_the_blank_first(self) -> None:
        # Blank first because it is the label the bulk of a real model carries, so
        # a hierarchy-unaware viewer shows that bulk as the primary conformer.
        assert altloc.original_labels(_atoms(["B", ".", "A", "B", "."])) == [".", "A", "B"]

    def test_an_input_with_no_altlocs_at_all_still_needs_one_label(self) -> None:
        assert altloc.original_labels(_atoms([".", "."])) == ["."]

    def test_an_input_with_no_blanks_does_not_get_one(self) -> None:
        assert altloc.original_labels(_atoms(["A", "B"])) == ["A", "B"]


class TestPartition:
    def test_the_first_input_starts_at_the_beginning_of_the_alphabet(self) -> None:
        ground, changed = altloc.partition([[".", "A", "B"], [".", "A"]])
        assert ground == {".": "A", "A": "B", "B": "C"}
        assert changed == {".": "D", "A": "E"}

    def test_no_label_is_shared_between_inputs(self) -> None:
        mappings = altloc.partition([[".", "A"], [".", "A"]])
        assert set(mappings[0].values()).isdisjoint(mappings[1].values())

    def test_the_labels_widen_past_the_uppercase_letters(self) -> None:
        mappings = altloc.partition([list(altloc.ALPHABET[:26]), [".", "A"]])
        assert mappings[1] == {".": "a", "A": "b"}

    def test_the_last_two_labels_are_digits(self) -> None:
        mappings = altloc.partition([list(altloc.ALPHABET[:60]), [".", "A"]])
        assert mappings[1] == {".": "8", "A": "9"}

    def test_exhausting_the_alphabet_names_the_count_required(self) -> None:
        with pytest.raises(PdbxValidationError) as exc:
            altloc.partition([list(altloc.ALPHABET), [".", "A"]])
        assert "64" in str(exc.value)
        assert "62" in str(exc.value)

    def test_exactly_sixty_two_labels_fit(self) -> None:
        mappings = altloc.partition([list(altloc.ALPHABET[:61]), ["."]])
        assert mappings[1] == {".": "9"}


class TestRelabel:
    def test_every_atom_gets_its_mapped_label(self) -> None:
        atoms = _atoms([".", "A", "B", "."])
        altloc.relabel(atoms, {".": "A", "A": "B", "B": "C"})
        assert [atom.altloc for atom in atoms] == ["A", "B", "C", "A"]

    def test_a_new_label_does_not_cascade_onto_a_later_atom(self) -> None:
        # The mapping's keys and values overlap by construction — ground's blank
        # becomes "A" while an "A" of its own is also in the map — so relabelling
        # must read each atom's original label, not the one just written.
        atoms = _atoms(["A", "."])
        altloc.relabel(atoms, {".": "A", "A": "B"})
        assert [atom.altloc for atom in atoms] == ["B", "A"]
