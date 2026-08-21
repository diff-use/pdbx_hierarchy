"""Spend the altloc alphabet: partition it between a merge's two inputs, or leave it.

Both things this module does are about the same scarce resource, and change for
the same reason — the partition hands out the 62 single-character labels, and the
widening is the exit for a file those labels cannot serve.

Every atom in a merged file carries a real ``label_alt_id``, blanks included, so
that a reader knowing nothing about the hierarchy extension sees a coherent
multiconformer model rather than thousands of duplicated atoms at the same
``(chain, residue, atom)`` key. That is what makes ``label_alt_id`` a resource
this command spends deliberately, and the supply is small.

The alphabet is ``A``-``Z``, then ``a``-``z``, then ``0``-``9``: 62
single-character labels. Single characters are a hard constraint rather than a
preference — ``gemmi.Atom.altloc`` is one ``char`` and raises on ``"AA"`` — so a
pair of inputs needing more than 62 labels between them is a validation error
naming the count required, not a silent wrap.

Both halves of the extension past ``A``-``Z`` are expedients, and knowingly so.
``A``-``Z`` is the range PDBx/mmCIF files use in practice; lowercase labels
collapse into their uppercase counterparts under a reader that folds case,
silently fusing two unrelated conformations under one label, and digits are
unconventional enough that a reader treating altloc as a letter may mangle them.
Three things cannot all hold at once — a single-character ``label_alt_id``, more
than 26 labels at one position, and safety under a case-insensitive reader — and
this module keeps the first two. A file that needs the third has to leave the
extended alphabet altogether, by widening every label to the multi-character
sequence ``A``...``Z``, ``AA``, ``AB``, ...; that is a one-way exit, because
gemmi cannot write a multi-character altloc back.

Each input gets a contiguous run of the alphabet — ground from the start, changed
continuing above it — assigned to that input's **original** labels. Two decisions
in that sentence carry weight.

*The unit is a label, not an atom.* One entry per distinct original label per
input is what keeps the mapping small enough to print in full and to trace a
merged atom back through, and it preserves each input's own conformer structure:
two atoms that shared a letter still share one, and two that did not still do
not.

*The blank sorts first.* Roughly 97% of the atoms in a typical model have no
altloc, so the blank's label is the one the bulk of the file ends up carrying;
putting it first means that bulk reads as the primary conformer.

The partition never touches hierarchy state ids, which are a separate namespace
that happens to draw on overlapping characters, and it must run *after* state
assignment: inference reads ``label_alt_id``, so relabelling first would read
almost every atom in a typical model as a conformer.

:func:`widen_file` is that exit — the one function of it a caller needs — for a
user whose downstream reader folds case.
It is the other end of the same trade — it gives up gemmi as a write path and any
route back into the extended alphabet, and buys labels that survive a
case-insensitive comparison. It is not on the CLI for exactly that reason: a
command whose output gemmi cannot read back is a different kind of thing from the
commands around it.
"""

from __future__ import annotations

import string
from typing import TYPE_CHECKING

import gemmi

from pdbx_hierarchy.exceptions import PdbxValidationError
from pdbx_hierarchy.io.reader import UNSET_VALUES, read_mmcif
from pdbx_hierarchy.io.writer import write_mmcif
from pdbx_hierarchy.models.id_generator import HierarchyIdGenerator

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

#: The label a file uses for "this atom has no alternate conformation".
BLANK_ALT_ID = "."

#: The labels available, in assignment order. Uppercase first because it is the
#: only part of the range PDBx/mmCIF files use in practice.
ALPHABET = string.ascii_uppercase + string.ascii_lowercase + string.digits

#: Where each label sits in the alphabet, for ordering. A dict rather than
#: ``ALPHABET.find``, which is a substring search: it answers 0 for ``"AB"``,
#: sorting a label that is already widened as though it were ``A``.
_ALPHABET_POSITION = {label: position for position, label in enumerate(ALPHABET)}

#: Printed alongside a widening's mapping. The two namespaces overlap in the
#: characters they draw on and in the sequence they draw them in, which is
#: exactly why saying so is worth the line: a reader looking at a widened file's
#: ``AA`` has no way to tell from the file alone that the ``AA`` in its hierarchy
#: table is unrelated.
WIDENING_NAMESPACE_NOTE = (
    "hierarchy state ids are a separate namespace and are untouched. "
    "Where a widened label and a state id happen to read the same, they still name unrelated things"
)


def read_altloc(atom: gemmi.Atom) -> str:
    """Return an atom's altloc label as a file would write it.

    Args:
        atom: The atom to read.

    Returns:
        The label, or ``.`` for an atom with no altloc. gemmi stores "no altloc"
        as a NUL char, which is truthy in Python and would otherwise reach a file
        as an unprintable label.
    """
    return atom.altloc if atom.altloc not in ("", "\x00") else BLANK_ALT_ID


def original_labels(atoms: Iterable[gemmi.Atom]) -> list[str]:
    """Return one input's distinct altloc labels, in the order they get relabelled.

    Args:
        atoms: The input's atoms. Unlike the occupancy fold, nothing here depends
            on which chain or residue an atom sits in: a label means the same
            conformation everywhere in a file.

    Returns:
        The distinct labels, blank first and the rest sorted, so that the mapping
        an input receives depends only on the set of labels it uses and not on
        which atom happened to come first.
    """
    labels = {read_altloc(atom) for atom in atoms}
    blank = [BLANK_ALT_ID] if BLANK_ALT_ID in labels else []
    return blank + sorted(labels - {BLANK_ALT_ID})


def partition(label_sets: Sequence[Sequence[str]]) -> list[dict[str, str]]:
    """Give each input a contiguous run of the alphabet for its labels.

    Args:
        label_sets: Each input's distinct original labels, in output order — the
            first input's run starts at the beginning of the alphabet and each
            later one continues above the previous.

    Returns:
        One old -> new mapping per input, in the same order. No label appears in
        two mappings' values.

    Raises:
        PdbxValidationError: If the inputs need more labels between them than the
            alphabet has. Raised before any input is relabelled and counted over
            all of them, so the message names what the whole merge would require
            rather than how far it got.
    """
    required = sum(len(labels) for labels in label_sets)
    if required > len(ALPHABET):
        raise PdbxValidationError(
            f"the merged file needs {required} altloc labels but only {len(ALPHABET)} are available: "
            f"label_alt_id is a single character (A-Z, a-z, 0-9)"
        )

    mappings: list[dict[str, str]] = []
    start = 0
    for labels in label_sets:
        mappings.append(dict(zip(labels, ALPHABET[start : start + len(labels)], strict=True)))
        start += len(labels)
    return mappings


def relabel(atoms: Iterable[gemmi.Atom], mapping: Mapping[str, str]) -> None:
    """Write an input's new altloc labels onto its atoms, in place.

    Each atom's original label is read once and replaced once, so a new label
    equal to some other atom's original label cannot cascade onward — which it
    otherwise would, since the map's keys and values overlap by construction.

    Args:
        atoms: The input's atoms.
        mapping: Old -> new label, covering every label the atoms carry — the
            mapping :func:`partition` returned for this input.

    Raises:
        KeyError: If an atom carries a label the mapping does not cover, which
            means the mapping was built from a different set of atoms than this.
    """
    for atom in atoms:
        atom.altloc = mapping[read_altloc(atom)]


# === Widening: the one-way exit from the extended alphabet ===


def _widen_labels(labels: Iterable[str]) -> dict[str, str]:
    """Map single-character labels onto labels unique under a case fold.

    The new labels are ``A``, ``B``, ..., ``Z``, ``AA``, ``AB``, ... — the
    sequence :class:`HierarchyIdGenerator` already produces, reused rather than
    reimplemented. Being all uppercase, they are as distinct to a reader that
    folds case as to one that does not.

    The whole set is remapped, not only the labels past ``Z``. Widening ``a`` and
    leaving ``A`` alone would collide the two at whichever label ``a`` landed on
    if that label were ``A``; more to the point, a file where some labels are one
    character and others are two is harder to reason about than one where the rule
    is uniform.

    Args:
        labels: The labels a file uses. Duplicates and the unset markers ``.`` and
            ``?`` are ignored; those two mean the atom has no alternate
            conformation, which is not a label and cannot collide with one.

    Returns:
        Old -> new label, one entry per distinct label. The old labels are
        considered in alphabet order, so a file whose labels are a contiguous run
        from ``A`` maps each onto itself — the common case, a merged file, where
        the widening is visibly a no-op. A file skipping labels still compacts
        (``B``, ``D`` -> ``A``, ``B``), and a label outside the alphabet is
        considered after every label inside it, sorted, rather than being left
        alone.
    """
    distinct = {label for label in labels if label not in UNSET_VALUES}
    ordered = sorted(distinct, key=_widening_order)
    # The generator is unbounded, which is the whole point of widening, so the
    # labels are what runs out first and strict=True would reject every call.
    return dict(zip(ordered, HierarchyIdGenerator(), strict=False))


def _widening_order(label: str) -> tuple[int, int, str]:
    """Return the sort key placing a label in the order it gets widened.

    Args:
        label: One of the file's labels.

    Returns:
        Its alphabet position, or a key sorting it after every label in the
        alphabet and among its own kind by value.
    """
    if label in _ALPHABET_POSITION:
        return 0, _ALPHABET_POSITION[label], label
    return 1, 0, label


def _widen_block(block: gemmi.cif.Block) -> dict[str, str]:
    """Widen a block's ``label_alt_id`` column in place.

    ``label_alt_id`` is the only column touched. Nothing else in the table set a
    merged file carries references an altloc — ``_struct_conn``, which would, is
    among the categories the merge drops.

    This works on the raw mmCIF column rather than on a
    :class:`gemmi.Structure` because it has to: ``gemmi.Atom.altloc`` is a single
    ``char`` and raises on ``"AA"``, so a widened file cannot be held in gemmi's
    atom model at all. It can still be *parsed* by gemmi at the cif level, which
    is what this reads and writes.

    Args:
        block: The data block to widen, mutated in place.

    Returns:
        Old -> new label, for the caller to print.

    Raises:
        PdbxValidationError: If the block has no ``_atom_site.label_alt_id``
            column, leaving nothing to widen.
    """
    # Presence is asked of the loop item rather than the column, because a column
    # is falsy both when the tag is absent and when the loop holds no rows, and
    # only the first of those is worth an error.
    if block.find_loop_item("_atom_site.label_alt_id") is None:
        raise PdbxValidationError(
            "no _atom_site.label_alt_id column to widen; widening is for a file whose atoms already carry labels"
        )

    column = block.find_loop("_atom_site.label_alt_id")
    mapping = _widen_labels(column)
    for index, label in enumerate(list(column)):
        if label in UNSET_VALUES:
            continue
        column[index] = mapping[label]
    return mapping


def widen_file(source: Path, output: Path) -> dict[str, str]:
    """Write ``source`` out to ``output`` with its altloc labels widened.

    One-way by design. The output leaves the single-character alphabet for good:
    nothing here reverses the mapping, and no library reading the result back into
    a gemmi ``Structure`` can, since the labels no longer fit an atom's ``altloc``
    field.

    Args:
        source: The file to widen. Read, never written.
        output: Where the widened file goes. A separate path rather than an
            in-place rewrite, so the file that still round-trips through gemmi
            survives the conversion.

    Returns:
        Old -> new label, for the caller to print.

    Raises:
        FileNotFoundError: If ``source`` does not exist.
        PdbxParseError: If ``source`` is malformed or holds more than one block.
        PdbxValidationError: If ``source`` has no ``_atom_site.label_alt_id``
            column.
    """
    block = read_mmcif(source)
    mapping = _widen_block(block)
    doc = gemmi.cif.Document()
    doc.add_copied_block(block)
    write_mmcif(doc, output)
    return mapping
