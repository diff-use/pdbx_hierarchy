"""Partition the altloc alphabet between a merge's two inputs.

Every atom in a merged file carries a real ``label_alt_id``, blanks included, so
that a reader knowing nothing about the hierarchy extension sees a coherent
multiconformer model rather than thousands of duplicated atoms at the same
``(chain, residue, atom)`` key. That is what makes ``label_alt_id`` a resource
this command spends deliberately, and the supply is small.

The alphabet is ``A``-``Z``, then ``a``-``z``, then ``0``-``9``: 62
single-character labels. Single characters are a hard constraint rather than a
preference — ``gemmi.Atom.altloc`` is one ``char`` and raises on ``"AA"`` — so a
pair of inputs needing more than 62 labels between them is a validation error
naming the count required, not a silent wrap. Both halves of the extension past
``A``-``Z`` are expedients: lowercase labels collapse into their uppercase
counterparts under a case-insensitive reader, and digits are unconventional
enough that a reader treating altloc as a letter may mangle them. See
``.scratch/merge-states/findings.md``, finding 1, and the one-way widening
transformation it points at for the way back out.

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
"""

from __future__ import annotations

import string
from typing import TYPE_CHECKING

from pdbx_hierarchy.exceptions import PdbxValidationError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    import gemmi

#: The label a file uses for "this atom has no alternate conformation".
BLANK_ALT_ID = "."

#: The labels available, in assignment order. Uppercase first because it is the
#: only part of the range PDBx/mmCIF files use in practice.
ALPHABET = string.ascii_uppercase + string.ascii_lowercase + string.digits


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
