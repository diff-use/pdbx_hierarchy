"""Separate the two inputs' non-polymer residues by authored residue number.

Merging pools both inputs' atoms into shared residues keyed by chain and authored
residue number, so what a number means has to be settled before assembly. The two
kinds of residue answer differently.

*Polymer residues correspond.* Residue 47 is the same residue in both models of
the same crystal, and that correspondence is exactly what the merged file's
hierarchy encodes: one residue carrying a ground conformation and a changed one.
Polymer numbering is therefore left as it was, in **both** inputs.

*Non-polymers do not correspond.* The same water number in two independently
refined models is two unrelated waters, and the same number can hold a
cryoprotectant in one input and a water in the other. Preserving those numbers
would assert a correspondence that does not exist — fusing two unrelated
molecules into one residue with two "conformations" — so the changed input's
non-polymer numbers are shifted above the ground input's highest.

The asymmetry is deliberate. Ground is usually the source of truth and stays
recognisable, matching the ground-first ordering used for state ids and altloc
labels. Its cost is real and is recorded in the effort's findings: the changed
state's ligand no longer sits at its deposited residue number, and the printed
offset is the only trace back to it — it lives in a terminal, because the merged
file's table set has nowhere to record it.

Three details of the shift carry weight.

*One offset, over every chain.* A per-chain offset would renumber two chains'
waters by different amounts, and then no single number printed to the user would
describe what happened to the file.

*The offset is the smallest one that clears the ceiling*, so the merged file's
numbers stay as close to the inputs' as the guarantee allows. The ceiling counts
every number ground uses, non-polymers included, and every number changed's own
polymer residues use: a shifted water must clear the other input's residues and
its own input's.

*Classification comes from ``entity_type``, not from ``HETATM``.* A modified
polymer residue such as ``MSE`` is a ``HETATM`` row and a polymer residue, and
shifting it would tear it out of its chain. gemmi populates ``entity_type`` from
the input's own ``_entity`` family, or infers it from the coordinates when the
input has none. A residue it cannot classify is treated as polymer and left
alone: leaving a collision is loud, because :func:`reject_shared_numbers` reports
it, where shifting a polymer residue would be silent.

Every function here takes an input's residues as a sequence rather than an
iterable, because one walk is read more than once: planned over, written to, and
then checked.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import gemmi

from pdbx_hierarchy.exceptions import PdbxValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence

#: The entity types that do not correspond between two models of one crystal.
#: Branched sits here with the ligands and waters: an oligosaccharide is matched
#: between two depositions no more reliably than a cryoprotectant is.
NON_POLYMER_ENTITY_TYPES = frozenset(
    {
        gemmi.EntityType.NonPolymer,
        gemmi.EntityType.Water,
        gemmi.EntityType.Branched,
    }
)


class NonPolymerShift(NamedTuple):
    """The one offset applied to the changed input's non-polymer residues.

    Attributes:
        offset: How much every shifted number moved up; ``0`` when the changed
            input's non-polymers already clear the ceiling, or has none.
        residue_count: How many residues the offset applies to.
        old_min: The lowest non-polymer number before the shift, or None when
            there is nothing to shift.
        old_max: The highest non-polymer number before the shift, or None.
    """

    offset: int
    residue_count: int
    old_min: int | None
    old_max: int | None

    @property
    def new_min(self) -> int | None:
        """The lowest non-polymer number after the shift, or None."""
        return None if self.old_min is None else self.old_min + self.offset

    @property
    def new_max(self) -> int | None:
        """The highest non-polymer number after the shift, or None."""
        return None if self.old_max is None else self.old_max + self.offset


def is_non_polymer(residue: gemmi.Residue) -> bool:
    """Return whether a residue is one whose number means nothing across inputs.

    Args:
        residue: The residue to classify. Its ``entity_type`` must already be
            populated — gemmi reads it from the file's ``_entity`` family, or
            fills it in from the coordinates via ``add_entity_types``.

    Returns:
        True for a ligand, a water, or a branched entity; False for a polymer
        residue and for a residue gemmi could not classify. See the module
        docstring for why the unclassified case falls on this side.
    """
    return residue.entity_type in NON_POLYMER_ENTITY_TYPES


def separate_non_polymers(
    ground: Sequence[tuple[gemmi.Chain, gemmi.Residue]],
    changed: Sequence[tuple[gemmi.Chain, gemmi.Residue]],
    *,
    ground_label: str,
    changed_label: str,
) -> NonPolymerShift:
    """Shift the changed input's non-polymers clear of the ground input's residues.

    The three steps belong together: a shift planned and not applied leaves the
    collision it was computed to remove, and a shift applied and not checked
    leaves the caller believing an invariant that the asymmetric rule cannot
    always deliver on its own.

    Args:
        ground: A walk over the ground input's residues, left untouched.
        changed: A walk over the changed input's residues; its non-polymers are
            renumbered in place.
        ground_label: What to call the ground input in an error message.
        changed_label: What to call the changed input in an error message.

    Returns:
        The shift applied, for the caller to print.

    Raises:
        PdbxValidationError: If the shifted numbering still has a non-polymer
            residue sharing an authored position with a residue of the other
            input.
    """
    shift = plan_shift(ground, changed)
    apply_shift(changed, shift)
    reject_shared_numbers(ground, changed, ground_label=ground_label, changed_label=changed_label)
    return shift


def plan_shift(
    ground: Sequence[tuple[gemmi.Chain, gemmi.Residue]],
    changed: Sequence[tuple[gemmi.Chain, gemmi.Residue]],
) -> NonPolymerShift:
    """Work out the offset the changed input's non-polymer residues need.

    Args:
        ground: A walk over the ground input's residues.
        changed: A walk over the changed input's residues.

    Returns:
        The offset and the range it applies to. The offset is the smallest one
        putting every changed non-polymer number above every number ground uses
        and above every number changed's own polymer residues use — zero when
        they already are, or when there is nothing to shift.
    """
    shifting = [number for number in _classified_numbers(changed) if number.non_polymer]
    if not shifting:
        return NonPolymerShift(offset=0, residue_count=0, old_min=None, old_max=None)

    ceiling = [number.value for number in _classified_numbers(ground)]
    ceiling.extend(number.value for number in _classified_numbers(changed) if not number.non_polymer)
    old_min = min(number.value for number in shifting)
    old_max = max(number.value for number in shifting)

    # A negative offset would pull the non-polymers down onto numbers they
    # already clear, so the shift is only ever upward.
    offset = max(0, max(ceiling) - old_min + 1) if ceiling else 0
    return NonPolymerShift(offset=offset, residue_count=len(shifting), old_min=old_min, old_max=old_max)


def apply_shift(changed: Sequence[tuple[gemmi.Chain, gemmi.Residue]], shift: NonPolymerShift) -> None:
    """Move the changed input's non-polymer residues up by the offset, in place.

    A translation rather than a renumbering: the gaps and the order within the
    input's non-polymers survive, so a merged number can be read back against the
    input's by subtracting the offset.

    Args:
        changed: A walk over the changed input's residues. Polymer residues are
            left alone, and so is a residue carrying no sequence number —
            assembly rejects that with a message of its own.
        shift: The shift to apply, from :func:`plan_shift`.
    """
    if shift.offset == 0:
        return
    for _, residue in changed:
        if residue.seqid.num is not None and is_non_polymer(residue):
            residue.seqid.num += shift.offset


def reject_shared_numbers(
    ground: Sequence[tuple[gemmi.Chain, gemmi.Residue]],
    changed: Sequence[tuple[gemmi.Chain, gemmi.Residue]],
    *,
    ground_label: str,
    changed_label: str,
) -> None:
    """Confirm no authored position holds a non-polymer from one input and a residue of the other.

    Two polymer residues at one position are the merge working as intended, even
    where they name different components: an alternate residue identity is a real
    thing to model, and the two inputs' altloc labels are disjoint, so a reader
    can still tell the two apart.

    A non-polymer at that position is not. Because only the changed input's
    numbers move, the shift cannot clear a *ground* non-polymer sitting inside the
    changed input's polymer numbering — vanishingly unlikely, since non-polymers
    are numbered above the polymer they accompany, but silent if it happened.

    Args:
        ground: A walk over the ground input's residues.
        changed: A walk over the changed input's residues, already shifted.
        ground_label: What to call the ground input in the error message.
        changed_label: What to call the changed input in the error message.

    Raises:
        PdbxValidationError: On the first such position found.
    """
    ground_residues = {
        _Position(chain.name, residue.seqid.num, residue.seqid.icode): residue
        for chain, residue in ground
        if residue.seqid.num is not None
    }
    for chain, residue in changed:
        if residue.seqid.num is None:
            continue
        other = ground_residues.get(_Position(chain.name, residue.seqid.num, residue.seqid.icode))
        if other is None or not (is_non_polymer(residue) or is_non_polymer(other)):
            continue
        position = f"chain {chain.name} residue {residue.seqid.num}{residue.seqid.icode.strip()}"
        raise PdbxValidationError(
            f"{position} holds {other.name} in {ground_label} and {residue.name} in {changed_label}, "
            f"and one of them is a non-polymer: merging them would fuse two unrelated molecules into "
            f"one residue. Only the changed model's non-polymer numbering is shifted, so a ground "
            f"non-polymer numbered inside the changed model's polymer range cannot be separated"
        )


class _Position(NamedTuple):
    """The authored position two inputs' residues can collide at.

    Coarser than the occupancy module's key, which goes down to the atom, and
    coarser than assembly's, which includes the component id: what matters here is
    exactly what a viewer reads as one residue slot.
    """

    auth_asym_id: str
    auth_seq_id: int
    insertion_code: str


class _ResidueNumber(NamedTuple):
    """A residue's sequence number and whether its number crosses inputs."""

    value: int
    non_polymer: bool


def _classified_numbers(walk: Sequence[tuple[gemmi.Chain, gemmi.Residue]]) -> list[_ResidueNumber]:
    """Return the numbered residues of a walk, classified.

    Residues carrying no sequence number are dropped: there is nothing to shift
    and nothing to compare, and assembly rejects them by name.
    """
    return [
        _ResidueNumber(residue.seqid.num, is_non_polymer(residue))
        for _, residue in walk
        if residue.seqid.num is not None
    ]
