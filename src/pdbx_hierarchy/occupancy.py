"""Fold two models' occupancies into one file's worth of occupancy.

When a ground and a changed model are merged, the crystallographer's estimate of
how much of the crystal is in the changed state — ``occ`` — decides how the two
inputs share the whole: ground atoms keep ``1 - occ`` of their original
occupancy and changed atoms ``occ``. The guarantee downstream refinement needs
out of that is a hard one: **no atom position may sum above 1.00 in the merged
file.**

Two decisions make that guarantee exact rather than approximate.

*All arithmetic happens in integer hundredths.* Input occupancies are rounded to
hundredths on read and never leave that grid, so there is no floating-point
tolerance anywhere and "sums above 1.00" means what it says. A file written to
two decimal places cannot express anything finer, so nothing is lost by refusing
to carry it.

*Distribution is largest-remainder.* Scaling a group exactly and then flooring
each atom keeps the bound, but only by biasing every atom downward — a group
entitled to 0.40 lands at 0.39, and the error grows with the number of
conformers. Instead each atom is floored and the leftover hundredths are handed
back to the atoms whose truncated remainders were largest, up to the group's
exact entitlement. Every atom then sits within 0.01 of its exact scaled value.

The bound follows from the entitlement being a floor: a group summing to ``S``
hundredths in one input receives ``floor(factor * S / 100)`` hundredths, so with
``S <= 100`` in each input and the two factors summing to 100, the merged group
sums to at most 100. That is why an input group above ``1.00`` is rejected rather
than repaired — see :func:`check_input_occupancies`.

The fold is one-way. Two-decimal output cannot be inverted back to two inputs'
occupancies, and no inverse is offered.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import TYPE_CHECKING, NamedTuple

from pdbx_hierarchy.exceptions import PdbxValidationError

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import gemmi

#: Occupancies are held as integers on this grid, matching the two decimal places
#: the file format carries.
HUNDREDTHS_PER_UNIT = 100


class PositionKey(NamedTuple):
    """What makes one occupancy group: the positions one physical atom can occupy.

    The key is ``(auth_asym_id, auth_seq_id, label_atom_id)``, with the insertion
    code carried alongside the sequence number because the two together are what
    authored residue numbering means by "which residue".

    Component id is deliberately *not* part of the key. A ligand in the ground
    model and a different ligand in the changed model, sharing an authored
    position, compete for the same physical space; constraining them separately
    would let the merged file claim both are fully present.
    """

    auth_asym_id: str
    #: None only for a residue carrying no sequence number, which assembly
    #: rejects; such atoms are still grouped rather than dropped on the way there.
    auth_seq_id: int | None
    insertion_code: str
    atom_id: str


def to_hundredths(value: float) -> int:
    """Round an occupancy onto the integer-hundredths grid.

    Rounding goes through the value's shortest decimal form so that a file
    reading ``0.505`` rounds half away from zero, to ``0.51``. Multiplying the
    float by 100 and rounding would decide such a case on whichever side of the
    half the binary representation happens to fall, which is not a rule anyone
    could predict from the file.

    Args:
        value: An occupancy as read from a file.

    Returns:
        The occupancy in hundredths, rounded to nearest with halves away from
        zero.
    """
    exact = Decimal(str(value)).scaleb(2)
    return int(exact.to_integral_value(rounding=ROUND_HALF_UP))


def from_hundredths(value: int) -> float:
    """Return an occupancy in hundredths as the float to write to a file.

    Args:
        value: An occupancy in hundredths.

    Returns:
        The occupancy as a fraction of one, exact to two decimal places.
    """
    return value / HUNDREDTHS_PER_UNIT


def scale_group(occupancies: Sequence[int], *, factor: int) -> list[int]:
    """Scale one group's occupancies by ``factor``, distributing the remainder.

    Largest-remainder apportionment: the group's entitlement is the floor of its
    exact scaled total, each atom takes the floor of its own exact scaled value,
    and the hundredths still unspent go one each to the atoms with the largest
    truncated remainders. Ties go to the earlier atom, so the result depends only
    on the input and its order.

    Args:
        occupancies: The group's occupancies in hundredths, in file order.
        factor: The scaling factor in hundredths — ``1 - occ`` for a ground atom,
            ``occ`` for a changed atom.

    Returns:
        The scaled occupancies in hundredths, in the same order. The total is at
        most the exact scaled total, and each value is within one hundredth of
        its own exact scaled value.
    """
    # In ten-thousandths, so the floor and the remainder are both exact integers.
    products = [value * factor for value in occupancies]
    scaled = [product // HUNDREDTHS_PER_UNIT for product in products]

    entitlement = sum(products) // HUNDREDTHS_PER_UNIT
    leftover = entitlement - sum(scaled)
    # Only atoms with a truncated remainder can receive one, and there are always
    # strictly more of those than there are hundredths to hand out.
    by_remainder = sorted(range(len(products)), key=lambda index: (-(products[index] % HUNDREDTHS_PER_UNIT), index))
    for index in by_remainder[:leftover]:
        scaled[index] += 1
    return scaled


def position_groups(
    atoms: Iterable[tuple[gemmi.Chain, gemmi.Residue, gemmi.Atom]],
) -> dict[PositionKey, list[gemmi.Atom]]:
    """Group atoms by the position they occupy.

    Args:
        atoms: A walk over a structure's atoms, each with its chain and residue.

    Returns:
        The atoms of each position, keyed by position and in walk order. The
        atoms are the walk's own objects, so writing to them writes through to
        the structure.
    """
    groups: dict[PositionKey, list[gemmi.Atom]] = {}
    for chain, residue, atom in atoms:
        key = PositionKey(chain.name, residue.seqid.num, residue.seqid.icode, atom.name)
        groups.setdefault(key, []).append(atom)
    return groups


def check_input_occupancies(groups: dict[PositionKey, list[gemmi.Atom]], *, label: str) -> int:
    """Reject an input whose own occupancies are already impossible.

    Scaling cannot repair a position summing above ``1.00``: handing it a share
    of the crystal only shrinks it proportionally, and the merged file would
    still claim more of one position than exists. The strictness is intentional —
    three conformers at ``0.50 / 0.50 / 0.01`` is a modelling error, and because
    the arithmetic is in whole hundredths it cannot be float noise.

    A position summing *below* ``1.00`` is left alone and not reported: it is the
    honest encoding of an atom that is only partly there, and on a real model it
    applies to many atoms.

    Args:
        groups: The input's position groups, from :func:`position_groups`.
        label: What to call the file in the error message.

    Returns:
        How many atoms carry an occupancy of ``0.00``. Reported once by the
        caller, with the count, rather than per atom.

    Raises:
        PdbxValidationError: If any position sums above ``1.00``.
    """
    zero_count = 0
    for key, atoms in groups.items():
        hundredths = [to_hundredths(atom.occ) for atom in atoms]
        zero_count += sum(1 for value in hundredths if value == 0)
        total = sum(hundredths)
        if total > HUNDREDTHS_PER_UNIT:
            number = "?" if key.auth_seq_id is None else str(key.auth_seq_id)
            residue = f"{number}{key.insertion_code.strip()}"
            raise PdbxValidationError(
                f"{label}: chain {key.auth_asym_id} residue {residue} atom {key.atom_id} "
                f"has occupancies summing to {from_hundredths(total):.2f}, above 1.00"
            )
    return zero_count


def apply_scaling(groups: dict[PositionKey, list[gemmi.Atom]], *, factor: int) -> None:
    """Write each group's scaled occupancies back onto its atoms.

    Args:
        groups: The position groups to scale, from :func:`position_groups`.
        factor: The scaling factor in hundredths, applied to every group.
    """
    for atoms in groups.values():
        scaled = scale_group([to_hundredths(atom.occ) for atom in atoms], factor=factor)
        for atom, value in zip(atoms, scaled, strict=True):
            atom.occ = from_hundredths(value)
