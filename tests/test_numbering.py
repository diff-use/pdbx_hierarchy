"""Unit tests for non-polymer residue renumbering.

The renumbering is a single integer offset, which makes it small enough that a
whole-file assertion would look like proof while testing almost none of it: which
residues count toward the ceiling, which residues move, and what happens when a
collision survives the shift are all invisible in the fixture pair, where the
offset comes out as ``+1``. They are tested here directly.
"""

from __future__ import annotations

import gemmi
import pytest

from pdbx_hierarchy import numbering
from pdbx_hierarchy.exceptions import PdbxValidationError

POLYMER = gemmi.EntityType.Polymer
NON_POLYMER = gemmi.EntityType.NonPolymer
WATER = gemmi.EntityType.Water
BRANCHED = gemmi.EntityType.Branched
UNKNOWN = gemmi.EntityType.Unknown

#: One hand-made residue: component id, sequence number, entity type.
_Residue = tuple[str, int | None, gemmi.EntityType]


def _walk(*chains: tuple[str, list[_Residue]]) -> list[tuple[gemmi.Chain, gemmi.Residue]]:
    """Build a residue walk over hand-made chains.

    Every residue is added to its chain before any reference to it is taken, so
    that growing the chain cannot dangle a reference already handed out.
    """
    walk: list[tuple[gemmi.Chain, gemmi.Residue]] = []
    for chain_name, residues in chains:
        chain = gemmi.Chain(chain_name)
        for comp_id, number, entity_type in residues:
            residue = gemmi.Residue()
            residue.name = comp_id
            residue.seqid = gemmi.SeqId(1, " ")
            residue.seqid.num = number
            residue.entity_type = entity_type
            chain.add_residue(residue)
        walk.extend((chain, residue) for residue in chain)
    return walk


def _residues(walk: list[tuple[gemmi.Chain, gemmi.Residue]]) -> list[tuple[str, str, int | None]]:
    """Return each residue as (chain, comp_id, sequence number)."""
    return [(chain.name, residue.name, residue.seqid.num) for chain, residue in walk]


class TestIsNonPolymer:
    @pytest.mark.parametrize("entity_type", [NON_POLYMER, WATER, BRANCHED])
    def test_ligands_waters_and_sugars_are_non_polymers(self, entity_type: gemmi.EntityType) -> None:
        # None of the three correspond between two independently refined models.
        walk = _walk(("A", [("LIG", 1, entity_type)]))
        assert numbering.is_non_polymer(walk[0][1])

    def test_a_polymer_residue_is_not(self) -> None:
        walk = _walk(("A", [("ALA", 1, POLYMER)]))
        assert not numbering.is_non_polymer(walk[0][1])

    def test_an_unclassified_residue_is_not(self) -> None:
        # Fail-safe: shifting a residue that turns out to be polymer would break
        # the correspondence the hierarchy encodes, silently. Leaving it alone can
        # only leave a collision, which reject_shared_numbers then reports.
        walk = _walk(("A", [("ALA", 1, UNKNOWN)]))
        assert not numbering.is_non_polymer(walk[0][1])


class TestPlanShift:
    def test_non_polymers_land_immediately_above_grounds_highest(self) -> None:
        ground = _walk(("A", [("ALA", 1, POLYMER), ("EDO", 101, NON_POLYMER)]))
        changed = _walk(("A", [("ALA", 1, POLYMER), ("HOH", 51, WATER), ("HOH", 60, WATER)]))
        shift = numbering.plan_shift(ground, changed)
        assert shift.offset == 51  # 51 + 51 == 102, one above ground's 101
        assert (shift.old_min, shift.old_max) == (51, 60)
        assert (shift.new_min, shift.new_max) == (102, 111)
        assert shift.residue_count == 2

    def test_grounds_highest_counts_its_non_polymers_too(self) -> None:
        # Ground keeps its own numbering, non-polymers included, so the ceiling is
        # the highest number ground uses anywhere and not just its polymer's.
        ground = _walk(("A", [("ALA", 1, POLYMER), ("HOH", 400, WATER)]))
        changed = _walk(("A", [("ALA", 1, POLYMER), ("HOH", 200, WATER)]))
        assert numbering.plan_shift(ground, changed).new_min == 401

    def test_changeds_own_polymer_numbering_also_raises_the_ceiling(self) -> None:
        # A shifted non-polymer must clear changed's polymer residues as well:
        # landing on one would fuse two residues of the same input.
        ground = _walk(("A", [("ALA", 1, POLYMER)]))
        changed = _walk(("A", [("ALA", 900, POLYMER), ("HOH", 20, WATER)]))
        assert numbering.plan_shift(ground, changed).new_min == 901

    def test_non_polymers_already_above_the_ceiling_are_left_alone(self) -> None:
        ground = _walk(("A", [("ALA", 1, POLYMER), ("HOH", 100, WATER)]))
        changed = _walk(("A", [("ALA", 1, POLYMER), ("HOH", 500, WATER)]))
        shift = numbering.plan_shift(ground, changed)
        assert shift.offset == 0
        assert shift.residue_count == 1

    def test_a_changed_input_with_no_non_polymers_shifts_nothing(self) -> None:
        ground = _walk(("A", [("ALA", 1, POLYMER), ("HOH", 100, WATER)]))
        changed = _walk(("A", [("ALA", 1, POLYMER)]))
        shift = numbering.plan_shift(ground, changed)
        assert shift == numbering.NonPolymerShift(offset=0, residue_count=0, old_min=None, old_max=None)

    def test_the_shift_is_one_offset_across_every_chain(self) -> None:
        # Per-chain offsets would let two chains' waters be renumbered by
        # different amounts, so the printed offset would not describe the file.
        ground = _walk(("A", [("HOH", 300, WATER)]), ("B", [("HOH", 500, WATER)]))
        changed = _walk(("A", [("HOH", 100, WATER)]), ("B", [("HOH", 700, WATER)]))
        shift = numbering.plan_shift(ground, changed)
        assert shift.offset == 401  # clears 500, the highest number ground uses
        assert (shift.new_min, shift.new_max) == (501, 1101)

    def test_a_residue_without_a_number_is_left_for_assembly_to_reject(self) -> None:
        ground = _walk(("A", [("ALA", 1, POLYMER)]))
        changed = _walk(("A", [("HOH", None, WATER)]))
        shift = numbering.plan_shift(ground, changed)
        assert shift.residue_count == 0


class TestApplyShift:
    def test_only_non_polymers_move(self) -> None:
        changed = _walk(("A", [("ALA", 1, POLYMER), ("SER", 2, POLYMER), ("EDO", 101, NON_POLYMER)]))
        numbering.apply_shift(changed, numbering.NonPolymerShift(offset=100, residue_count=1, old_min=101, old_max=101))
        assert _residues(changed) == [("A", "ALA", 1), ("A", "SER", 2), ("A", "EDO", 201)]

    def test_gaps_and_order_within_the_non_polymers_survive(self) -> None:
        # The offset is a translation, so two waters ten apart stay ten apart: a
        # merged file's numbering is still readable against its input's.
        changed = _walk(("A", [("HOH", 10, WATER), ("HOH", 20, WATER), ("HOH", 21, WATER)]))
        numbering.apply_shift(changed, numbering.NonPolymerShift(offset=90, residue_count=3, old_min=10, old_max=21))
        assert [residue.seqid.num for _, residue in changed] == [100, 110, 111]

    def test_a_zero_offset_writes_nothing(self) -> None:
        changed = _walk(("A", [("HOH", 500, WATER)]))
        numbering.apply_shift(changed, numbering.NonPolymerShift(offset=0, residue_count=1, old_min=500, old_max=500))
        assert _residues(changed) == [("A", "HOH", 500)]

    def test_insertion_codes_are_untouched(self) -> None:
        changed = _walk(("A", [("HOH", 10, WATER)]))
        changed[0][1].seqid.icode = "B"
        numbering.apply_shift(changed, numbering.NonPolymerShift(offset=90, residue_count=1, old_min=10, old_max=10))
        assert (changed[0][1].seqid.num, changed[0][1].seqid.icode) == (100, "B")


class TestRejectSharedNumbers:
    def test_a_polymer_residue_present_in_both_inputs_is_the_point(self) -> None:
        ground = _walk(("A", [("ALA", 1, POLYMER)]))
        changed = _walk(("A", [("ALA", 1, POLYMER)]))
        numbering.reject_shared_numbers(ground, changed, ground_label="g.cif", changed_label="c.cif")

    def test_two_polymer_residues_of_differing_components_are_allowed(self) -> None:
        # An alternate residue identity at one position is a real thing to model,
        # and the disjoint altloc labels keep the two readable apart.
        ground = _walk(("A", [("ALA", 47, POLYMER)]))
        changed = _walk(("A", [("SER", 47, POLYMER)]))
        numbering.reject_shared_numbers(ground, changed, ground_label="g.cif", changed_label="c.cif")

    def test_the_same_number_in_two_chains_is_not_a_collision(self) -> None:
        ground = _walk(("A", [("HOH", 300, WATER)]))
        changed = _walk(("B", [("HOH", 300, WATER)]))
        numbering.reject_shared_numbers(ground, changed, ground_label="g.cif", changed_label="c.cif")

    def test_a_ground_non_polymer_sharing_a_number_is_rejected(self) -> None:
        # The one case the asymmetric shift cannot fix: only changed's numbers
        # move, so a ground water sitting inside changed's polymer range stays put.
        ground = _walk(("A", [("HOH", 47, WATER)]))
        changed = _walk(("A", [("ALA", 47, POLYMER)]))
        with pytest.raises(PdbxValidationError) as exc:
            numbering.reject_shared_numbers(ground, changed, ground_label="g.cif", changed_label="c.cif")
        message = str(exc.value)
        assert "chain A" in message
        assert "47" in message
        assert "HOH" in message
        assert "ALA" in message

    def test_a_changed_non_polymer_sharing_a_number_is_rejected(self) -> None:
        ground = _walk(("A", [("ALA", 47, POLYMER)]))
        changed = _walk(("A", [("HOH", 47, WATER)]))
        with pytest.raises(PdbxValidationError, match="47"):
            numbering.reject_shared_numbers(ground, changed, ground_label="g.cif", changed_label="c.cif")

    def test_an_insertion_code_separates_two_residues(self) -> None:
        ground = _walk(("A", [("HOH", 47, WATER)]))
        changed = _walk(("A", [("ALA", 47, POLYMER)]))
        changed[0][1].seqid.icode = "A"
        numbering.reject_shared_numbers(ground, changed, ground_label="g.cif", changed_label="c.cif")


class TestSeparateNonPolymers:
    def test_colliding_non_polymers_come_apart(self) -> None:
        ground = _walk(("A", [("ALA", 1, POLYMER), ("EDO", 101, NON_POLYMER)]))
        changed = _walk(("A", [("ALA", 1, POLYMER), ("W2S", 101, NON_POLYMER)]))
        shift = numbering.separate_non_polymers(ground, changed, ground_label="g.cif", changed_label="c.cif")
        assert shift.offset == 1
        assert _residues(changed) == [("A", "ALA", 1), ("A", "W2S", 102)]
        # Polymer numbering is what the hierarchy encodes; it is left as it was.
        assert _residues(ground) == [("A", "ALA", 1), ("A", "EDO", 101)]

    def test_a_collision_the_shift_cannot_fix_is_reported(self) -> None:
        ground = _walk(("A", [("HOH", 47, WATER)]))
        changed = _walk(("A", [("ALA", 47, POLYMER)]))
        with pytest.raises(PdbxValidationError, match="47"):
            numbering.separate_non_polymers(ground, changed, ground_label="g.cif", changed_label="c.cif")

    def test_the_check_runs_on_the_shifted_numbers(self) -> None:
        # Before the shift these two collide; after it they do not, so verifying
        # the wrong side of the shift would reject a pair that is perfectly fine.
        ground = _walk(("A", [("EDO", 101, NON_POLYMER)]))
        changed = _walk(("A", [("HOH", 101, WATER)]))
        assert numbering.separate_non_polymers(ground, changed, ground_label="g", changed_label="c").offset == 1
