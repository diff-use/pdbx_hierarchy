"""Unit tests for the occupancy arithmetic.

Integer-hundredths largest-remainder distribution is the fiddliest logic in
``merge-states`` and the one whose bugs a whole-file assertion would fail to
localise, so it is tested here directly rather than only through the CLI.
"""

from __future__ import annotations

import gemmi
import pytest

from pdbx_hierarchy import occupancy
from pdbx_hierarchy.exceptions import PdbxValidationError


class TestToHundredths:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0.0, 0), (0.4, 40), (0.6, 60), (1.0, 100), (0.33, 33), (0.505, 51), (0.001, 0)],
    )
    def test_an_occupancy_becomes_integer_hundredths(self, value: float, expected: int) -> None:
        assert occupancy.to_hundredths(value) == expected


class TestScaleGroup:
    def test_an_exactly_divisible_group_needs_no_distribution(self) -> None:
        # 0.50 / 0.50 scaled by 0.60 is 0.30 / 0.30 with nothing left over.
        assert occupancy.scale_group([50, 50], factor=60) == [30, 30]

    def test_a_single_fully_occupied_atom_takes_the_factor(self) -> None:
        assert occupancy.scale_group([100], factor=40) == [40]

    def test_the_leftover_hundredth_goes_to_the_largest_remainder(self) -> None:
        # 0.33 / 0.33 / 0.34 by 0.40 is exactly 0.132 / 0.132 / 0.136: flooring
        # alone gives 0.39 for a group that is entitled to 0.40, and the atom
        # with the largest truncated remainder is the third.
        assert occupancy.scale_group([33, 33, 34], factor=40) == [13, 13, 14]

    def test_a_tie_between_remainders_is_broken_by_position(self) -> None:
        # 0.50 / 0.50 by 0.25 is 0.125 each; only one atom can have the leftover
        # hundredth, and it goes to the earlier one so the result is stable.
        assert occupancy.scale_group([50, 50], factor=25) == [13, 12]

    def test_every_atom_stays_within_one_hundredth_of_its_exact_value(self) -> None:
        occupancies = [33, 33, 34]
        scaled = occupancy.scale_group(occupancies, factor=40)
        for original, result in zip(occupancies, scaled, strict=True):
            assert abs(result - original * 40 / 100) < 1

    def test_a_full_group_never_exceeds_its_share_of_the_whole(self) -> None:
        # The bound that makes the merge safe: a group summing to 1.00 in each
        # input cannot sum above 1.00 once both shares are added.
        ground = occupancy.scale_group([33, 33, 34], factor=40)
        changed = occupancy.scale_group([50, 25, 25], factor=60)
        assert sum(ground) + sum(changed) <= 100

    @pytest.mark.parametrize("factor", list(range(1, 100)))
    def test_the_bound_holds_for_every_admissible_factor(self, factor: int) -> None:
        group = [33, 33, 34]
        assert (
            sum(occupancy.scale_group(group, factor=factor)) + sum(occupancy.scale_group(group, factor=100 - factor))
            <= 100
        )

    def test_a_group_summing_below_one_stays_below(self) -> None:
        # An atom modelled in only one state: 0.60 of the crystal is all it gets,
        # and nothing tops it up.
        assert occupancy.scale_group([100], factor=60) == [60]

    def test_a_zero_occupancy_atom_passes_through_as_zero(self) -> None:
        assert occupancy.scale_group([0, 100], factor=60) == [0, 60]

    def test_a_zero_occupancy_atom_is_never_handed_a_leftover(self) -> None:
        # A zero has no truncated remainder, so distribution must not reach it
        # even when there is a hundredth going spare.
        assert occupancy.scale_group([0, 33, 33, 34], factor=40) == [0, 13, 13, 14]

    def test_an_empty_group(self) -> None:
        assert occupancy.scale_group([], factor=60) == []


def _structure(rows: list[tuple[str, int, str, str, float]]) -> gemmi.Structure:
    """Build a one-model structure from (chain, seq_num, comp, atom, occ) rows."""
    structure = gemmi.Structure()
    model = gemmi.Model(1)
    chains: dict[str, gemmi.Chain] = {}
    for chain_name, seq_num, comp, atom_name, occ in rows:
        if chain_name not in chains:
            chains[chain_name] = gemmi.Chain(chain_name)
        chain = chains[chain_name]
        if len(chain) == 0 or chain[len(chain) - 1].seqid.num != seq_num or chain[len(chain) - 1].name != comp:
            residue = gemmi.Residue()
            residue.seqid = gemmi.SeqId(seq_num, " ")
            residue.name = comp
            chain.add_residue(residue)
        atom = gemmi.Atom()
        atom.name = atom_name
        atom.occ = occ
        atom.element = gemmi.Element("C")
        chain[len(chain) - 1].add_atom(atom)
    for chain in chains.values():
        model.add_chain(chain)
    structure.add_model(model)
    return structure


def _walk(structure: gemmi.Structure) -> list[tuple[gemmi.Chain, gemmi.Residue, gemmi.Atom]]:
    return [(chain, residue, atom) for chain in structure[0] for residue in chain for atom in residue]


def _groups(structure: gemmi.Structure) -> dict[occupancy.PositionKey, list[gemmi.Atom]]:
    return occupancy.position_groups(_walk(structure))


class TestPositionGroups:
    def test_conformers_of_one_atom_share_a_group(self) -> None:
        structure = _structure([("A", 2, "SER", "CA", 0.5), ("A", 2, "SER", "CA", 0.5)])
        assert [len(atoms) for atoms in _groups(structure).values()] == [2]

    def test_different_components_at_one_position_share_a_group(self) -> None:
        # comp_id is excluded from the key on purpose: a ground ligand and a
        # changed ligand at the same authored position compete for the same
        # physical space, so their occupancies are constrained together.
        structure = _structure([("A", 101, "EDO", "C1", 1.0), ("A", 101, "W2S", "C1", 1.0)])
        assert [len(atoms) for atoms in _groups(structure).values()] == [2]

    def test_different_atom_names_are_different_groups(self) -> None:
        structure = _structure([("A", 2, "SER", "CA", 1.0), ("A", 2, "SER", "CB", 1.0)])
        assert len(_groups(structure)) == 2

    def test_the_same_atom_in_two_chains_is_two_groups(self) -> None:
        structure = _structure([("A", 2, "SER", "CA", 1.0), ("B", 2, "SER", "CA", 1.0)])
        assert len(_groups(structure)) == 2


class TestCheckInputOccupancies:
    def test_a_group_summing_to_one_is_accepted(self) -> None:
        structure = _structure([("A", 2, "SER", "CA", 0.5), ("A", 2, "SER", "CA", 0.5)])
        assert occupancy.check_input_occupancies(_groups(structure), label="in.cif") == 0

    def test_a_group_summing_above_one_names_chain_residue_and_atom(self) -> None:
        structure = _structure([("A", 2, "SER", "CA", 0.5), ("A", 2, "SER", "CA", 0.5), ("A", 2, "SER", "CA", 0.01)])
        with pytest.raises(PdbxValidationError) as excinfo:
            occupancy.check_input_occupancies(_groups(structure), label="in.cif")
        message = str(excinfo.value)
        assert "in.cif" in message
        assert "chain A" in message
        assert "residue 2" in message
        assert "atom CA" in message
        assert "1.01" in message

    def test_a_group_below_one_is_neither_corrected_nor_reported(self) -> None:
        structure = _structure([("A", 2, "SER", "CA", 0.5)])
        assert occupancy.check_input_occupancies(_groups(structure), label="in.cif") == 0

    def test_zero_occupancy_atoms_are_counted_not_rejected(self) -> None:
        structure = _structure([("A", 2, "SER", "CA", 0.0), ("A", 2, "SER", "CB", 0.0)])
        assert occupancy.check_input_occupancies(_groups(structure), label="in.cif") == 2


class TestApplyScaling:
    def test_the_atoms_carry_their_scaled_occupancies(self) -> None:
        structure = _structure([("A", 2, "SER", "CA", 0.5), ("A", 2, "SER", "CA", 0.5)])
        occupancy.apply_scaling(_groups(structure), factor=60)
        assert [atom.occ for _, _, atom in _walk(structure)] == pytest.approx([0.30, 0.30])

    def test_an_input_occupancy_off_the_hundredths_grid_is_rounded_first(self) -> None:
        structure = _structure([("A", 2, "SER", "CA", 0.333)])
        occupancy.apply_scaling(_groups(structure), factor=60)
        # 0.333 rounds to 0.33 on read, and 0.33 * 0.60 floors to 0.19.
        assert [atom.occ for _, _, atom in _walk(structure)] == pytest.approx([0.19])
