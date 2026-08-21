"""Tests for the ``merge-states`` command.

The fixture pair is ``merge_ground.cif`` / ``merge_changed.cif``: 19 atoms each,
polymer residues 1-3 in chain A with a two-conformer SER at 2, and a non-polymer
at auth_seq_id 101 that collides between the two inputs (EDO in ground, W2S in
changed).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import gemmi
import pytest
import typer
from typer.testing import CliRunner

from pdbx_hierarchy.altloc import ALPHABET
from pdbx_hierarchy.io.reader import read_atom_site_heterogeneity_ids, read_hierarchy

MERGED_NAME = "merge_changed_merge_ground_hierarchy.cif"
GROUND_INTERMEDIATE = "merge_ground_hierarchy.cif"
CHANGED_INTERMEDIATE = "merge_changed_hierarchy.cif"

ALLOWED_CATEGORIES = {
    "_entry.",
    "_cell.",
    "_symmetry.",
    "_chem_comp.",
    "_entity.",
    "_struct_asym.",
    "_atom_site.",
    "_pdbx_heterogeneity_hierarchy.",
}


@pytest.fixture
def pair(copy_fixture: Callable[[str], Path]) -> tuple[Path, Path]:
    """Copy the ground/changed fixture pair into tmp_path."""
    return copy_fixture("merge_ground.cif"), copy_fixture("merge_changed.cif")


@pytest.fixture
def work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run from a directory that is neither the inputs' nor the output's.

    The intermediates land in the working directory, so telling all three apart
    is the only way to assert that.
    """
    directory = tmp_path / "work"
    directory.mkdir()
    monkeypatch.chdir(directory)
    return directory


def _merge(
    runner: CliRunner,
    app: typer.Typer,
    pair: tuple[Path, Path],
    out: Path,
    *extra: str,
    yes: bool = True,
    answer: str | None = None,
) -> tuple[int, str]:
    """Invoke ``merge-states`` on the fixture pair.

    Args:
        runner: The CLI runner.
        app: The CLI app.
        pair: The ground/changed inputs.
        out: The ``-o`` path.
        extra: Further command-line arguments.
        yes: If False, omit ``-y`` so the overwrite prompt is left to fire.
        answer: What to feed that prompt on stdin.

    Returns:
        Tuple of (exit code, stdout and stderr together).
    """
    ground, changed = pair
    result = runner.invoke(
        app,
        [
            "merge-states",
            "--ground",
            str(ground),
            "--changed",
            str(changed),
            "--occ",
            "0.60",
            "-o",
            str(out),
            *(["-y"] if yes else []),
            *extra,
        ],
        input=answer,
    )
    return result.exit_code, result.output + result.stderr


def _categories(path: Path) -> set[str]:
    block = gemmi.cif.read(str(path)).sole_block()
    found: set[str] = set()
    for item in block:
        if item.pair is not None:
            found.add(item.pair[0].split(".")[0] + ".")
        elif item.loop is not None:
            found.add(item.loop.tags[0].split(".")[0] + ".")
    return found


def _column(path: Path, tag: str) -> list[str]:
    block = gemmi.cif.read(str(path)).sole_block()
    return list(block.find_loop(tag))


def _labels_by_branch(path: Path) -> tuple[set[str], set[str]]:
    """Return the altloc labels used by the ground branch and by the changed branch."""
    tree = read_hierarchy(path)
    ground_ids = {"Ground"} | {state.id for state in tree.get_descendants("Ground")}
    ground: set[str] = set()
    changed: set[str] = set()
    labels = _column(path, "_atom_site.label_alt_id")
    for label, state in zip(labels, read_atom_site_heterogeneity_ids(path), strict=True):
        (ground if state in ground_ids else changed).add(label)
    return ground, changed


def _residues_by_branch(path: Path) -> tuple[set[tuple[str, int, str]], set[tuple[str, int, str]]]:
    """Return the authored residues of the ground branch and of the changed branch.

    Each residue is ``(auth_asym_id, auth_seq_id, label_comp_id)``, read back from
    the merged file's rows so that what is asserted is what a reader sees.
    """
    tree = read_hierarchy(path)
    ground_ids = {"Ground"} | {state.id for state in tree.get_descendants("Ground")}
    ground: set[tuple[str, int, str]] = set()
    changed: set[tuple[str, int, str]] = set()
    columns = ("auth_asym_id", "auth_seq_id", "label_comp_id")
    chains, numbers, comps = (_column(path, f"_atom_site.{tag}") for tag in columns)
    rows = zip(chains, numbers, comps, read_atom_site_heterogeneity_ids(path), strict=True)
    for chain, number, comp, state in rows:
        (ground if state in ground_ids else changed).add((chain, int(number), comp))
    return ground, changed


def _residues(path: Path) -> set[tuple[str, int, str]]:
    """Return the authored residues of a file as ``(auth_asym_id, auth_seq_id, label_comp_id)``."""
    columns = ("auth_asym_id", "auth_seq_id", "label_comp_id")
    chains, numbers, comps = (_column(path, f"_atom_site.{tag}") for tag in columns)
    return {(chain, int(number), comp) for chain, number, comp in zip(chains, numbers, comps, strict=True)}


def _add_altloc_atoms(path: Path, labels: str) -> None:
    """Rewrite an input so its first residue carries one extra atom per label.

    Each atom gets its own name, so the additions are separate positions and the
    occupancy bound is untouched — the point is only to spend altloc labels.

    Args:
        path: The input file, overwritten in place.
        labels: One altloc label per atom to add.
    """
    structure = gemmi.read_structure(str(path))
    additions = []
    for index, label in enumerate(labels):
        atom = gemmi.Atom()
        atom.name = f"Z{index}"
        atom.element = gemmi.Element("C")
        atom.altloc = label
        atom.pos = gemmi.Position(20.0 + index, 20.0, 20.0)
        atom.occ = 1.0
        atom.b_iso = 20.0
        additions.append(atom)
    residue = structure[0]["A"][0]
    for atom in additions:
        residue.add_atom(atom)
    structure.make_mmcif_document().write_file(str(path))


def _substitute(path: Path, old: str, new: str) -> None:
    """Replace every occurrence of a string in an input file, insisting it was there.

    Used to perturb a cell or symmetry value in a copied fixture. The
    assertion is what keeps a test honest if the fixture's wording ever changes:
    without it, a stale pattern would silently perturb nothing and the test would
    pass by asserting a warning is absent from an unmodified pair.

    Args:
        path: The copied fixture to edit.
        old: The exact text to replace.
        new: What to put in its place.
    """
    text = path.read_text()
    assert old in text, f"{path.name} does not contain {old!r}"
    path.write_text(text.replace(old, new))


def _with_coexistence(runner: CliRunner, app: typer.Typer, source: Path, out: Path) -> Path:
    """Return a copy of an input carrying an inferred hierarchy and one coexistence rule.

    Built through the CLI rather than hand-written so the rule is exactly what
    this toolbox writes — the only way an input reaching ``merge-states`` comes to
    hold coexistence rules at all.

    Args:
        runner: The CLI runner.
        app: The CLI app.
        source: The input to start from.
        out: Where to write the result.

    Returns:
        The path written.
    """
    inferred = out.with_name(f"{out.stem}_inferred.cif")
    assert runner.invoke(app, ["infer", str(source), "-o", str(inferred), "-y"]).exit_code == 0
    result = runner.invoke(
        app,
        ["coexist", "add", str(inferred), "--rule", "OR", "--source", "Base", "--related", "A", "-o", str(out), "-y"],
    )
    assert result.exit_code == 0, result.output + result.stderr
    return out


def _position_sums(path: Path) -> dict[tuple[str, str, str], int]:
    """Sum the written occupancies, in hundredths, over each atom position.

    The key is the occupancy group of the spec: authored chain and residue number
    plus atom name, with comp_id deliberately left out.
    """
    columns = ("auth_asym_id", "auth_seq_id", "label_atom_id", "occupancy")
    chains, residues, atoms, occupancies = (_column(path, f"_atom_site.{tag}") for tag in columns)
    sums: dict[tuple[str, str, str], int] = {}
    for chain, residue, atom, value in zip(chains, residues, atoms, occupancies, strict=True):
        sums[(chain, residue, atom)] = sums.get((chain, residue, atom), 0) + round(float(value) * 100)
    return sums


class TestTree:
    def test_base_has_exactly_ground_and_changed(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        out = tmp_path / "merged.cif"
        code, output = _merge(runner, app, pair, out)
        assert code == 0, output
        tree = read_hierarchy(out)
        root = tree.get_root()
        assert (root.id, root.name) == ("Base", "base_state")
        children = {(s.id, s.name) for s in tree.get_children("Base")}
        assert children == {("Ground", "ground_state"), ("Changed", "changed_state")}

    def test_base_owns_no_atoms(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        out = tmp_path / "merged.cif"
        assert _merge(runner, app, pair, out)[0] == 0
        assert "Base" not in read_atom_site_heterogeneity_ids(out)

    def test_changed_ids_reassigned_above_ground(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        out = tmp_path / "merged.cif"
        assert _merge(runner, app, pair, out)[0] == 0
        tree = read_hierarchy(out)
        # Each input infers two conformer states from its SER; ground keeps A/B and
        # changed's are pushed above them.
        assert {s.id for s in tree.get_children("Ground")} == {"A", "B"}
        assert {s.id for s in tree.get_children("Changed")} == {"C", "D"}

    def test_reassigned_states_are_renamed(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        out = tmp_path / "merged.cif"
        assert _merge(runner, app, pair, out)[0] == 0
        tree = read_hierarchy(out)
        assert {s.name for s in tree.get_children("Changed")} == {"state_C", "state_D"}

    def test_every_atom_references_a_defined_state(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        out = tmp_path / "merged.cif"
        assert _merge(runner, app, pair, out)[0] == 0
        tree = read_hierarchy(out)
        assert all(tree.contains(state_id) for state_id in read_atom_site_heterogeneity_ids(out))

    def test_output_passes_the_validate_command(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        out = tmp_path / "merged.cif"
        assert _merge(runner, app, pair, out)[0] == 0
        result = runner.invoke(app, ["validate", str(out)])
        assert result.exit_code == 0, result.output + result.stderr

    def test_provenance_in_details(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        out = tmp_path / "merged.cif"
        assert _merge(runner, app, pair, out)[0] == 0
        tree = read_hierarchy(out)
        base_details = tree.get_state("Base").details or ""
        assert "merge_ground.cif" in base_details
        assert "merge_changed.cif" in base_details
        assert "0.60" in base_details
        assert "merge_ground.cif" in (tree.get_state("Ground").details or "")
        assert "0.40" in (tree.get_state("Ground").details or "")
        assert "merge_changed.cif" in (tree.get_state("Changed").details or "")
        assert "0.60" in (tree.get_state("Changed").details or "")

    def test_existing_hierarchy_is_reused(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ground, changed = pair
        # Give the ground input a hierarchy carrying a state inference would never
        # produce, then check that state survives into the merged tree.
        inferred = tmp_path / "ground_inferred.cif"
        assert runner.invoke(app, ["infer", str(ground), "-o", str(inferred), "-y"]).exit_code == 0
        hand_tuned = tmp_path / "ground_hand.cif"
        assert (
            runner.invoke(
                app,
                [
                    "hierarchy",
                    "add",
                    str(inferred),
                    "--id",
                    "X",
                    "--name",
                    "hand_tuned",
                    "--parent",
                    "Base",
                    "-o",
                    str(hand_tuned),
                    "-y",
                ],
            ).exit_code
            == 0
        )
        out = tmp_path / "merged.cif"
        result = runner.invoke(
            app,
            [
                "merge-states",
                "--ground",
                str(hand_tuned),
                "--changed",
                str(changed),
                "--occ",
                "0.60",
                "-o",
                str(out),
                "-y",
            ],
        )
        assert result.exit_code == 0, result.output + result.stderr
        tree = read_hierarchy(out)
        assert {s.id for s in tree.get_children("Ground")} == {"A", "B", "X"}

    def test_an_id_whose_generated_name_is_taken_is_skipped(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ground, changed = pair
        # A reused ground hierarchy can already hold a name of the state_X form.
        # Reassigning the changed tree onto C would then produce a second
        # "state_C", which the format forbids as firmly as a duplicate id.
        inferred = tmp_path / "ground_inferred.cif"
        assert runner.invoke(app, ["infer", str(ground), "-o", str(inferred), "-y"]).exit_code == 0
        renamed = tmp_path / "ground_renamed.cif"
        assert (
            runner.invoke(
                app,
                ["hierarchy", "rename", str(inferred), "--id", "A", "--name", "state_C", "-o", str(renamed), "-y"],
            ).exit_code
            == 0
        )
        out = tmp_path / "merged.cif"
        result = runner.invoke(
            app,
            [
                "merge-states",
                "--ground",
                str(renamed),
                "--changed",
                str(changed),
                "--occ",
                "0.60",
                "-o",
                str(out),
                "-y",
            ],
        )
        assert result.exit_code == 0, result.output + result.stderr
        tree = read_hierarchy(out)
        assert {s.id for s in tree.get_children("Changed")} == {"D", "E"}
        assert {s.name for s in tree.get_children("Changed")} == {"state_D", "state_E"}


class TestGuards:
    def test_a_multi_model_input_is_rejected(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ground, changed = pair
        two_models = tmp_path / "two_models.cif"
        structure = gemmi.read_structure(str(ground))
        second = structure[0].clone()
        second.num = 2
        structure.add_model(second)
        structure.make_mmcif_document().write_file(str(two_models))
        result = runner.invoke(
            app,
            [
                "merge-states",
                "--ground",
                str(two_models),
                "--changed",
                str(changed),
                "--occ",
                "0.60",
                "-o",
                str(tmp_path / "merged.cif"),
                "-y",
            ],
        )
        assert result.exit_code == 1
        assert "2 models" in (result.output + result.stderr)

    def test_a_hierarchy_without_atom_assignments_is_rejected(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ground, changed = pair
        # A hierarchy table with no _atom_site.pdbx_heterogeneity_id column says
        # which states exist but not which atoms are in them.
        unassigned = tmp_path / "unassigned.cif"
        unassigned.write_text(
            ground.read_text()
            + "loop_\n"
            + "_pdbx_heterogeneity_hierarchy.id\n"
            + "_pdbx_heterogeneity_hierarchy.name\n"
            + "_pdbx_heterogeneity_hierarchy.parent\n"
            + "_pdbx_heterogeneity_hierarchy.details\n"
            + "Base base_state . ?\n"
            + "A state_A Base ?\n"
            + "#\n"
        )
        result = runner.invoke(
            app,
            [
                "merge-states",
                "--ground",
                str(unassigned),
                "--changed",
                str(changed),
                "--occ",
                "0.60",
                "-o",
                str(tmp_path / "merged.cif"),
                "-y",
            ],
        )
        assert result.exit_code == 1
        assert "pdbx_heterogeneity_id" in (result.output + result.stderr)

    def test_a_unit_cell_mismatch_is_a_warning(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        _, changed = pair
        _substitute(changed, "_cell.length_a 30.000", "_cell.length_a 35.000")
        out = tmp_path / "merged.cif"
        code, output = _merge(runner, app, pair, out)
        assert code == 0, output
        assert "unit cell" in output
        assert "merge_ground.cif" in output
        assert "merge_changed.cif" in output
        # A warning, not an error: the file is still written.
        assert out.exists()

    def test_a_cell_angle_mismatch_is_a_warning(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        _, changed = pair
        _substitute(changed, "_cell.angle_beta 90.00", "_cell.angle_beta 95.00")
        code, output = _merge(runner, app, pair, tmp_path / "merged.cif")
        assert code == 0, output
        assert "unit cell" in output

    def test_the_cell_warning_says_it_should_be_reconsidered_as_an_error(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        _, changed = pair
        _substitute(changed, "_cell.length_a 30.000", "_cell.length_a 35.000")
        code, output = _merge(runner, app, pair, tmp_path / "merged.cif")
        assert code == 0, output
        assert "reconsider making this an error" in output

    def test_a_cell_difference_within_tolerance_is_not_warned(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        _, changed = pair
        # Two refinements of one crystal disagree in the last decimal place; that
        # is not a mispaired file.
        _substitute(changed, "_cell.length_a 30.000", "_cell.length_a 30.050")
        _substitute(changed, "_cell.angle_beta 90.00", "_cell.angle_beta 90.05")
        code, output = _merge(runner, app, pair, tmp_path / "merged.cif")
        assert code == 0, output
        assert "unit cell" not in output

    def test_a_space_group_mismatch_is_a_warning(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        _, changed = pair
        _substitute(changed, "_symmetry.space_group_name_H-M 'P 1'", "_symmetry.space_group_name_H-M 'P 21 21 21'")
        out = tmp_path / "merged.cif"
        code, output = _merge(runner, app, pair, out)
        assert code == 0, output
        assert "space group" in output
        assert out.exists()

    def test_a_matching_pair_warns_about_neither(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        code, output = _merge(runner, app, pair, tmp_path / "merged.cif")
        assert code == 0, output
        assert "unit cell" not in output
        assert "space group" not in output

    def test_reusing_an_existing_hierarchy_is_a_warning(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ground, changed = pair
        inferred = tmp_path / "ground_inferred.cif"
        assert runner.invoke(app, ["infer", str(ground), "-o", str(inferred), "-y"]).exit_code == 0
        code, output = _merge(runner, app, (inferred, changed), tmp_path / "merged.cif")
        assert code == 0, output
        assert "reused the hierarchy" in output
        assert "ground_inferred.cif" in output
        # Only the ground input had one, so the changed input is not warned about.
        assert "merge_changed.cif: reused" not in output

    def test_coexistence_rules_are_dropped_with_a_warning(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ground, changed = pair
        with_rules = _with_coexistence(runner, app, ground, tmp_path / "ground_rules.cif")
        out = tmp_path / "merged.cif"
        code, output = _merge(runner, app, (with_rules, changed), out)
        assert code == 0, output
        assert "1 coexistence rule" in output
        assert "ground_rules.cif" in output
        assert "_pdbx_state_coexistence." not in _categories(out)

    def test_both_inputs_carrying_rules_are_each_named(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ground, changed = pair
        ground_rules = _with_coexistence(runner, app, ground, tmp_path / "ground_rules.cif")
        changed_rules = _with_coexistence(runner, app, changed, tmp_path / "changed_rules.cif")
        code, output = _merge(runner, app, (ground_rules, changed_rules), tmp_path / "merged.cif")
        assert code == 0, output
        assert "ground_rules.cif: dropped 1 coexistence rule" in output
        assert "changed_rules.cif: dropped 1 coexistence rule" in output

    def test_a_rule_written_as_pairs_rather_than_a_loop_is_still_reported(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ground, changed = pair
        # A single row is conventionally written as _tag value pairs, which is what
        # a hand-written or third-party file will hold. Counting only loops would
        # drop this rule in silence — the one outcome the warning exists to avoid.
        as_pairs = tmp_path / "ground_pairs.cif"
        as_pairs.write_text(
            ground.read_text()
            + "_pdbx_state_coexistence.id 1\n"
            + "_pdbx_state_coexistence.rule NOT\n"
            + "_pdbx_state_coexistence.heterogeneity_id A\n"
            + "_pdbx_state_coexistence.heterogeneity_ids B\n"
            + "_pdbx_state_coexistence.description ?\n"
        )
        code, output = _merge(runner, app, (as_pairs, changed), tmp_path / "merged.cif")
        assert code == 0, output
        assert "ground_pairs.cif: dropped 1 coexistence rule" in output


class TestAtoms:
    def test_atoms_from_both_inputs_are_present(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        out = tmp_path / "merged.cif"
        assert _merge(runner, app, pair, out)[0] == 0
        comps = _column(out, "_atom_site.label_comp_id")
        assert len(comps) == 38  # 19 + 19
        assert "EDO" in comps  # ground-only non-polymer
        assert "W2S" in comps  # changed-only non-polymer

    def test_atoms_split_between_the_two_branches(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        out = tmp_path / "merged.cif"
        assert _merge(runner, app, pair, out)[0] == 0
        tree = read_hierarchy(out)
        ground_ids = {"Ground"} | {s.id for s in tree.get_descendants("Ground")}
        changed_ids = {"Changed"} | {s.id for s in tree.get_descendants("Changed")}
        assigned = read_atom_site_heterogeneity_ids(out)
        assert sum(1 for a in assigned if a in ground_ids) == 19
        assert sum(1 for a in assigned if a in changed_ids) == 19

    def test_each_altloc_label_belongs_to_exactly_one_state(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        out = tmp_path / "merged.cif"
        assert _merge(runner, app, pair, out)[0] == 0
        alts = _column(out, "_atom_site.label_alt_id")
        assigned = read_atom_site_heterogeneity_ids(out)
        # What makes the merged file coherent to a hierarchy-unaware reader: a
        # label names one conformation, so every atom carrying it is in one state.
        states_per_label: dict[str, set[str]] = {}
        for alt, state in zip(alts, assigned, strict=True):
            states_per_label.setdefault(alt, set()).add(state)
        assert all(len(states) == 1 for states in states_per_label.values()), states_per_label
        # Formerly-blank atoms sit on their branch root; conformers on a child.
        assert {label for label, states in states_per_label.items() if states <= {"Ground", "Changed"}} == {"A", "D"}
        assert {label for label, states in states_per_label.items() if states <= {"A", "B", "C", "D"}} == {
            "B",
            "C",
            "E",
            "F",
        }


class TestAltloc:
    """The altloc partition: every atom labelled, ground first, no label shared.

    On the fixture pair each input carries three distinct original labels — blank,
    ``A`` and ``B`` — so ground's map to ``A``, ``B``, ``C`` and changed's continue
    at ``D``, ``E``, ``F``. Blank sorts first within an input, which is why the 15
    formerly-blank ground atoms all come out as ``A``.
    """

    def test_every_atom_carries_a_real_label(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        out = tmp_path / "merged.cif"
        assert _merge(runner, app, pair, out)[0] == 0
        labels = _column(out, "_atom_site.label_alt_id")
        assert len(labels) == 38
        assert all(len(label) == 1 and label not in {".", "?"} for label in labels), sorted(set(labels))

    def test_blank_altloc_atoms_receive_real_labels(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        out = tmp_path / "merged.cif"
        assert _merge(runner, app, pair, out)[0] == 0
        labels = _column(out, "_atom_site.label_alt_id")
        # 15 of each input's 19 atoms have no altloc in the input; they take their
        # input's first label rather than staying blank.
        assert labels.count("A") == 15
        assert labels.count("D") == 15

    def test_ground_labels_run_contiguously_from_the_start_of_the_alphabet(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        out = tmp_path / "merged.cif"
        assert _merge(runner, app, pair, out)[0] == 0
        ground, _ = _labels_by_branch(out)
        assert ground == {"A", "B", "C"}

    def test_changed_labels_continue_above_the_highest_ground_label(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        out = tmp_path / "merged.cif"
        assert _merge(runner, app, pair, out)[0] == 0
        ground, changed = _labels_by_branch(out)
        assert max(ALPHABET.index(label) for label in ground) < min(ALPHABET.index(label) for label in changed)
        assert changed == {"D", "E", "F"}

    def test_no_label_is_shared_between_the_two_inputs(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        out = tmp_path / "merged.cif"
        assert _merge(runner, app, pair, out)[0] == 0
        ground, changed = _labels_by_branch(out)
        assert ground.isdisjoint(changed)

    def test_the_full_mapping_is_printed_for_each_input(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        out = tmp_path / "merged.cif"
        code, output = _merge(runner, app, pair, out)
        assert code == 0, output
        assert "merge_ground.cif: .->A, A->B, B->C" in output
        assert "merge_changed.cif: .->D, A->E, B->F" in output

    def test_relabelling_does_not_feed_back_into_state_assignment(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        out = tmp_path / "merged.cif"
        assert _merge(runner, app, pair, out)[0] == 0
        tree = read_hierarchy(out)
        # Inference reads the original labels, so each branch keeps the two
        # conformer states its SER 2 called for. Inferring after the relabel would
        # read all 15 formerly-blank atoms as conformers instead.
        assert len(tree.get_children("Ground")) == 2
        assert len(tree.get_children("Changed")) == 2
        assigned = read_atom_site_heterogeneity_ids(out)
        assert assigned.count("Ground") == 15
        assert assigned.count("Changed") == 15

    def test_exhausting_the_alphabet_is_a_validation_error(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ground, changed = pair
        # 62 labels in the ground input alone, plus its blanks and the changed
        # input's three, needs 66 of the 62 the alphabet has.
        _add_altloc_atoms(ground, ALPHABET)
        code, output = _merge(runner, app, pair, tmp_path / "merged.cif")
        assert code == 1
        assert "66" in output
        assert "62" in output


class TestNonPolymerNumbering:
    """Non-polymer renumbering: changed's non-polymers clear of ground's residues.

    On the fixture pair the offset comes out as ``+1``: ground's highest number is
    its EDO at 101, and changed's only non-polymer is its W2S at 101, so the W2S
    lands at 102 — the smallest shift that clears the ceiling. The unit tests in
    ``test_numbering.py`` cover what the ceiling counts and what happens when a
    collision survives, neither of which this fixture can show.
    """

    def test_changed_non_polymers_are_shifted_above_grounds_highest(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        out = tmp_path / "merged.cif"
        assert _merge(runner, app, pair, out)[0] == 0
        ground, changed = _residues_by_branch(out)
        # The colliding pair the fixture exists for: EDO and W2S both at 101 in
        # their inputs, no longer one residue with two "conformations".
        assert ("A", 101, "EDO") in ground
        assert ("A", 102, "W2S") in changed
        highest_ground = max(number for _, number, _ in ground)
        assert all(number > highest_ground for chain, number, comp in changed if comp == "W2S")

    def test_polymer_numbering_is_unchanged_in_both_inputs(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        out = tmp_path / "merged.cif"
        assert _merge(runner, app, pair, out)[0] == 0
        ground, changed = _residues_by_branch(out)
        polymer = {("A", 1, "ALA"), ("A", 2, "SER"), ("A", 3, "GLY")}
        # Residue 47 is the same residue in both models; that correspondence is
        # what the hierarchy encodes, so neither input's polymer numbers move.
        assert polymer <= ground
        assert polymer <= changed

    def test_a_shared_position_holds_one_polymer_residue_or_nothing(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        out = tmp_path / "merged.cif"
        assert _merge(runner, app, pair, out)[0] == 0
        ground, changed = _residues_by_branch(out)
        ground_comps = {(chain, number): comp for chain, number, comp in ground}
        changed_comps = {(chain, number): comp for chain, number, comp in changed}
        shared = set(ground_comps) & set(changed_comps)
        assert shared == {("A", 1), ("A", 2), ("A", 3)}
        for position in shared:
            assert ground_comps[position] == changed_comps[position]

    def test_the_offset_is_printed(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        code, output = _merge(runner, app, pair, tmp_path / "merged.cif")
        assert code == 0, output
        # The merged file has nowhere to record a non-polymer's deposited number,
        # so this line is the only trace back to it.
        assert "Changed non-polymer auth_seq_id: 101-101 -> 102-102 (offset +1, 1 residue(s))" in output

    def test_a_changed_input_with_no_non_polymer_says_so(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ground, changed = pair
        polymer_only = [line for line in changed.read_text().splitlines(keepends=True) if "W2S" not in line]
        changed.write_text("".join(polymer_only))
        code, output = _merge(runner, app, pair, tmp_path / "merged.cif")
        assert code == 0, output
        assert "Changed non-polymer residues: none to renumber" in output

    def test_a_ground_non_polymer_inside_changeds_polymer_range_is_rejected(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ground, _ = pair
        # Only changed's numbering moves, so this collision is the one the shift
        # cannot clear: ground's EDO sits on the number changed's SER 2 uses.
        ground.write_text(ground.read_text().replace("30.00 101 A 1", "30.00 2 A 1"))
        code, output = _merge(runner, app, pair, tmp_path / "merged.cif")
        assert code == 1
        assert "chain A residue 2" in output
        assert "EDO" in output
        assert "SER" in output


class TestOccupancy:
    def test_each_branch_carries_its_share(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        out = tmp_path / "merged.cif"
        assert _merge(runner, app, pair, out)[0] == 0
        tree = read_hierarchy(out)
        ground_ids = {"Ground"} | {s.id for s in tree.get_descendants("Ground")}
        occupancies = _column(out, "_atom_site.occupancy")
        assigned = read_atom_site_heterogeneity_ids(out)
        by_branch: dict[str, set[str]] = {"ground": set(), "changed": set()}
        for value, state in zip(occupancies, assigned, strict=True):
            by_branch["ground" if state in ground_ids else "changed"].add(value)
        # Every fixture atom is at 1.00 except the SER conformers at 0.50.
        assert by_branch["ground"] == {"0.40", "0.20"}
        assert by_branch["changed"] == {"0.60", "0.30"}

    def test_occupancies_are_written_to_two_decimal_places(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        out = tmp_path / "merged.cif"
        assert _merge(runner, app, pair, out)[0] == 0
        for value in _column(out, "_atom_site.occupancy"):
            assert re.fullmatch(r"\d\.\d\d", value), value

    def test_no_position_sums_above_one(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        out = tmp_path / "merged.cif"
        assert _merge(runner, app, pair, out)[0] == 0
        for key, total in _position_sums(out).items():
            assert total <= 100, f"{key} sums to {total} hundredths"

    def test_a_position_present_in_both_inputs_sums_to_exactly_one(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        out = tmp_path / "merged.cif"
        assert _merge(runner, app, pair, out)[0] == 0
        sums = _position_sums(out)
        assert sums[("A", "1", "CA")] == 100  # polymer atom, both inputs
        assert sums[("A", "2", "CA")] == 100  # two conformers in each input
        # Only a polymer position can span both inputs: renumbering moves the
        # changed input's non-polymers off the ground input's numbers, so the
        # ground EDO's C1 at 101 and the changed W2S's C1 are two positions now.
        assert sums[("A", "101", "C1")] == 40
        assert sums[("A", "102", "C1")] == 60

    def test_a_position_only_one_input_models_keeps_its_partial_share(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        out = tmp_path / "merged.cif"
        assert _merge(runner, app, pair, out)[0] == 0
        sums = _position_sums(out)
        assert sums[("A", "101", "O2")] == 40  # ground EDO only
        assert sums[("A", "102", "N1")] == 60  # changed W2S only, renumbered

    def test_an_input_position_above_one_is_rejected(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ground, changed = pair
        # Push the SER 2 CA conformers to 0.50 / 0.51: a modelling error scaling
        # cannot repair, so it must stop the merge rather than be folded in.
        ground.write_text(ground.read_text().replace("5.100 1.200 1.000 0.50", "5.100 1.200 1.000 0.51"))
        code, output = _merge(runner, app, pair, tmp_path / "merged.cif")
        assert code == 1
        assert "chain A" in output
        assert "residue 2" in output
        assert "atom CA" in output

    def test_a_zero_occupancy_input_atom_is_reported_once_with_a_count(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ground, changed = pair
        ground.write_text(ground.read_text().replace("1.00 20.00 1   A 1", "0.00 20.00 1   A 1"))
        out = tmp_path / "merged.cif"
        result = runner.invoke(
            app,
            ["merge-states", "--ground", str(ground), "--changed", str(changed), "--occ", "0.60", "-o", str(out), "-y"],
        )
        assert result.exit_code == 0, result.output + result.stderr
        # Four ALA atoms are zeroed, and they are reported once between them with
        # a count rather than once each.
        assert result.stderr.count("zero occupancy") == 1
        assert "4 atom" in result.stderr
        assert _column(out, "_atom_site.occupancy").count("0.00") == 4


class TestTableSet:
    def test_only_the_allowed_categories_are_written(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        out = tmp_path / "merged.cif"
        assert _merge(runner, app, pair, out)[0] == 0
        assert _categories(out) == ALLOWED_CATEGORIES

    def test_chem_comp_is_the_union_of_both_inputs(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        out = tmp_path / "merged.cif"
        assert _merge(runner, app, pair, out)[0] == 0
        assert set(_column(out, "_chem_comp.id")) == {"ALA", "SER", "GLY", "EDO", "W2S"}

    def test_conflicting_chem_comp_formula_is_an_error(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ground, changed = pair
        changed.write_text(changed.read_text().replace("ALA 'C3 H7 N O2'", "ALA 'C9 H9 N O9'"))
        out = tmp_path / "merged.cif"
        result = runner.invoke(
            app,
            ["merge-states", "--ground", str(ground), "--changed", str(changed), "--occ", "0.60", "-o", str(out), "-y"],
        )
        assert result.exit_code == 1
        assert "ALA" in (result.output + result.stderr)

    def test_label_columns_are_regenerated_not_copied(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        out = tmp_path / "merged.cif"
        assert _merge(runner, app, pair, out)[0] == 0
        # Both inputs call their non-polymer label_asym_id B / label_entity_id 2
        # while meaning different components, so the merged file must carry
        # neither input's labels: one entity per distinct component, numbered.
        assert sorted(set(_column(out, "_atom_site.label_entity_id"))) == ["1", "2", "3"]
        assert len(set(_column(out, "_atom_site.label_asym_id"))) == 3

    def test_no_atom_references_an_undefined_entity_or_chain(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        out = tmp_path / "merged.cif"
        assert _merge(runner, app, pair, out)[0] == 0
        entities = set(_column(out, "_entity.id"))
        chains = set(_column(out, "_struct_asym.id"))
        assert set(_column(out, "_atom_site.label_entity_id")) <= entities
        assert set(_column(out, "_atom_site.label_asym_id")) <= chains
        # _struct_asym points at _entity in turn, so the chain of references has
        # to close there too.
        assert set(_column(out, "_struct_asym.entity_id")) <= entities

    def test_each_distinct_component_gets_its_own_entity(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        out = tmp_path / "merged.cif"
        assert _merge(runner, app, pair, out)[0] == 0
        # The ground-only EDO and the changed-only W2S must not share an entity,
        # which is exactly what copying one input's _entity table would do.
        comps = _column(out, "_atom_site.label_comp_id")
        entities = _column(out, "_atom_site.label_entity_id")
        by_comp = {comp: entity for comp, entity in zip(comps, entities, strict=True)}
        assert by_comp["EDO"] != by_comp["W2S"]
        assert _column(out, "_entity.type") == ["polymer", "non-polymer", "non-polymer"]

    def test_entry_id_and_block_name_are_the_output_stem(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        out = tmp_path / "chosen_name.cif"
        assert _merge(runner, app, pair, out)[0] == 0
        block = gemmi.cif.read(str(out)).sole_block()
        assert block.name == "chosen_name"
        assert gemmi.cif.as_string(block.find_value("_entry.id")) == "chosen_name"


class TestOutputPaths:
    def test_default_name_in_the_working_directory(
        self,
        runner: CliRunner,
        app: typer.Typer,
        pair: tuple[Path, Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ground, changed = pair
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        result = runner.invoke(
            app, ["merge-states", "--ground", str(ground), "--changed", str(changed), "--occ", "0.60", "-y"]
        )
        assert result.exit_code == 0, result.output + result.stderr
        assert (work / MERGED_NAME).exists()

    def test_output_option_overrides_the_path(
        self,
        runner: CliRunner,
        app: typer.Typer,
        pair: tuple[Path, Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        out = tmp_path / "elsewhere" / "merged.cif"
        out.parent.mkdir()
        assert _merge(runner, app, pair, out)[0] == 0
        assert out.exists()
        assert not (work / MERGED_NAME).exists()

    def test_existing_output_prompts_without_yes(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ground, changed = pair
        out = tmp_path / "merged.cif"
        out.write_text("placeholder\n")
        result = runner.invoke(
            app,
            ["merge-states", "--ground", str(ground), "--changed", str(changed), "--occ", "0.60", "-o", str(out)],
            input="n\n",
        )
        assert result.exit_code != 0
        assert out.read_text() == "placeholder\n"


class TestKeepIntermediates:
    """``--keep-intermediates``: each input as a standalone single-state file.

    An intermediate shows one model on its own, already wearing every label it
    will have in the merged output. Opening one in a viewer shows the model as it
    was refined; comparing one against the merged file traces any atom back to
    its source, which is what makes the label agreement asserted here the point
    of the whole flag.
    """

    def test_an_intermediate_is_written_for_each_input(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path, work: Path
    ) -> None:
        code, output = _merge(runner, app, pair, tmp_path / "merged.cif", "--keep-intermediates")
        assert code == 0, output
        assert (work / GROUND_INTERMEDIATE).exists()
        assert (work / CHANGED_INTERMEDIATE).exists()

    def test_intermediates_stay_in_the_working_directory_when_output_moves_the_merged_file(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path, work: Path
    ) -> None:
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        out = elsewhere / "merged.cif"
        assert _merge(runner, app, pair, out, "--keep-intermediates")[0] == 0
        # -o directs the primary output only; the intermediates are not dragged
        # along with it, so a run started here leaves its side files here.
        assert out.exists()
        assert not (elsewhere / GROUND_INTERMEDIATE).exists()
        assert (work / GROUND_INTERMEDIATE).exists()
        assert (work / CHANGED_INTERMEDIATE).exists()

    def test_nothing_extra_is_written_without_the_flag(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path, work: Path
    ) -> None:
        assert _merge(runner, app, pair, tmp_path / "merged.cif")[0] == 0
        assert list(work.iterdir()) == []

    def test_each_intermediate_is_rooted_at_its_own_state(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path, work: Path
    ) -> None:
        assert _merge(runner, app, pair, tmp_path / "merged.cif", "--keep-intermediates")[0] == 0
        # Deliberately not Base: individually each file *is* rooted at that state,
        # and a Base above it only means something in the context of the pair.
        ground_root = read_hierarchy(work / GROUND_INTERMEDIATE).get_root()
        changed_root = read_hierarchy(work / CHANGED_INTERMEDIATE).get_root()
        assert (ground_root.id, ground_root.name) == ("Ground", "ground_state")
        assert (changed_root.id, changed_root.name) == ("Changed", "changed_state")

    def test_no_base_state_appears_in_an_intermediate(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path, work: Path
    ) -> None:
        assert _merge(runner, app, pair, tmp_path / "merged.cif", "--keep-intermediates")[0] == 0
        for name in (GROUND_INTERMEDIATE, CHANGED_INTERMEDIATE):
            assert not read_hierarchy(work / name).contains("Base"), name

    def test_state_ids_and_names_match_the_merged_output(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path, work: Path
    ) -> None:
        out = tmp_path / "merged.cif"
        assert _merge(runner, app, pair, out, "--keep-intermediates")[0] == 0
        merged = read_hierarchy(out)
        for name, branch in ((GROUND_INTERMEDIATE, "Ground"), (CHANGED_INTERMEDIATE, "Changed")):
            tree = read_hierarchy(work / name)
            # The deconfliction of step 3 happens before the intermediates are
            # written, so the changed input's C/D are already C/D here.
            assert {(s.id, s.name) for s in tree.states} == {
                (s.id, s.name) for s in [merged.get_state(branch), *merged.get_descendants(branch)]
            }

    def test_per_atom_state_assignments_match_the_merged_output(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path, work: Path
    ) -> None:
        out = tmp_path / "merged.cif"
        assert _merge(runner, app, pair, out, "--keep-intermediates")[0] == 0
        merged = read_hierarchy(out)
        ground_ids = {"Ground"} | {s.id for s in merged.get_descendants("Ground")}
        assigned = read_atom_site_heterogeneity_ids(out)
        from_merged = [state for state in assigned if state in ground_ids]
        assert read_atom_site_heterogeneity_ids(work / GROUND_INTERMEDIATE) == from_merged

    def test_altloc_labels_match_the_merged_output(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path, work: Path
    ) -> None:
        out = tmp_path / "merged.cif"
        assert _merge(runner, app, pair, out, "--keep-intermediates")[0] == 0
        ground_labels, changed_labels = _labels_by_branch(out)
        assert set(_column(work / GROUND_INTERMEDIATE, "_atom_site.label_alt_id")) == ground_labels == {"A", "B", "C"}
        assert set(_column(work / CHANGED_INTERMEDIATE, "_atom_site.label_alt_id")) == changed_labels == {"D", "E", "F"}

    def test_non_polymer_numbering_matches_the_merged_output(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path, work: Path
    ) -> None:
        out = tmp_path / "merged.cif"
        assert _merge(runner, app, pair, out, "--keep-intermediates")[0] == 0
        ground_residues, changed_residues = _residues_by_branch(out)
        assert _residues(work / GROUND_INTERMEDIATE) == ground_residues
        # The renumbering of step 5 has already run, so the changed input's W2S
        # sits at 102 here exactly as it does in the merged file.
        assert _residues(work / CHANGED_INTERMEDIATE) == changed_residues
        assert ("A", 102, "W2S") in _residues(work / CHANGED_INTERMEDIATE)

    def test_occupancies_are_the_originals_unscaled(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path, work: Path
    ) -> None:
        assert _merge(runner, app, pair, tmp_path / "merged.cif", "--keep-intermediates")[0] == 0
        # Scaling is a property of the pair and meaningless in a file holding one
        # member of it, so the fixture's 1.00 and 0.50 survive rather than
        # becoming the merged file's 0.40 / 0.20 and 0.60 / 0.30.
        for name in (GROUND_INTERMEDIATE, CHANGED_INTERMEDIATE):
            assert set(_column(work / name, "_atom_site.occupancy")) == {"1.00", "0.50"}, name

    def test_the_merged_file_is_still_scaled(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path, work: Path
    ) -> None:
        out = tmp_path / "merged.cif"
        assert _merge(runner, app, pair, out, "--keep-intermediates")[0] == 0
        assert set(_column(out, "_atom_site.occupancy")) == {"0.40", "0.20", "0.60", "0.30"}

    def test_each_intermediate_holds_only_its_own_input_atoms(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path, work: Path
    ) -> None:
        assert _merge(runner, app, pair, tmp_path / "merged.cif", "--keep-intermediates")[0] == 0
        ground_comps = _column(work / GROUND_INTERMEDIATE, "_atom_site.label_comp_id")
        changed_comps = _column(work / CHANGED_INTERMEDIATE, "_atom_site.label_comp_id")
        assert len(ground_comps) == 19
        assert len(changed_comps) == 19
        assert "EDO" in ground_comps
        assert "W2S" not in ground_comps
        assert "W2S" in changed_comps
        assert "EDO" not in changed_comps

    def test_each_intermediate_passes_the_validate_command(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path, work: Path
    ) -> None:
        assert _merge(runner, app, pair, tmp_path / "merged.cif", "--keep-intermediates")[0] == 0
        for name in (GROUND_INTERMEDIATE, CHANGED_INTERMEDIATE):
            result = runner.invoke(app, ["validate", str(work / name)])
            assert result.exit_code == 0, name + ": " + result.output + result.stderr

    def test_each_intermediate_opens_as_a_standalone_model(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path, work: Path
    ) -> None:
        assert _merge(runner, app, pair, tmp_path / "merged.cif", "--keep-intermediates")[0] == 0
        for name in (GROUND_INTERMEDIATE, CHANGED_INTERMEDIATE):
            structure = gemmi.read_structure(str(work / name))
            assert len(structure) == 1
            assert sum(1 for chain in structure[0] for residue in chain for _ in residue) == 19
            # Nothing in the file may reference bookkeeping the file does not carry.
            path = work / name
            assert set(_column(path, "_atom_site.label_entity_id")) <= set(_column(path, "_entity.id"))
            assert set(_column(path, "_atom_site.label_asym_id")) <= set(_column(path, "_struct_asym.id"))

    def test_intermediates_carry_the_merged_files_table_set(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path, work: Path
    ) -> None:
        assert _merge(runner, app, pair, tmp_path / "merged.cif", "--keep-intermediates")[0] == 0
        for name in (GROUND_INTERMEDIATE, CHANGED_INTERMEDIATE):
            assert _categories(work / name) == ALLOWED_CATEGORIES, name

    def test_chem_comp_covers_only_the_inputs_own_components(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path, work: Path
    ) -> None:
        assert _merge(runner, app, pair, tmp_path / "merged.cif", "--keep-intermediates")[0] == 0
        # The union is the merged file's business; an intermediate defines what it
        # contains, so the other input's ligand is not declared here.
        assert set(_column(work / GROUND_INTERMEDIATE, "_chem_comp.id")) == {"ALA", "SER", "GLY", "EDO"}
        assert set(_column(work / CHANGED_INTERMEDIATE, "_chem_comp.id")) == {"ALA", "SER", "GLY", "W2S"}

    def test_entry_id_and_block_name_are_the_intermediates_own_stem(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path, work: Path
    ) -> None:
        assert _merge(runner, app, pair, tmp_path / "merged.cif", "--keep-intermediates")[0] == 0
        for name in (GROUND_INTERMEDIATE, CHANGED_INTERMEDIATE):
            block = gemmi.cif.read(str(work / name)).sole_block()
            # The input's own _entry.id would otherwise ride along on the clone and
            # name a different file than the block does.
            assert block.name == Path(name).stem
            assert gemmi.cif.as_string(block.find_value("_entry.id")) == Path(name).stem

    def test_every_written_path_is_printed(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path, work: Path
    ) -> None:
        out = tmp_path / "merged.cif"
        code, output = _merge(runner, app, pair, out, "--keep-intermediates")
        assert code == 0, output
        assert str(out) in output
        assert str(work / GROUND_INTERMEDIATE) in output
        assert str(work / CHANGED_INTERMEDIATE) in output

    def test_yes_covers_all_three_files(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path, work: Path
    ) -> None:
        out = tmp_path / "merged.cif"
        targets = [out, work / GROUND_INTERMEDIATE, work / CHANGED_INTERMEDIATE]
        for target in targets:
            target.write_text("placeholder\n")
        code, output = _merge(runner, app, pair, out, "--keep-intermediates")
        assert code == 0, output
        for target in targets:
            assert target.read_text() != "placeholder\n", target

    def test_one_confirmation_covers_all_three_files(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path, work: Path
    ) -> None:
        out = tmp_path / "merged.cif"
        targets = [out, work / GROUND_INTERMEDIATE, work / CHANGED_INTERMEDIATE]
        for target in targets:
            target.write_text("placeholder\n")
        code, output = _merge(runner, app, pair, out, "--keep-intermediates", yes=False, answer="y\n")
        assert code == 0, output
        # A run must not stop three times to ask about overwrites.
        assert output.count("Overwrite") == 1
        for target in targets:
            assert target.read_text() != "placeholder\n", target

    def test_declining_the_confirmation_writes_nothing(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path, work: Path
    ) -> None:
        out = tmp_path / "merged.cif"
        # Only an intermediate exists, so the prompt has to consider more than the
        # merged file to fire at all.
        existing = work / CHANGED_INTERMEDIATE
        existing.write_text("placeholder\n")
        code, _ = _merge(runner, app, pair, out, "--keep-intermediates", yes=False, answer="n\n")
        assert code != 0
        assert existing.read_text() == "placeholder\n"
        assert not out.exists()
        assert not (work / GROUND_INTERMEDIATE).exists()

    def test_two_inputs_sharing_a_stem_is_a_usage_error(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path, work: Path
    ) -> None:
        ground, changed = pair
        # Both intermediates would resolve to one path, and the second would
        # silently overwrite the first.
        twin = tmp_path / "twin"
        twin.mkdir()
        clash = twin / ground.name
        clash.write_text(changed.read_text())
        result = runner.invoke(
            app,
            [
                "merge-states",
                "--ground",
                str(ground),
                "--changed",
                str(clash),
                "--occ",
                "0.60",
                "-o",
                str(tmp_path / "merged.cif"),
                "--keep-intermediates",
                "-y",
            ],
        )
        assert result.exit_code == 2, result.output + result.stderr
        assert GROUND_INTERMEDIATE in (result.output + result.stderr)

    def test_a_failed_merge_leaves_no_intermediates_behind(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path, work: Path
    ) -> None:
        ground, changed = pair
        # The conflicting-formula check runs after the intermediates are built but
        # before anything is written, so a run that fails writes no files at all.
        changed.write_text(changed.read_text().replace("ALA 'C3 H7 N O2'", "ALA 'C9 H9 N O9'"))
        code, output = _merge(runner, app, pair, tmp_path / "merged.cif", "--keep-intermediates")
        assert code == 1
        assert "ALA" in output
        assert list(work.iterdir()) == []


class TestUsage:
    @pytest.mark.parametrize("occ", ["0", "1", "1.5", "-0.2", "0.605"])
    def test_occ_out_of_range_or_too_precise_is_a_usage_error(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path, occ: str
    ) -> None:
        ground, changed = pair
        result = runner.invoke(
            app,
            [
                "merge-states",
                "--ground",
                str(ground),
                "--changed",
                str(changed),
                "--occ",
                occ,
                "-o",
                str(tmp_path / "merged.cif"),
                "-y",
            ],
        )
        assert result.exit_code == 2, result.output + result.stderr

    @pytest.mark.parametrize("occ", ["0.01", "0.5", "0.99"])
    def test_occ_in_range_is_accepted(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path, occ: str
    ) -> None:
        ground, changed = pair
        result = runner.invoke(
            app,
            [
                "merge-states",
                "--ground",
                str(ground),
                "--changed",
                str(changed),
                "--occ",
                occ,
                "-o",
                str(tmp_path / "merged.cif"),
                "-y",
            ],
        )
        assert result.exit_code == 0, result.output + result.stderr

    def test_merge_states_is_a_top_level_command(self, runner: CliRunner, app: typer.Typer) -> None:
        result = runner.invoke(app, ["--help"])
        assert "merge-states" in result.output

    def test_missing_input_is_an_error(
        self, runner: CliRunner, app: typer.Typer, pair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        _, changed = pair
        result = runner.invoke(
            app,
            [
                "merge-states",
                "--ground",
                str(tmp_path / "nope.cif"),
                "--changed",
                str(changed),
                "--occ",
                "0.60",
                "-o",
                str(tmp_path / "merged.cif"),
                "-y",
            ],
        )
        assert result.exit_code != 0

    def test_input_without_atoms_is_a_parse_error(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path], tmp_path: Path
    ) -> None:
        atomless = copy_fixture("hierarchy_only.cif")
        changed = copy_fixture("merge_changed.cif")
        result = runner.invoke(
            app,
            [
                "merge-states",
                "--ground",
                str(atomless),
                "--changed",
                str(changed),
                "--occ",
                "0.60",
                "-o",
                str(tmp_path / "merged.cif"),
                "-y",
            ],
        )
        assert result.exit_code == 1
        assert "no atoms" in (result.output + result.stderr)
