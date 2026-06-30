"""Tests for the ``hierarchy`` sub-app (add, remove, rename, reparent, merge, split, reassign)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import typer
from typer.testing import CliRunner

from pdbx_hierarchy.io.reader import read_atom_site_heterogeneity_ids, read_hierarchy
from pdbx_hierarchy.models.hierarchy import HierarchyState, HierarchyTree


def _infer(runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path], name: str, tmp_path: Path) -> Path:
    """Infer a hierarchy from a fixture into a fresh file and return its path."""
    cif = copy_fixture(name)
    out = tmp_path / "inferred.cif"
    result = runner.invoke(app, ["infer", str(cif), "-o", str(out), "-y"])
    assert result.exit_code == 0, result.output
    return out


class TestAdd:
    def test_add_state(self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path]) -> None:
        cif = copy_fixture("full_hierarchy.cif")
        out = cif.with_name("out.cif")
        result = runner.invoke(
            app, ["hierarchy", "add", str(cif), "--id", "C", "--name", "state_c", "--parent", "Base", "-o", str(out)]
        )
        assert result.exit_code == 0
        assert "C" in {s.id for s in read_hierarchy(out).states}

    def test_add_unknown_parent_fails(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path]
    ) -> None:
        cif = copy_fixture("full_hierarchy.cif")
        result = runner.invoke(
            app, ["hierarchy", "add", str(cif), "--id", "C", "--name", "state_c", "--parent", "Z", "-o", str(cif), "-y"]
        )
        assert result.exit_code == 1


class TestRemove:
    def test_remove_folds_atoms_into_parent(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path]
    ) -> None:
        cif = copy_fixture("cli_multichain_state.cif")
        out = cif.with_name("out.cif")
        result = runner.invoke(app, ["hierarchy", "remove", str(cif), "--id", "A", "-o", str(out)])
        assert result.exit_code == 0
        assert {s.id for s in read_hierarchy(out).states} == {"Base"}
        assert set(read_atom_site_heterogeneity_ids(out)) == {"Base"}

    def test_remove_root_fails(self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path]) -> None:
        cif = copy_fixture("cli_multichain_state.cif")
        result = runner.invoke(app, ["hierarchy", "remove", str(cif), "--id", "Base", "-o", str(cif), "-y"])
        assert result.exit_code == 1


class TestRename:
    def test_rename(self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path]) -> None:
        cif = copy_fixture("full_hierarchy.cif")
        out = cif.with_name("out.cif")
        result = runner.invoke(
            app, ["hierarchy", "rename", str(cif), "--id", "A", "--name", "renamed_a", "-o", str(out)]
        )
        assert result.exit_code == 0
        assert read_hierarchy(out).get_state("A").name == "renamed_a"


def _import_tree(
    runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path], tmp_path: Path, tree: HierarchyTree
) -> Path:
    """Write ``tree`` onto a plain fixture and return the resulting file path."""
    cif = copy_fixture("atom_site_no_hierarchy.cif")
    spec = tmp_path / "spec.json"
    spec.write_text(tree.model_dump_json())
    out = tmp_path / "tree.cif"
    result = runner.invoke(app, ["import", str(cif), "--spec", str(spec), "-o", str(out), "-y"])
    assert result.exit_code == 0, result.output
    return out


class TestReparent:
    @staticmethod
    def _four_states() -> HierarchyTree:
        # Base ─ A ─ C  and  Base ─ B
        return HierarchyTree(
            states=[
                HierarchyState(id="Base", name="base_state", parent=None),
                HierarchyState(id="A", name="state_a", parent="Base"),
                HierarchyState(id="B", name="state_b", parent="Base"),
                HierarchyState(id="C", name="state_c", parent="A"),
            ]
        )

    def test_reparent_moves_subtree(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path], tmp_path: Path
    ) -> None:
        cif = _import_tree(runner, app, copy_fixture, tmp_path, self._four_states())
        out = tmp_path / "out.cif"
        result = runner.invoke(app, ["hierarchy", "reparent", str(cif), "--id", "C", "--parent", "B", "-o", str(out)])
        assert result.exit_code == 0, result.output
        assert read_hierarchy(out).get_state("C").parent == "B"

    def test_reparent_root_fails(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path], tmp_path: Path
    ) -> None:
        cif = _import_tree(runner, app, copy_fixture, tmp_path, self._four_states())
        args = ["hierarchy", "reparent", str(cif), "--id", "Base", "--parent", "A", "-o", str(cif), "-y"]
        result = runner.invoke(app, args)
        assert result.exit_code == 1
        assert "root" in (result.output + result.stderr)

    def test_reparent_unknown_parent_fails(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path], tmp_path: Path
    ) -> None:
        cif = _import_tree(runner, app, copy_fixture, tmp_path, self._four_states())
        result = runner.invoke(
            app, ["hierarchy", "reparent", str(cif), "--id", "C", "--parent", "Z", "-o", str(cif), "-y"]
        )
        assert result.exit_code == 1
        assert "unknown parent" in (result.output + result.stderr)

    def test_reparent_cycle_fails(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path], tmp_path: Path
    ) -> None:
        # Moving A beneath its own descendant C would create a cycle.
        cif = _import_tree(runner, app, copy_fixture, tmp_path, self._four_states())
        result = runner.invoke(
            app, ["hierarchy", "reparent", str(cif), "--id", "A", "--parent", "C", "-o", str(cif), "-y"]
        )
        assert result.exit_code == 1


class TestMerge:
    def test_merge_absorbs_atoms(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path], tmp_path: Path
    ) -> None:
        cif = _infer(runner, app, copy_fixture, "assign_backbone_merge.cif", tmp_path)
        out = tmp_path / "merged.cif"
        result = runner.invoke(app, ["hierarchy", "merge", str(cif), "--ids", "A,B", "-o", str(out)])
        assert result.exit_code == 0
        assert set(read_atom_site_heterogeneity_ids(out)) == {"A"}
        assert {s.id for s in read_hierarchy(out).states} == {"Base", "A"}

    def test_merge_requires_two_ids(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path], tmp_path: Path
    ) -> None:
        cif = _infer(runner, app, copy_fixture, "assign_backbone_merge.cif", tmp_path)
        result = runner.invoke(app, ["hierarchy", "merge", str(cif), "--ids", "A", "-o", str(cif), "-y"])
        assert result.exit_code == 1

    def test_merge_duplicate_ids_fails(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path]
    ) -> None:
        cif = copy_fixture("full_hierarchy.cif")
        result = runner.invoke(app, ["hierarchy", "merge", str(cif), "--ids", "A,A", "-o", str(cif), "-y"])
        assert result.exit_code == 1
        assert "distinct ids" in (result.output + result.stderr)


class TestSplit:
    def test_split_single_chain(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path], tmp_path: Path
    ) -> None:
        cif = _infer(runner, app, copy_fixture, "assign_backbone_merge.cif", tmp_path)
        out = tmp_path / "split.cif"
        result = runner.invoke(
            app, ["hierarchy", "split", str(cif), "--id", "A", "--select-a", "1", "--select-b", "2", "-o", str(out)]
        )
        assert result.exit_code == 0, result.output
        tree = read_hierarchy(out)
        assert len(tree.get_children("A")) == 2
        runner.invoke(app, ["validate", str(out)])  # smoke

    def test_split_multichain_ambiguous(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path]
    ) -> None:
        cif = copy_fixture("cli_multichain_state.cif")
        result = runner.invoke(
            app,
            ["hierarchy", "split", str(cif), "--id", "A", "--select-a", "1", "--select-b", "2", "-o", str(cif), "-y"],
        )
        assert result.exit_code == 1
        assert "Ambiguous" in (result.output + result.stderr)

    def test_split_chain_qualified(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path]
    ) -> None:
        cif = copy_fixture("cli_multichain_state.cif")
        out = cif.with_name("out.cif")
        result = runner.invoke(
            app,
            [
                "hierarchy",
                "split",
                str(cif),
                "--id",
                "A",
                "--select-a",
                "A/1,A/2",
                "--select-b",
                "B/1,B/2",
                "-o",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert len(read_hierarchy(out).get_children("A")) == 2

    def test_split_auth_numbering(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path]
    ) -> None:
        cif = copy_fixture("cli_multichain_state.cif")
        out = cif.with_name("out.cif")
        result = runner.invoke(
            app,
            [
                "hierarchy", "split", str(cif), "--id", "A",
                "--select-a", "X/101,X/102", "--select-b", "Y/201,Y/202", "--auth", "-o", str(out),
            ],
        )  # fmt: skip
        assert result.exit_code == 0, result.output

    def test_split_no_match_fails(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path]
    ) -> None:
        cif = copy_fixture("cli_multichain_state.cif")
        result = runner.invoke(
            app,
            [
                "hierarchy",
                "split",
                str(cif),
                "--id",
                "A",
                "--select-a",
                "A/9",
                "--select-b",
                "B/9",
                "-o",
                str(cif),
                "-y",
            ],
        )
        assert result.exit_code == 1

    def test_split_warns_on_absent_residue(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path]
    ) -> None:
        cif = copy_fixture("cli_multichain_state.cif")
        out = cif.with_name("out.cif")
        result = runner.invoke(
            app,
            ["hierarchy", "split", str(cif), "--id", "A", "--select-a", "A/1,A/9", "--select-b", "B/1", "-o", str(out)],
        )
        assert result.exit_code == 0, result.output
        assert "Warning" in (result.output + result.stderr)

    def test_split_warns_on_unselected_residue(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path]
    ) -> None:
        cif = copy_fixture("cli_multichain_state.cif")
        out = cif.with_name("out.cif")
        # Cover only chain A; chain B's residues are matched by neither selection.
        result = runner.invoke(
            app,
            ["hierarchy", "split", str(cif), "--id", "A", "--select-a", "A/1", "--select-b", "A/2", "-o", str(out)],
        )
        assert result.exit_code == 0, result.output
        assert "neither selection" in (result.output + result.stderr)


def _write_named_hierarchy(runner: CliRunner, app: typer.Typer, src: Path, tmp_path: Path) -> Path:
    """Import a hierarchy that includes a non-canonical id (``M1``) and return the file."""
    spec = tmp_path / "named.json"
    spec.write_text(
        HierarchyTree(
            states=[
                HierarchyState(id="Base", name="base_state", parent=None),
                HierarchyState(id="M1", name="named", parent="Base"),
                HierarchyState(id="B", name="state_b", parent="Base"),
            ]
        ).model_dump_json()
    )
    out = tmp_path / "named.cif"
    result = runner.invoke(app, ["import", str(src), "--spec", str(spec), "-o", str(out), "-y"])
    assert result.exit_code == 0, result.output
    return out


class TestReassign:
    def test_canonical_reassign(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path], tmp_path: Path
    ) -> None:
        src = copy_fixture("atom_site_no_hierarchy.cif")
        cif = _write_named_hierarchy(runner, app, src, tmp_path)
        out = tmp_path / "reassigned.cif"
        result = runner.invoke(app, ["hierarchy", "reassign", str(cif), "-o", str(out)])
        assert result.exit_code == 0
        assert {s.id for s in read_hierarchy(out).states} == {"Base", "A", "B"}

    def test_preserve_named(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path], tmp_path: Path
    ) -> None:
        src = copy_fixture("atom_site_no_hierarchy.cif")
        cif = _write_named_hierarchy(runner, app, src, tmp_path)
        out = tmp_path / "reassigned.cif"
        result = runner.invoke(app, ["hierarchy", "reassign", str(cif), "--preserve-named", "-o", str(out)])
        assert result.exit_code == 0
        ids = {s.id for s in read_hierarchy(out).states}
        assert "M1" in ids  # non-canonical id preserved
