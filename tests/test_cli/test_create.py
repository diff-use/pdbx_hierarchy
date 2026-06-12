"""Tests for the ``infer`` and ``import`` commands."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import typer
from typer.testing import CliRunner

from pdbx_hierarchy.io.reader import read_hierarchy
from pdbx_hierarchy.models.hierarchy import HierarchyState, HierarchyTree


class TestInfer:
    def test_infer_single_state(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path], tmp_path: Path
    ) -> None:
        cif = copy_fixture("atom_site_no_hierarchy.cif")
        out = tmp_path / "out.cif"
        result = runner.invoke(app, ["infer", str(cif), "-o", str(out), "-y"])
        assert result.exit_code == 0
        assert out.exists()
        assert len(read_hierarchy(out)) == 1  # all "." -> Base only

    def test_infer_multiple_states(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path], tmp_path: Path
    ) -> None:
        cif = copy_fixture("assign_single_residue_sidechain.cif")
        out = tmp_path / "out.cif"
        result = runner.invoke(app, ["infer", str(cif), "-o", str(out), "-y"])
        assert result.exit_code == 0
        assert "Inferred 3 state(s)" in result.output

    def test_infer_auto_suffix(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path], tmp_path: Path
    ) -> None:
        cif = copy_fixture("atom_site_no_hierarchy.cif")
        result = runner.invoke(app, ["infer", str(cif)])
        assert result.exit_code == 0
        assert (tmp_path / "atom_site_no_hierarchy_pdbx_1.cif").exists()

    def test_infer_in_place_prompt_declined(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path]
    ) -> None:
        cif = copy_fixture("atom_site_no_hierarchy.cif")
        result = runner.invoke(app, ["infer", str(cif), "-o", str(cif)], input="n\n")
        assert result.exit_code != 0  # aborted

    def test_infer_warns_when_hierarchy_exists(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path], tmp_path: Path
    ) -> None:
        cif = copy_fixture("assign_single_residue_sidechain.cif")
        first = tmp_path / "first.cif"
        assert runner.invoke(app, ["infer", str(cif), "-o", str(first), "-y"]).exit_code == 0
        # Inferring again over a file that already has a hierarchy must warn before replacing it.
        out = tmp_path / "second.cif"
        result = runner.invoke(app, ["infer", str(first), "-o", str(out), "-y"])
        assert result.exit_code == 0
        assert "already contains a hierarchy" in (result.output + result.stderr)


class TestImport:
    def test_import_spec(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path], tmp_path: Path
    ) -> None:
        cif = copy_fixture("atom_site_no_hierarchy.cif")
        spec = tmp_path / "h.json"
        spec.write_text(
            HierarchyTree(
                states=[
                    HierarchyState(id="Base", name="base_state", parent=None),
                    HierarchyState(id="A", name="state_a", parent="Base"),
                ]
            ).model_dump_json()
        )
        out = tmp_path / "out.cif"
        result = runner.invoke(app, ["import", str(cif), "--spec", str(spec), "-o", str(out), "-y"])
        assert result.exit_code == 0
        assert {s.id for s in read_hierarchy(out).states} == {"Base", "A"}

    def test_import_bad_spec(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path], tmp_path: Path
    ) -> None:
        cif = copy_fixture("atom_site_no_hierarchy.cif")
        spec = tmp_path / "h.json"
        spec.write_text('{"states": [{"id": "A", "name": "orphan", "parent": "Base"}]}')  # no root
        out = tmp_path / "out.cif"
        result = runner.invoke(app, ["import", str(cif), "--spec", str(spec), "-o", str(out), "-y"])
        assert result.exit_code == 1

    def test_import_warns_on_dangling_assignments(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path], tmp_path: Path
    ) -> None:
        # full_hierarchy.cif assigns atoms to Base and A; importing a spec without A
        # leaves those atoms dangling. The command should warn but still write.
        cif = copy_fixture("full_hierarchy.cif")
        spec = tmp_path / "h.json"
        base_only = HierarchyTree(states=[HierarchyState(id="Base", name="base_state", parent=None)])
        spec.write_text(base_only.model_dump_json())
        out = tmp_path / "out.cif"
        result = runner.invoke(app, ["import", str(cif), "--spec", str(spec), "-o", str(out), "-y"])
        assert result.exit_code == 0, result.output
        assert "not found in hierarchy" in (result.output + result.stderr)
        assert out.exists()
