"""Tests for the ``show`` command."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import typer
from typer.testing import CliRunner


class TestShow:
    def test_default_lists_all_sections(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path]
    ) -> None:
        cif = copy_fixture("full_hierarchy.cif")
        result = runner.invoke(app, ["show", str(cif)])
        assert result.exit_code == 0
        assert "Base" in result.output
        assert "NOT" in result.output  # coexistence rule
        assert "Atom assignments" in result.output

    def test_tree_view(self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path]) -> None:
        cif = copy_fixture("full_hierarchy.cif")
        result = runner.invoke(app, ["show", str(cif), "--tree"])
        assert result.exit_code == 0
        assert "Base (base_state)" in result.output
        assert "  A (state_a)" in result.output

    def test_json_output(self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path]) -> None:
        cif = copy_fixture("full_hierarchy.cif")
        result = runner.invoke(app, ["show", str(cif), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["hierarchy"]["states"][0]["id"] == "Base"
        assert data["assignments"] == ["Base", "A"]

    def test_hierarchy_only_narrows_output(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path]
    ) -> None:
        cif = copy_fixture("full_hierarchy.cif")
        result = runner.invoke(app, ["show", str(cif), "--hierarchy"])
        assert result.exit_code == 0
        assert "Atom assignments" not in result.output

    def test_missing_hierarchy(self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path]) -> None:
        cif = copy_fixture("atom_site_no_hierarchy.cif")
        result = runner.invoke(app, ["show", str(cif)])
        assert result.exit_code == 0
        assert "No hierarchy table found." in result.output
