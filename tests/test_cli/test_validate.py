"""Tests for the ``validate`` command."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import typer
from typer.testing import CliRunner


class TestValidate:
    def test_valid_file_passes(self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path]) -> None:
        cif = copy_fixture("full_hierarchy.cif")
        result = runner.invoke(app, ["validate", str(cif)])
        assert result.exit_code == 0
        assert "Validation passed." in result.output

    def test_bad_references_fail(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path]
    ) -> None:
        cif = copy_fixture("bad_atom_site_refs.cif")
        result = runner.invoke(app, ["validate", str(cif)])
        assert result.exit_code == 1

    def test_strict_fails_on_bad_references(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path]
    ) -> None:
        cif = copy_fixture("bad_atom_site_refs.cif")
        result = runner.invoke(app, ["validate", str(cif), "--strict"])
        assert result.exit_code == 1

    def test_no_hierarchy_is_noop(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path]
    ) -> None:
        cif = copy_fixture("atom_site_no_hierarchy.cif")
        result = runner.invoke(app, ["validate", str(cif)])
        assert result.exit_code == 0
        assert "nothing to validate" in result.output
