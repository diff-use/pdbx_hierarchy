"""Tests for the ``coexist`` sub-app (add, remove)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import typer
from typer.testing import CliRunner

from pdbx_hierarchy.io.reader import read_coexistence


class TestCoexistAdd:
    def test_add_rule(self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path]) -> None:
        cif = copy_fixture("full_hierarchy.cif")
        out = cif.with_name("out.cif")
        result = runner.invoke(
            app, ["coexist", "add", str(cif), "--rule", "OR", "--state", "Base", "--related", "A", "-o", str(out)]
        )
        assert result.exit_code == 0
        table = read_coexistence(out)
        assert table is not None
        assert any(rule.rule.value == "OR" for rule in table.rules)

    def test_add_unknown_reference_fails(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path]
    ) -> None:
        cif = copy_fixture("full_hierarchy.cif")
        result = runner.invoke(
            app,
            ["coexist", "add", str(cif), "--rule", "OR", "--state", "Base", "--related", "Z", "-o", str(cif), "-y"],
        )
        assert result.exit_code == 1

    def test_invalid_rule_choice_fails(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path]
    ) -> None:
        cif = copy_fixture("full_hierarchy.cif")
        result = runner.invoke(
            app, ["coexist", "add", str(cif), "--rule", "MAYBE", "--state", "Base", "--related", "A", "-o", str(cif)]
        )
        assert result.exit_code != 0

    def test_self_reference_fails(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path]
    ) -> None:
        cif = copy_fixture("full_hierarchy.cif")
        result = runner.invoke(
            app,
            ["coexist", "add", str(cif), "--rule", "OR", "--state", "A", "--related", "A,Base", "-o", str(cif), "-y"],
        )
        assert result.exit_code == 1
        assert "self-reference" in (result.output + result.stderr)

    def test_duplicate_related_is_deduped(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path]
    ) -> None:
        cif = copy_fixture("full_hierarchy.cif")
        out = cif.with_name("out.cif")
        result = runner.invoke(
            app, ["coexist", "add", str(cif), "--rule", "OR", "--state", "Base", "--related", "A,A", "-o", str(out)]
        )
        assert result.exit_code == 0, result.output
        table = read_coexistence(out)
        assert table is not None
        added = max(table.rules, key=lambda r: r.id)
        assert added.heterogeneity_ids == ["A"]

    def test_add_to_invalid_file_fails(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path]
    ) -> None:
        cif = copy_fixture("bad_atom_site_refs.cif")
        result = runner.invoke(
            app, ["coexist", "add", str(cif), "--rule", "OR", "--state", "Base", "--related", "A", "-o", str(cif), "-y"]
        )
        assert result.exit_code == 1


class TestCoexistRemove:
    def test_remove_rule(self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path]) -> None:
        cif = copy_fixture("full_hierarchy.cif")
        out = cif.with_name("out.cif")
        result = runner.invoke(app, ["coexist", "remove", str(cif), "--id", "1", "-o", str(out)])
        assert result.exit_code == 0
        table = read_coexistence(out)
        assert table is None or len(table) == 0

    def test_remove_unknown_id_fails(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path]
    ) -> None:
        cif = copy_fixture("full_hierarchy.cif")
        result = runner.invoke(app, ["coexist", "remove", str(cif), "--id", "99", "-o", str(cif), "-y"])
        assert result.exit_code == 1

    def test_remove_without_table_fails(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path]
    ) -> None:
        cif = copy_fixture("hierarchy_only.cif")
        result = runner.invoke(app, ["coexist", "remove", str(cif), "--id", "1", "-o", str(cif), "-y"])
        assert result.exit_code == 1
