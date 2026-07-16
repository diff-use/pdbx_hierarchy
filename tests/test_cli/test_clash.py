"""Tests for the ``clash`` sub-app (detect, apply)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import typer
from typer.testing import CliRunner

from pdbx_hierarchy.clash import ClashReport
from pdbx_hierarchy.io.reader import read_coexistence, read_hierarchy


class TestClashDetect:
    def test_detect_prints_summary(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path]
    ) -> None:
        cif = copy_fixture("clashing_states.cif")
        result = runner.invoke(app, ["clash", "detect", str(cif)])
        assert result.exit_code == 0, result.output
        assert "actionable" in result.output
        assert "Proposed merges" in result.output

    def test_detect_writes_report(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path], tmp_path: Path
    ) -> None:
        cif = copy_fixture("clashing_states.cif")
        report = tmp_path / "clashes.json"
        result = runner.invoke(app, ["clash", "detect", str(cif), "--report", str(report)])
        assert result.exit_code == 0, result.output
        assert report.exists()
        parsed = ClashReport.from_json(report.read_text())
        assert [sorted(m.states) for m in parsed.merges] == [["A", "D"], ["B", "C"]]

    def test_detect_not_only(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path], tmp_path: Path
    ) -> None:
        cif = copy_fixture("clashing_states.cif")
        report = tmp_path / "clashes.json"
        result = runner.invoke(app, ["clash", "detect", str(cif), "--report", str(report), "--not-only"])
        assert result.exit_code == 0, result.output
        parsed = ClashReport.from_json(report.read_text())
        assert parsed.merges == []
        assert {frozenset(n.states) for n in parsed.not_rules} == {frozenset({"A", "C"}), frozenset({"E", "H"})}

    def test_detect_requires_hierarchy(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path]
    ) -> None:
        cif = copy_fixture("atom_site_no_hierarchy.cif")
        result = runner.invoke(app, ["clash", "detect", str(cif)])
        assert result.exit_code == 1


class TestClashApply:
    def _detect(self, runner: CliRunner, app: typer.Typer, cif: Path, report: Path, *args: str) -> None:
        result = runner.invoke(app, ["clash", "detect", str(cif), "--report", str(report), *args])
        assert result.exit_code == 0, result.output

    def test_apply_merges_round_trip(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path], tmp_path: Path
    ) -> None:
        cif = copy_fixture("clashing_states.cif")
        report = tmp_path / "clashes.json"
        out = tmp_path / "out.cif"
        self._detect(runner, app, cif, report)

        result = runner.invoke(app, ["clash", "apply", str(cif), "--report", str(report), "-o", str(out), "-y"])
        assert result.exit_code == 0, result.output

        tree = read_hierarchy(out)
        # Two merge groups (A+D, B+C) remove two states from the original 13.
        assert len(tree) == 11
        assert not tree.contains("D")
        assert not tree.contains("C")
        assert tree.contains("A")
        assert tree.contains("B")

        # The file still round-trips through validation.
        validate = runner.invoke(app, ["validate", str(out)])
        assert validate.exit_code == 0, validate.output

    def test_apply_not_rules(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path], tmp_path: Path
    ) -> None:
        cif = copy_fixture("clashing_states.cif")
        report = tmp_path / "clashes.json"
        out = tmp_path / "out.cif"
        self._detect(runner, app, cif, report, "--not-only")

        result = runner.invoke(app, ["clash", "apply", str(cif), "--report", str(report), "-o", str(out), "-y"])
        assert result.exit_code == 0, result.output
        table = read_coexistence(out)
        assert table is not None
        not_pairs = {(r.heterogeneity_id, tuple(r.heterogeneity_ids)) for r in table.rules if r.rule.value == "NOT"}
        assert ("A", ("C",)) in not_pairs
        assert ("E", ("H",)) in not_pairs

    def test_apply_respects_disabled_action(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path], tmp_path: Path
    ) -> None:
        cif = copy_fixture("clashing_states.cif")
        report = tmp_path / "clashes.json"
        out = tmp_path / "out.cif"
        self._detect(runner, app, cif, report)

        # Disable the A+D merge; keep B+C.
        data = json.loads(report.read_text())
        for merge in data["merges"]:
            if sorted(merge["states"]) == ["A", "D"]:
                merge["enabled"] = False
        report.write_text(json.dumps(data))

        result = runner.invoke(app, ["clash", "apply", str(cif), "--report", str(report), "-o", str(out), "-y"])
        assert result.exit_code == 0, result.output
        tree = read_hierarchy(out)
        # A+D merge skipped, so D survives; B+C still merged, so C is gone.
        assert tree.contains("D")
        assert not tree.contains("C")

    def test_apply_rejects_missing_state(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path], tmp_path: Path
    ) -> None:
        cif = copy_fixture("clashing_states.cif")
        report = tmp_path / "clashes.json"
        self._detect(runner, app, cif, report)

        data = json.loads(report.read_text())
        data["merges"].append({"action": "merge", "states": ["Zzz", "A"], "enabled": True})
        report.write_text(json.dumps(data))

        result = runner.invoke(
            app, ["clash", "apply", str(cif), "--report", str(report), "-o", str(tmp_path / "o.cif"), "-y"]
        )
        assert result.exit_code == 1
        assert "not in the hierarchy" in result.output

    def test_apply_rejects_bad_version(
        self, runner: CliRunner, app: typer.Typer, copy_fixture: Callable[[str], Path], tmp_path: Path
    ) -> None:
        cif = copy_fixture("clashing_states.cif")
        report = tmp_path / "clashes.json"
        report.write_text('{"version": 999, "clashes": [], "merges": [], "not_rules": [], "conflicts": []}')
        result = runner.invoke(
            app, ["clash", "apply", str(cif), "--report", str(report), "-o", str(tmp_path / "o.cif"), "-y"]
        )
        assert result.exit_code == 1
