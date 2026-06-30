"""Shared fixtures for CLI tests."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from pdbx_hierarchy.cli.app import app as cli_app

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def app() -> typer.Typer:
    return cli_app


@pytest.fixture
def copy_fixture(tmp_path: Path) -> Callable[[str], Path]:
    """Return a function copying a named fixture into tmp_path and returning its path."""

    def _copy(name: str) -> Path:
        dest = tmp_path / name
        shutil.copy(FIXTURES_DIR / name, dest)
        return dest

    return _copy
