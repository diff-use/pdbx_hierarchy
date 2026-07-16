"""Reader functions for PDBx/mmCIF files with hierarchy extensions."""

from __future__ import annotations

from pathlib import Path

import gemmi

from pdbx_hierarchy.exceptions import HierarchyNotFoundError, PdbxParseError
from pdbx_hierarchy.models.coexistence import CoexistenceTable
from pdbx_hierarchy.models.hierarchy import HierarchyTree


def _unquote(raw: str) -> str:
    """Return the string value of a raw mmCIF token, keeping ``.``/``?`` nulls.

    gemmi returns loop cells verbatim, so a quoted value arrives with its quotes
    (e.g. ``"'an alternate loop'"``). This strips the quoting via gemmi while
    leaving the bare null markers ``.`` and ``?`` intact so downstream null
    handling still recognizes them.

    Args:
        raw: The raw token as returned by gemmi.

    Returns:
        The unquoted string value, or the bare null marker unchanged.
    """
    if raw in (".", "?"):
        return raw
    return gemmi.cif.as_string(raw)


def _resolve_block(source: Path | gemmi.cif.Block) -> gemmi.cif.Block:
    """Return a gemmi Block from either a file path or an existing Block.

    Args:
        source: Path to a mmCIF file, or an already-loaded Block.

    Returns:
        The resolved gemmi Block.

    Raises:
        FileNotFoundError: If source is a Path and the file does not exist.
        PdbxParseError: If source is a Path and the file is malformed or contains
            more than one data block.
    """
    if isinstance(source, gemmi.cif.Block):
        return source
    try:
        doc = gemmi.cif.read(str(source))
    except FileNotFoundError:
        raise
    except ValueError as exc:
        raise PdbxParseError(str(exc)) from exc
    try:
        return doc.sole_block()
    except (RuntimeError, IndexError) as exc:
        raise PdbxParseError(f"Expected exactly one data block in {source}: {exc}") from exc


def read_mmcif(path: Path) -> gemmi.cif.Block:
    """Read a PDBx/mmCIF file, requiring exactly one data block.

    Args:
        path: Path to the mmCIF file.

    Returns:
        The sole data block from the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        PdbxParseError: If the file is malformed or has more than one data block.
    """
    return _resolve_block(path)


def has_hierarchy(source: Path | gemmi.cif.Block) -> bool:
    """Return True if the source contains a _pdbx_heterogeneity_hierarchy loop.

    Args:
        source: Path to a mmCIF file, or an already-loaded Block.

    Returns:
        True if the hierarchy table is present.
    """
    block = _resolve_block(source)
    return bool(block.find_loop("_pdbx_heterogeneity_hierarchy.id"))


def read_hierarchy(source: Path | gemmi.cif.Block) -> HierarchyTree:
    """Parse the _pdbx_heterogeneity_hierarchy loop into a HierarchyTree.

    Args:
        source: Path to a mmCIF file, or an already-loaded Block.

    Returns:
        A validated HierarchyTree.

    Raises:
        FileNotFoundError: If source is a Path and the file does not exist.
        PdbxParseError: If the file is malformed.
        HierarchyNotFoundError: If no _pdbx_heterogeneity_hierarchy table is present.
        InvalidHierarchyError: If the table data forms an invalid tree.
    """
    block = _resolve_block(source)
    tbl = block.find("_pdbx_heterogeneity_hierarchy.", ["id", "name", "parent", "details"])
    if not tbl:
        raise HierarchyNotFoundError("No _pdbx_heterogeneity_hierarchy table found")
    rows = [
        {"id": _unquote(row[0]), "name": _unquote(row[1]), "parent": _unquote(row[2]), "details": _unquote(row[3])}
        for row in tbl
    ]
    return HierarchyTree.from_mmcif_rows(rows)


def read_coexistence(source: Path | gemmi.cif.Block) -> CoexistenceTable | None:
    """Parse the _pdbx_state_coexistence loop, or return None if absent.

    Args:
        source: Path to a mmCIF file, or an already-loaded Block.

    Returns:
        A CoexistenceTable, or None if no coexistence table is present.

    Raises:
        FileNotFoundError: If source is a Path and the file does not exist.
        PdbxParseError: If the file is malformed.
    """
    block = _resolve_block(source)
    tbl = block.find(
        "_pdbx_state_coexistence.",
        ["id", "rule", "heterogeneity_id", "heterogeneity_ids", "description"],
    )
    if not tbl:
        return None
    rows = [
        {
            "id": _unquote(row[0]),
            "rule": _unquote(row[1]),
            "heterogeneity_id": _unquote(row[2]),
            "heterogeneity_ids": _unquote(row[3]),
            "description": _unquote(row[4]),
        }
        for row in tbl
    ]
    return CoexistenceTable.from_mmcif_rows(rows)


def read_atom_site_heterogeneity_ids(source: Path | gemmi.cif.Block) -> list[str]:
    """Return the _atom_site.pdbx_heterogeneity_id column as a list.

    Args:
        source: Path to a mmCIF file, or an already-loaded Block.

    Returns:
        List of heterogeneity ID strings, one per atom_site row.

    Raises:
        FileNotFoundError: If source is a Path and the file does not exist.
        PdbxParseError: If the file is malformed.
        HierarchyNotFoundError: If the _atom_site.pdbx_heterogeneity_id column
            is not present (including when there is no _atom_site loop at all).
    """
    block = _resolve_block(source)
    col = block.find_loop("_atom_site.pdbx_heterogeneity_id")
    if not col:
        raise HierarchyNotFoundError("No _atom_site.pdbx_heterogeneity_id column found")
    return list(col)
