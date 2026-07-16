"""Writer functions for PDBx/mmCIF files with hierarchy extensions."""

from __future__ import annotations

from pathlib import Path

import gemmi

from pdbx_hierarchy.exceptions import PdbxValidationError
from pdbx_hierarchy.models.coexistence import CoexistenceTable
from pdbx_hierarchy.models.hierarchy import HierarchyTree

_RESERVED_WORDS = frozenset({"loop_", "stop_", "global_"})


def _needs_quoting(value: str) -> bool:
    """Return True if a bare mmCIF token would be misread and must be quoted."""
    if value == "":
        return True
    if any(char.isspace() for char in value):
        return True
    if value[0] in "_#$[];'\"":
        return True
    lowered = value.lower()
    return lowered in _RESERVED_WORDS or lowered.startswith(("data_", "save_"))


def _mmcif_token(value: str) -> str:
    """Quote a value for mmCIF output, preserving the ``.``/``?`` null markers.

    gemmi's ``Loop.add_row`` writes values verbatim, so a value containing
    whitespace (or other characters that would confuse the parser) must be quoted
    here first, or it would split into multiple tokens and corrupt the loop. Values
    that are already safe bare tokens are returned unchanged, so ordinary ids and
    names stay unquoted. The bare tokens ``.`` (inapplicable) and ``?`` (unknown)
    are left as-is so they keep their null meaning rather than becoming the literal
    strings ``"."``/``"?"``.

    Args:
        value: The raw cell value.

    Returns:
        A token safe to place in an mmCIF loop row.
    """
    if value in (".", "?") or not _needs_quoting(value):
        return value
    if "'" not in value and "\n" not in value:
        return f"'{value}'"
    if '"' not in value and "\n" not in value:
        return f'"{value}"'
    # Hard case (e.g. contains both quote styles or newlines): defer to gemmi.
    return gemmi.cif.quote(value)


def write_hierarchy(block: gemmi.cif.Block, hierarchy: HierarchyTree, *, overwrite: bool = False) -> None:
    """Write a HierarchyTree into the _pdbx_heterogeneity_hierarchy loop in a block.

    Args:
        block: The gemmi Block to write into.
        hierarchy: The HierarchyTree to serialize.
        overwrite: If False (default), raise if the loop already exists.

    Raises:
        PdbxValidationError: If the loop already exists and overwrite is False.
    """
    if block.find_loop("_pdbx_heterogeneity_hierarchy.id") and not overwrite:
        raise PdbxValidationError("_pdbx_heterogeneity_hierarchy already exists; pass overwrite=True to replace")
    loop = block.init_loop("_pdbx_heterogeneity_hierarchy.", ["id", "name", "parent", "details"])
    for row in hierarchy.to_mmcif_rows():
        loop.add_row([_mmcif_token(row[key]) for key in ("id", "name", "parent", "details")])


def write_coexistence(block: gemmi.cif.Block, coexistence: CoexistenceTable, *, overwrite: bool = False) -> None:
    """Write a CoexistenceTable into the _pdbx_state_coexistence loop in a block.

    Args:
        block: The gemmi Block to write into.
        coexistence: The CoexistenceTable to serialize.
        overwrite: If False (default), raise if the loop already exists.

    Raises:
        PdbxValidationError: If the loop already exists and overwrite is False.
    """
    if block.find_loop("_pdbx_state_coexistence.id") and not overwrite:
        raise PdbxValidationError("_pdbx_state_coexistence already exists; pass overwrite=True to replace")
    loop = block.init_loop(
        "_pdbx_state_coexistence.",
        ["id", "rule", "heterogeneity_id", "heterogeneity_ids", "description"],
    )
    for row in coexistence.to_mmcif_rows():
        loop.add_row(
            [_mmcif_token(row[key]) for key in ("id", "rule", "heterogeneity_id", "heterogeneity_ids", "description")]
        )


def write_atom_site_heterogeneity_ids(block: gemmi.cif.Block, ids: list[str], *, overwrite: bool = False) -> None:
    """Write pdbx_heterogeneity_id values into the _atom_site loop.

    Appends a pdbx_heterogeneity_id column to the existing _atom_site loop.
    The length of ids must exactly match the number of rows in _atom_site.

    Args:
        block: The gemmi Block to write into.
        ids: One heterogeneity ID string per atom_site row.
        overwrite: If False (default), raise if the column already exists.

    Raises:
        PdbxValidationError: If there is no _atom_site loop, if the column already
            exists and overwrite is False, or if len(ids) != atom_site row count.
    """
    id_col = block.find_loop("_atom_site.id")
    if not id_col:
        raise PdbxValidationError("No _atom_site loop found in block")
    if block.find_loop("_atom_site.pdbx_heterogeneity_id") and not overwrite:
        raise PdbxValidationError("_atom_site.pdbx_heterogeneity_id already exists; pass overwrite=True to replace")
    if len(ids) != len(id_col):
        raise PdbxValidationError(f"ids length {len(ids)} does not match atom_site row count {len(id_col)}")
    cat = block.get_mmcif_category("_atom_site.")
    cat["pdbx_heterogeneity_id"] = list(ids)
    block.set_mmcif_category("_atom_site.", cat)


def write_mmcif(
    doc: gemmi.cif.Document,
    path: Path,
    *,
    options: gemmi.cif.WriteOptions | None = None,
) -> None:
    """Write a gemmi Document to a mmCIF file.

    Args:
        doc: The gemmi Document to write. Must contain the Block(s) to output.
        path: Destination file path.
        options: WriteOptions controlling output format. Defaults to PDBx style
            (misuse_hash=True), which produces # block separators and loop_ tables.
    """
    if options is None:
        options = gemmi.cif.WriteOptions()
        options.misuse_hash = True
    doc.write_file(str(path), options)
