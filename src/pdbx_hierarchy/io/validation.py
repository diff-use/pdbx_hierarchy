"""Validation functions for PDBx/mmCIF files with hierarchy extensions."""

from __future__ import annotations

from pathlib import Path

import gemmi

from pdbx_hierarchy.exceptions import AtomSiteReferenceError, HierarchyNotFoundError
from pdbx_hierarchy.io.reader import read_atom_site_heterogeneity_ids, read_coexistence, read_hierarchy
from pdbx_hierarchy.models.hierarchy import HierarchyTree


def validate_atom_site_references(
    source: Path | gemmi.cif.Block,
    hierarchy: HierarchyTree,
    *,
    raise_on_error: bool = True,
) -> list[str]:
    """Validate that every _atom_site.pdbx_heterogeneity_id references a known hierarchy state.

    Args:
        source: Path to an mmCIF file or an already-loaded gemmi Block.
        hierarchy: The HierarchyTree to validate references against.
        raise_on_error: If True (default), raise AtomSiteReferenceError on the first bad reference.
            If False, collect all errors and return them.

    Returns:
        List of error strings. Empty when all references are valid.

    Raises:
        AtomSiteReferenceError: If raise_on_error=True and any atom references an unknown state.
        FileNotFoundError: If source is a Path that does not exist.
        PdbxParseError: If the file cannot be parsed.
    """
    try:
        atom_ids = read_atom_site_heterogeneity_ids(source)
    except HierarchyNotFoundError:
        return []

    errors: list[str] = []
    for i, atom_id in enumerate(atom_ids):
        if not hierarchy.contains(atom_id):
            msg = f"atom_site row {i}: pdbx_heterogeneity_id {atom_id!r} not found in hierarchy"
            if raise_on_error:
                raise AtomSiteReferenceError(msg)
            errors.append(msg)
    return errors


def validate_file(
    source: Path | gemmi.cif.Block,
    *,
    raise_on_error: bool = True,
) -> list[str]:
    """Validate all hierarchy-related data in an mmCIF file.

    Checks:
    1. Hierarchy table is structurally valid (cycles, missing root, etc.)
    2. Coexistence rules reference states that exist in the hierarchy.
    3. All atom_site pdbx_heterogeneity_id values reference known states.

    If no hierarchy table is present, returns an empty list without error.

    Args:
        source: Path to an mmCIF file or an already-loaded gemmi Block.
        raise_on_error: If True (default), raise on the first validation error.
            If False, collect all errors and return them.

    Returns:
        List of error strings. Empty when valid (or no hierarchy present).

    Raises:
        InvalidHierarchyError: If raise_on_error=True and hierarchy structure is invalid.
        InvalidCoexistenceError: If raise_on_error=True and coexistence rules are invalid.
        AtomSiteReferenceError: If raise_on_error=True and atom_site has bad references.
        FileNotFoundError: If source is a Path that does not exist.
        PdbxParseError: If the file cannot be parsed.
    """
    try:
        hierarchy = read_hierarchy(source)
    except HierarchyNotFoundError:
        return []

    errors: list[str] = []

    coexistence = read_coexistence(source)
    if coexistence is not None:
        errors.extend(coexistence.validate_against_hierarchy(hierarchy, raise_on_error=raise_on_error))

    errors.extend(validate_atom_site_references(source, hierarchy, raise_on_error=raise_on_error))

    return errors
