"""pdbx_hierarchy - Library for PDBx/mmCIF files with hierarchical heterogeneity extensions."""

from importlib.metadata import version

from pdbx_hierarchy.exceptions import (
    AtomSiteReferenceError,
    HierarchyNotFoundError,
    InvalidCoexistenceError,
    InvalidHierarchyError,
    PdbxHierarchyError,
    PdbxParseError,
    PdbxValidationError,
)
from pdbx_hierarchy.assignment import assign_from_alt_ids
from pdbx_hierarchy.io import (
    has_hierarchy,
    read_atom_site_heterogeneity_ids,
    read_coexistence,
    read_hierarchy,
    read_mmcif,
    validate_atom_site_references,
    validate_file,
    write_atom_site_heterogeneity_ids,
    write_coexistence,
    write_hierarchy,
    write_mmcif,
)
from pdbx_hierarchy.models import (
    CoexistenceRule,
    CoexistenceTable,
    HierarchyIdGenerator,
    HierarchyState,
    HierarchyTree,
    StateCoexistence,
)

__version__ = version("pdbx-hierarchy")

__all__ = [
    "__version__",
    # Exceptions
    "PdbxHierarchyError",
    "PdbxParseError",
    "PdbxValidationError",
    "HierarchyNotFoundError",
    "InvalidHierarchyError",
    "InvalidCoexistenceError",
    "AtomSiteReferenceError",
    # Models
    "HierarchyState",
    "HierarchyTree",
    "CoexistenceRule",
    "StateCoexistence",
    "CoexistenceTable",
    "HierarchyIdGenerator",
    # Assignment
    "assign_from_alt_ids",
    # I/O
    "read_mmcif",
    "has_hierarchy",
    "read_hierarchy",
    "read_coexistence",
    "read_atom_site_heterogeneity_ids",
    "write_hierarchy",
    "write_coexistence",
    "write_atom_site_heterogeneity_ids",
    "write_mmcif",
    "validate_atom_site_references",
    "validate_file",
]
