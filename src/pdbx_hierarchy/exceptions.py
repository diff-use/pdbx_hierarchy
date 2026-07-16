"""Custom exception hierarchy for pdbx_hierarchy.

All exceptions inherit from PdbxHierarchyError, allowing users to catch
all library-specific errors with a single except clause.
"""


class PdbxHierarchyError(Exception):
    """Base exception for all pdbx_hierarchy errors."""


class PdbxParseError(PdbxHierarchyError):
    """Raised when parsing an mmCIF file fails.

    This may indicate malformed syntax, missing required data,
    or unexpected file structure.
    """


class PdbxValidationError(PdbxHierarchyError):
    """Raised when data validation fails.

    This is a general validation error. More specific validation
    errors inherit from this class.
    """


class HierarchyNotFoundError(PdbxHierarchyError):
    """Raised when the _pdbx_heterogeneity_hierarchy table is not found.

    This indicates the mmCIF file does not contain hierarchy extension data.
    """


class InvalidHierarchyError(PdbxValidationError):
    """Raised when hierarchy data is structurally invalid.

    Examples include:
    - Duplicate state IDs or names
    - Missing or multiple root states
    - Invalid parent references
    - Cycles in the hierarchy tree
    """


class InvalidCoexistenceError(PdbxValidationError):
    """Raised when coexistence data is invalid.

    Examples include:
    - Invalid rule values (not AND/OR/NOT)
    - References to non-existent hierarchy states
    """


class AtomSiteReferenceError(PdbxValidationError):
    """Raised when _atom_site.pdbx_heterogeneity_id references an invalid state.

    This occurs when an atom's heterogeneity_id does not match any
    state defined in the hierarchy table.
    """


class ClashAnalysisError(PdbxValidationError):
    """Raised when clash analysis or report handling fails.

    Examples include:
    - A clash report with an unsupported or missing schema version
    - A report referencing hierarchy states that no longer exist in the file
    - Malformed report data that cannot be applied
    """
