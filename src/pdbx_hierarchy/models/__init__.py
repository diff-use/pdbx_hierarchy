"""Data models for pdbx_hierarchy.

This module provides Pydantic models for the hierarchical heterogeneity
extension tables:
    - _pdbx_heterogeneity_hierarchy (HierarchyState, HierarchyTree)
    - _pdbx_state_coexistence (StateCoexistence, CoexistenceTable)
"""

from pdbx_hierarchy.models.coexistence import (
    CoexistenceRule,
    CoexistenceTable,
    StateCoexistence,
)
from pdbx_hierarchy.models.hierarchy import HierarchyState, HierarchyTree
from pdbx_hierarchy.models.id_generator import HierarchyIdGenerator

__all__ = [
    # Hierarchy models
    "HierarchyState",
    "HierarchyTree",
    # Coexistence models
    "CoexistenceRule",
    "StateCoexistence",
    "CoexistenceTable",
    # Utilities
    "HierarchyIdGenerator",
]
