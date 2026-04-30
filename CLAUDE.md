# pdbx_hierarchy

Python library for reading, writing, and modifying PDBx/mmCIF protein structure files with support for hierarchical heterogeneity extensions.

## Quick Start

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Lint and format
uv run ruff check .
uv run ruff format .

# Type check
uv run mypy src/
```

## Requirements

- Python 3.14 (3.15 not yet supported due to pydantic-core/PyO3 compatibility)

## Project Structure

```
src/pdbx_hierarchy/
├── __init__.py
├── exceptions.py         # Custom exception hierarchy
├── models/
│   ├── hierarchy.py      # HierarchyState, HierarchyTree
│   ├── coexistence.py    # StateCoexistence, CoexistenceTable
│   └── id_generator.py   # HierarchyIdGenerator
├── io/                   # Read/write functions
└── cli/                  # Typer CLI
    ├── app.py            # Main Typer app
    └── commands/         # Subcommand modules
```

## Dependencies

| Package | Purpose |
|---------|---------|
| gemmi | PDBx/mmCIF file I/O and protein structure representations |
| pydantic | Data models for hierarchy tree structures |
| typer | CLI framework |

Dev dependencies: pytest, mypy, ruff

## Coding Standards

### Type Hints
- All functions must have complete type annotations
- Run `uv run mypy src/` to verify

### Docstrings
Use Google style:
```python
def read_structure(path: Path) -> gemmi.Structure:
    """Read a PDBx/mmCIF file into a gemmi Structure.

    Args:
        path: Path to the mmCIF file.

    Returns:
        Parsed structure object.

    Raises:
        FileNotFoundError: If the file does not exist.
        PdbxParseError: If the file is malformed.
    """
```

### Exceptions
Use custom exception hierarchy rooted at `PdbxHierarchyError`:
- `PdbxHierarchyError` - Base exception
- `PdbxParseError` - Malformed mmCIF syntax
- `PdbxValidationError` - General validation failure
  - `InvalidHierarchyError` - Invalid hierarchy structure (cycles, missing root, etc.)
  - `InvalidCoexistenceError` - Invalid coexistence rules/references
  - `AtomSiteReferenceError` - atom_site references unknown hierarchy state
- `HierarchyNotFoundError` - No hierarchy table in file

### Logging
Use standard library logging with named loggers:
```python
import logging
logger = logging.getLogger(__name__)
```

### Formatting
- Line length: 120 characters
- See `pyproject.toml` for full ruff configuration

### Pydantic Patterns
For mutable models with validation on modification:
- Use `model_config = {"validate_assignment": True}` to re-validate on field assignment
- Override `__setattr__` to rebuild internal indices when collection fields change
- Use `PrivateAttr` for internal lookup tables (e.g., `_by_id`, `_children`)

```python
class HierarchyTree(BaseModel):
    model_config = {"validate_assignment": True}
    states: list[HierarchyState] = Field(default_factory=list)
    _by_id: dict[str, HierarchyState] = PrivateAttr(default_factory=dict)

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name == "states":
            self._validate_and_rebuild_indices()
```

### mmCIF Serialization
When converting between Python and mmCIF format:
- `None` → `"."` for missing parent references
- `None` → `"?"` for optional fields (details, description)
- Comma-separated strings ↔ `list[str]` for `heterogeneity_ids`

## API Design Principles

1. **Mirror gemmi patterns**: API should feel familiar to gemmi users
2. **Functions over classes**: Use Go-like philosophy - classes for data grouping, functions for operations
3. **Strict by default**: Raise exceptions on invalid data; use warnings only when explicitly appropriate

## Domain Context

### PDBx/mmCIF Format
The master format for the Worldwide Protein Data Bank. Key concepts:
- **STAR grammar**: Whitespace-delimited tokens, `loop_` for tables, `_tag value` for key-value pairs
- **Categories**: Data organized into tables (e.g., `_atom_site`, `_entity`)
- **label_ vs auth_ identifiers**: `label_*` for software logic (unique), `auth_*` for display (legacy)

### Hierarchy Extension
This library implements two additional tables:

**`_pdbx_heterogeneity_hierarchy`** → `HierarchyState`, `HierarchyTree`
- `id`: Unique state identifier (Base, A, B, AA, etc.)
- `name`: Human-readable name (no whitespace)
- `parent`: Parent state's id (`.` for root)
- `details`: Optional description

**`_pdbx_state_coexistence`** → `StateCoexistence`, `CoexistenceTable`
- `id`: Unique integer identifier (>= 1)
- `rule`: AND, OR, NOT
- `heterogeneity_id`: Source state
- `heterogeneity_ids`: Related states (comma-separated)
- `description`: Optional description

**ID Generation**: `HierarchyIdGenerator` produces IDs in sequence A-Z, AA-ZZ, AAA-ZZZ, etc.

When hierarchy tables are present, `_atom_site.pdbx_heterogeneity_id` links each atom to its state.

## CLI Usage

```bash
uv run pdbx-hierarchy --help
```

## Testing

```bash
uv run pytest                    # Run all tests
uv run pytest -v                 # Verbose output
uv run pytest tests/test_io.py   # Run specific file
uv run pytest -k "test_read"     # Run tests matching pattern
```

Place test files in `tests/` directory, mirroring src structure.

## Git Commits

Use two `-m` flags for commit messages:
- First `-m`: One-line summary (lowercase, imperative)
- Second `-m`: Additional details if needed

```bash
git commit -m "phase 2: add data models" -m "Add HierarchyState, HierarchyTree, and coexistence models with validation."
```
