# pdbx_hierarchy

A Python library and command-line tool for reading, writing, and modifying
PDBx/mmCIF protein structure files with support for **hierarchical
heterogeneity extensions**.

Conformational and compositional heterogeneity in a structure is often more
than a flat list of alternate conformers: states can nest, branch, and
coexist. `pdbx_hierarchy` adds two tables to the standard PDBx/mmCIF format to
capture that structure, and gives you a [gemmi](https://gemmi.readthedocs.io/)-friendly
API plus a CLI to work with them.

## The hierarchy extension

Two additional categories are layered on top of standard mmCIF:

| Table | Purpose |
|-------|---------|
| `_pdbx_heterogeneity_hierarchy` | A tree of heterogeneity **states** (`id`, `name`, `parent`, `details`). Exactly one root; every other state points at its parent. |
| `_pdbx_state_coexistence` | **Coexistence rules** (`AND` / `OR` / `NOT`) relating one state to others. |

When a hierarchy is present, each atom is linked to its state through the
`_atom_site.pdbx_heterogeneity_id` column. State ids default to the
auto-generated sequence `A`, `B`, … `Z`, `AA`, … (the root is conventionally
`Base`), but any unique, whitespace-free label (e.g. `open`, `closed`) is
valid and preserved throughout.

## Installation

Requires **Python 3.14** (3.15 is not yet supported due to
pydantic-core/PyO3 compatibility).

```bash
# Clone, then sync the project environment (creates .venv with all deps)
uv sync
```

`uv sync` installs the `pdbx-hierarchy` console script into the project's
`.venv`, but does not put it on your PATH. Pick whichever of the following
fits your workflow:

```bash
# Option 1 — run via uv without activating anything (no PATH changes)
uv run pdbx-hierarchy --help

# Option 2 — activate the project venv, then call the command directly
source .venv/bin/activate
pdbx-hierarchy --help

# Option 3 — install it as a standalone tool on your PATH (works anywhere)
uv tool install .
pdbx-hierarchy --help
```

> The command examples below are written as bare `pdbx-hierarchy …`. If you
> haven't activated the venv (Option 2) or installed the tool (Option 3),
> prefix each one with `uv run` (Option 1) — e.g.
> `uv run pdbx-hierarchy show structure.cif`.

### Developing on the code

`uv sync` installs the project itself in **editable** mode inside `.venv`, so
source edits are picked up immediately by `uv run pdbx-hierarchy` (and after
`source .venv/bin/activate`) with no reinstall. To get an editable command on
your PATH instead, use an editable tool install — re-run with `--reinstall` to
refresh it — or the classic pip equivalent inside an activated environment:

```bash
uv tool install --editable .   # editable; `--reinstall` to rebuild
pip install -e .               # classic editable install (activated env)
```

## Command-line usage

```bash
pdbx-hierarchy --help
```

### Inspecting a file

```bash
# Show hierarchy, coexistence rules, and atom-assignment summary
pdbx-hierarchy show structure.cif

# Render the hierarchy as a tree, or emit machine-readable JSON
pdbx-hierarchy show structure.cif --tree
pdbx-hierarchy show structure.cif --json

# Limit the output to one section
pdbx-hierarchy show structure.cif --hierarchy

# Validate hierarchy structure, coexistence references, and atom assignments
pdbx-hierarchy validate structure.cif          # report every error
pdbx-hierarchy validate structure.cif --strict # report only the first
```

### Creating a hierarchy

```bash
# Infer a hierarchy from _atom_site.label_alt_id and write assignments
pdbx-hierarchy infer plain.cif -o with_hierarchy.cif

# Apply a hierarchy described by a JSON spec (a serialized HierarchyTree)
pdbx-hierarchy import plain.cif --spec tree.json -o with_hierarchy.cif
```

Output-producing commands default to a non-clobbering `<name>_pdbx_N.cif`
when `-o/--output` is omitted, and prompt before overwriting an existing file
unless you pass `-y/--yes`.

### Editing hierarchy states

```bash
pdbx-hierarchy hierarchy add      structure.cif --id C --name state_c --parent A
pdbx-hierarchy hierarchy rename    structure.cif --id C --name folded
pdbx-hierarchy hierarchy reparent  structure.cif --id C --parent B
pdbx-hierarchy hierarchy remove    structure.cif --id C     # folds atoms/children into the parent
pdbx-hierarchy hierarchy merge     structure.cif --ids A,B  # first id absorbs the rest
pdbx-hierarchy hierarchy reassign  structure.cif            # canonicalize ids (Base, A, B, …)

# Split a state's atoms into two children by residue selection
pdbx-hierarchy hierarchy split structure.cif --id A \
    --select-a "B/10-12,14" --select-b "B/20-25"
```

Selections are comma-separated `[CHAIN/]RANGE` tokens (`lo-hi` or a single
number); the chain prefix is optional when the state spans a single chain.
Use `--auth` to select on `auth_*` numbering instead of the canonical
`label_*`. Residues matched by neither selection keep their current
assignment (a warning lists them).

### Editing coexistence rules

```bash
pdbx-hierarchy coexist add    structure.cif --rule OR --source A --related B,C
pdbx-hierarchy coexist remove structure.cif --id 1
```

Hierarchy-editing commands keep everything in sync: atom assignments are folded or
remapped, and coexistence references are rewritten (dropping rules that become
degenerate). The whole file round-trips, so non-hierarchy data is preserved.

## Python API

```python
from pathlib import Path

from pdbx_hierarchy import (
    read_mmcif,
    read_hierarchy,
    validate_file,
    HierarchyState,
    HierarchyTree,
    assign_from_alt_ids,
)

block = read_mmcif(Path("structure.cif"))

# Read and traverse the hierarchy
tree = read_hierarchy(block)
print(tree.get_root().id)
print([s.id for s in tree.get_children("Base")])

# Infer a hierarchy from alternate-conformer ids
tree, atom_ids = assign_from_alt_ids(block)

# Build one by hand
tree = HierarchyTree(
    states=[
        HierarchyState(id="Base", name="base_state", parent=None),
        HierarchyState(id="A", name="state_a", parent="Base"),
    ]
)

# Validate (raises by default; pass raise_on_error=False to collect messages)
errors = validate_file(block, raise_on_error=False)
```

All library errors derive from `PdbxHierarchyError`
(`PdbxParseError`, `PdbxValidationError` and its subclasses,
`HierarchyNotFoundError`), so you can catch them with a single `except`.

## Development

```bash
uv run pytest              # run the test suite
uv run ruff check .        # lint
uv run ruff format .       # format
uv run mypy src/           # type-check (strict)
```

See [CLAUDE.md](CLAUDE.md) for coding standards, project structure, and
domain notes.

## License

MIT — see [LICENSE](LICENSE).
