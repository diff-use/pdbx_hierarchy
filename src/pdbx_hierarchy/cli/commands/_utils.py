"""Shared helpers for CLI command modules."""

from __future__ import annotations

import re
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import gemmi
import typer
from pydantic import ValidationError

from pdbx_hierarchy.exceptions import PdbxHierarchyError, PdbxParseError, PdbxValidationError
from pdbx_hierarchy.models.coexistence import CoexistenceTable, StateCoexistence
from pdbx_hierarchy.models.hierarchy import HierarchyState, HierarchyTree
from pdbx_hierarchy.models.id_generator import HierarchyIdGenerator

if TYPE_CHECKING:
    from collections.abc import Iterator

# A canonical state id is an uppercase-letter run (A, B, ..., Z, AA, ...), as produced
# by HierarchyIdGenerator. Anything else (e.g. "Base" or a hand-given label) is "named".
_CANONICAL_ID_RE = re.compile(r"^[A-Z]+$")


def fail(message: str) -> NoReturn:
    """Print an error message to stderr and exit with status 1."""
    typer.secho(f"Error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def warn(message: str) -> None:
    """Print a warning message to stderr."""
    typer.secho(f"Warning: {message}", fg=typer.colors.YELLOW, err=True)


@contextmanager
def error_handler() -> Iterator[None]:
    """Convert known, expected errors into a clean exit-1 message.

    Only expected failures are caught: a missing file, an unknown id/rule
    (``KeyError`` from the models), input that fails our own validation
    (``PdbxHierarchyError``), or a pydantic ``ValidationError`` (e.g. a malformed
    ``import`` spec). Unexpected exceptions are deliberately left to propagate so
    bugs surface with a full traceback rather than being disguised as user error.
    """
    try:
        yield
    except FileNotFoundError as exc:
        fail(f"File not found: {exc.filename or exc}")
    except KeyError as exc:
        # KeyError stringifies with surrounding quotes; strip the outer layer.
        fail(str(exc).strip("'\""))
    except ValidationError as exc:
        fail(str(exc))
    except PdbxHierarchyError as exc:
        fail(str(exc))


def load_document(path: Path) -> tuple[gemmi.cif.Document, gemmi.cif.Block]:
    """Read an mmCIF file, returning the Document and its sole Block.

    The full Document is returned so the entire file round-trips when written
    back out; commands mutate the Block and write the Document.

    Args:
        path: Path to the mmCIF file.

    Returns:
        Tuple of (Document, sole Block).

    Raises:
        FileNotFoundError: If the file does not exist.
        PdbxParseError: If the file is malformed or has more than one data block.
    """
    try:
        doc = gemmi.cif.read(str(path))
    except FileNotFoundError:
        raise
    except ValueError as exc:
        raise PdbxParseError(str(exc)) from exc
    try:
        block = doc.sole_block()
    except (RuntimeError, IndexError) as exc:
        raise PdbxParseError(f"Expected exactly one data block in {path}: {exc}") from exc
    return doc, block


def resolve_output(input_path: Path, output: Path | None, *, yes: bool) -> Path:
    """Resolve the destination path for an output-producing command.

    Args:
        input_path: The command's input file.
        output: The requested output path, or None for the auto-suffix default.
        yes: If True, skip the overwrite confirmation prompt.

    Returns:
        The path to write to.
    """
    if output is None:
        counter = 1
        while True:
            candidate = input_path.with_name(f"{input_path.stem}_pdbx_{counter}{input_path.suffix}")
            if not candidate.exists():
                return candidate
            counter += 1

    if output.resolve() == input_path.resolve():
        if not yes:
            typer.confirm(f"Overwrite {input_path}? This cannot be undone.", abort=True)
        return output

    if output.exists() and not yes:
        typer.confirm(f"{output} already exists. Overwrite?", abort=True)
    return output


def parse_residue_ranges(spec: str) -> set[int]:
    """Parse a comma-separated residue spec like ``10-12,14`` into ``{10, 11, 12, 14}``.

    Args:
        spec: Comma-separated ranges (``lo-hi``) and single integers.

    Returns:
        The set of residue numbers.

    Raises:
        PdbxValidationError: If a token is not a valid integer or range.
    """
    result: set[int] = set()
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        try:
            if "-" in token:
                lo_str, hi_str = token.split("-", 1)
                lo, hi = int(lo_str), int(hi_str)
                if hi < lo:
                    raise PdbxValidationError(f"Invalid residue range {token!r}: end is before start")
                result.update(range(lo, hi + 1))
            else:
                result.add(int(token))
        except ValueError:
            raise PdbxValidationError(f"Invalid residue selection token {token!r}") from None
    return result


def parse_selection(spec: str, available_chains: set[str]) -> set[tuple[str, int]]:
    """Parse a residue selection into a set of ``(chain, seq)`` pairs.

    Syntax: comma-separated ``[CHAIN/]RANGE`` tokens, where RANGE is ``lo-hi`` or a
    single integer. The chain prefix is optional when ``available_chains`` holds a
    single chain; otherwise an unqualified token is ambiguous and raises.

    Args:
        spec: The selection spec, e.g. ``"B/10-12,14"`` or ``"10-12,14"``.
        available_chains: Chains present among the atoms being selected over.

    Returns:
        Set of ``(chain, seq)`` pairs named by the spec.

    Raises:
        PdbxValidationError: If a token omits its chain while more than one chain
            is available, or if a range is malformed.
    """
    sole_chain = next(iter(available_chains)) if len(available_chains) == 1 else None

    selected: set[tuple[str, int]] = set()
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        if "/" in token:
            chain_part, range_part = token.split("/", 1)
            chain = chain_part.strip()
        elif sole_chain is not None:
            chain, range_part = sole_chain, token
        else:
            raise PdbxValidationError(
                f"Ambiguous selection token {token!r}: the state spans multiple chains "
                f"{sorted(available_chains)}; qualify it as CHAIN/{token}"
            )
        for seq in parse_residue_ranges(range_part):
            selected.add((chain, seq))
    return selected


def read_atom_residue_keys(block: gemmi.cif.Block, *, use_auth: bool) -> list[tuple[str, str]]:
    """Return per-atom ``(chain, seq)`` keys from the _atom_site loop.

    Args:
        block: The block to read.
        use_auth: If True, read ``auth_asym_id``/``auth_seq_id``; otherwise the
            canonical ``label_asym_id``/``label_seq_id``.

    Returns:
        One ``(chain, seq)`` string pair per atom_site row, in row order.

    Raises:
        PdbxValidationError: If the required columns are absent.
    """
    asym_tag, seq_tag = ("auth_asym_id", "auth_seq_id") if use_auth else ("label_asym_id", "label_seq_id")
    tbl = block.find("_atom_site.", [asym_tag, seq_tag])
    if not tbl:
        raise PdbxValidationError(f"_atom_site is missing {asym_tag} or {seq_tag}")
    return [(row[0], row[1]) for row in tbl]


def reassign_ids(tree: HierarchyTree, *, preserve_named: bool = False) -> dict[str, str]:
    """Canonicalize state ids in place and return the old->new id mapping.

    The root keeps its id; every other state is renumbered to the canonical
    sequence (A, B, C, ..., AA, ...) in breadth-first order. With ``preserve_named``,
    states whose id is not a canonical-pattern id are left untouched and the
    generated ids skip them.

    Args:
        tree: The tree to mutate.
        preserve_named: If True, keep non-canonical ids unchanged.

    Returns:
        Mapping from old id to new id (identity entries included).
    """
    root = tree.get_root()

    bfs_order: list[str] = []
    queue: deque[str] = deque([root.id])
    while queue:
        current = queue.popleft()
        bfs_order.append(current)
        queue.extend(child.id for child in tree.get_children(current))

    mapping: dict[str, str] = {root.id: root.id}
    reserved: set[str] = {root.id}
    if preserve_named:
        for state_id in bfs_order[1:]:
            if not _CANONICAL_ID_RE.match(state_id):
                mapping[state_id] = state_id
                reserved.add(state_id)

    generator = HierarchyIdGenerator()
    for state_id in bfs_order[1:]:
        if state_id in mapping:
            continue
        candidate = next(generator)
        while candidate in reserved:
            candidate = next(generator)
        mapping[state_id] = candidate
        reserved.add(candidate)

    new_states = [
        HierarchyState(
            id=mapping[state.id],
            name=state.name,
            parent=mapping[state.parent] if state.parent is not None else None,
            details=state.details,
        )
        for state in tree.states
    ]
    tree.states = new_states
    return mapping


def remap_coexistence(table: CoexistenceTable, id_map: dict[str, str]) -> tuple[CoexistenceTable, list[str]]:
    """Rewrite coexistence references through an id map, dropping degenerate rules.

    Each rule's source and related ids are remapped via ``id_map`` (identity for
    unmapped ids). Related ids are de-duplicated; a related id equal to the
    (remapped) source is dropped as a self-reference; a rule whose related list
    becomes empty is dropped entirely.

    Args:
        table: The coexistence table to rewrite.
        id_map: Mapping from old id to new id.

    Returns:
        Tuple of (new table, human-readable notes describing each change made).
    """
    notes: list[str] = []
    new_rules: list[StateCoexistence] = []
    for rule in table.rules:
        new_source = id_map.get(rule.heterogeneity_id, rule.heterogeneity_id)

        new_related: list[str] = []
        for ref in rule.heterogeneity_ids:
            mapped = id_map.get(ref, ref)
            if mapped not in new_related:
                new_related.append(mapped)

        if new_source in new_related:
            new_related = [ref for ref in new_related if ref != new_source]
            notes.append(f"rule {rule.id}: dropped self-reference to {new_source!r}")

        if not new_related:
            notes.append(f"rule {rule.id}: dropped (no related states remain after remap)")
            continue

        new_rules.append(
            StateCoexistence(
                id=rule.id,
                rule=rule.rule,
                heterogeneity_id=new_source,
                heterogeneity_ids=new_related,
                description=rule.description,
            )
        )
    return CoexistenceTable(rules=new_rules), notes


def render_tree(tree: HierarchyTree) -> str:
    """Render a HierarchyTree as an indented ASCII tree (2 spaces per level)."""
    lines: list[str] = []

    def walk(state_id: str, depth: int) -> None:
        state = tree.get_state(state_id)
        lines.append(f"{'  ' * depth}{state.id} ({state.name})")
        for child in tree.get_children(state_id):
            walk(child.id, depth + 1)

    walk(tree.get_root().id, 0)
    return "\n".join(lines)
