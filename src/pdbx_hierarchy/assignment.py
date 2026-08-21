"""Atom assignment algorithm for inferring hierarchy states from label_alt_id."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import gemmi

from pdbx_hierarchy.exceptions import PdbxParseError
from pdbx_hierarchy.io.reader import _resolve_block, count_atom_site_rows
from pdbx_hierarchy.models.hierarchy import HierarchyState, HierarchyTree
from pdbx_hierarchy.models.id_generator import HierarchyIdGenerator

logger = logging.getLogger(__name__)

_BACKBONE_SENTINELS = frozenset({"N", "CA", "C"})
_N_ADJACENT_ATOMS = frozenset({"H", "HN"})


@dataclass
class _AtomRecord:
    row_index: int
    residue_key: tuple[str, str]
    atom_name: str
    alt_id: str


@dataclass
class _GroupClusters:
    ca_side_rep: int
    n_side_rep: int | None
    all_n_adj: bool = False


class _UnionFind:
    """Disjoint Set Union (Union-Find) over atom row indices.

    Elements are atom row indices (0..n-1). Each row starts in its own component.
    `union` merges two components; `find` returns the canonical root used as a
    group key for state ID assignment.

    A split (residue, alt_id) group is represented by two *separate* components
    — N-side and CA-side — because Step 2 never calls `union` across the split.
    Each side stores one representative row in `_GroupClusters`; Step 3 merges
    components by unioning those representatives, which automatically pulls in
    all rows already grouped with them.

    Uses path halving in `find` and union by rank in `union` for near-O(1)
    amortized cost per operation.

    Args:
        n: Total number of atom rows.
    """

    def __init__(self, n: int) -> None:
        self._parent = list(range(n))
        self._rank = [0] * n

    def find(self, x: int) -> int:
        """Return the root of x's component (path halving for compression)."""
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        """Merge the components containing x and y (union by rank)."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1


def _compute_residue_keys(
    asym_ids: list[str],
    seq_ids: list[str],
    auth_asym_ids: list[str],
    auth_seq_ids: list[str],
) -> tuple[list[tuple[str, str]], set[tuple[str, str]]]:
    """Compute tiered residue keys. Returns (per-atom keys, Tier 1 key set).

    Args:
        asym_ids: label_asym_id values.
        seq_ids: label_seq_id values.
        auth_asym_ids: auth_asym_id values (fallback = label_asym_id).
        auth_seq_ids: auth_seq_id values (fallback = label_seq_id).

    Returns:
        Tuple of (per-atom residue key list, set of Tier 1 residue keys).

    Raises:
        PdbxParseError: If any input list has a different length than asym_ids.
    """
    n = len(asym_ids)
    for name, lst in (("seq_ids", seq_ids), ("auth_asym_ids", auth_asym_ids), ("auth_seq_ids", auth_seq_ids)):
        if len(lst) != n:
            raise PdbxParseError(f"_atom_site column length mismatch: asym_ids has {n} rows but {name} has {len(lst)}")

    dot_chain_auth_seqs: dict[str, set[str]] = {}
    for i, seq in enumerate(seq_ids):
        if seq == ".":
            dot_chain_auth_seqs.setdefault(asym_ids[i], set()).add(auth_seq_ids[i])

    ambiguous_chains = {chain for chain, auths in dot_chain_auth_seqs.items() if len(auths) > 1}

    keys: list[tuple[str, str]] = []
    tier1_keys: set[tuple[str, str]] = set()

    for i, seq in enumerate(seq_ids):
        asym = asym_ids[i]
        if seq != ".":
            key = (asym, seq)
            keys.append(key)
            tier1_keys.add(key)
        elif asym not in ambiguous_chains:
            keys.append((asym, "."))
        else:
            keys.append((auth_asym_ids[i], auth_seq_ids[i]))

    return keys, tier1_keys


def _parse_atoms(
    block: gemmi.cif.Block,
) -> tuple[list[_AtomRecord] | None, set[tuple[str, str]]]:
    """Parse _atom_site columns into _AtomRecord list.

    Args:
        block: The gemmi Block to parse.

    Returns:
        (None, set()) if no label_alt_id column exists.
        (records, tier1_keys) otherwise.

    Raises:
        PdbxParseError: If required columns are missing alongside label_alt_id.
    """
    if not block.find_loop("_atom_site.label_alt_id"):
        return None, set()

    tbl = block.find("_atom_site.", ["label_asym_id", "label_seq_id", "label_atom_id", "label_alt_id"])
    if not tbl:
        raise PdbxParseError("_atom_site is missing a required column (label_asym_id, label_seq_id, or label_atom_id)")

    rows_data = list(tbl)
    n = len(rows_data)
    asym_ids = [row[0] for row in rows_data]
    seq_ids = [row[1] for row in rows_data]
    atom_ids = [row[2] for row in rows_data]
    alt_ids = [row[3] for row in rows_data]

    auth_asym_col = block.find_loop("_atom_site.auth_asym_id")
    auth_seq_col = block.find_loop("_atom_site.auth_seq_id")
    auth_asym_ids = list(auth_asym_col) if auth_asym_col else asym_ids[:]
    auth_seq_ids = list(auth_seq_col) if auth_seq_col else seq_ids[:]

    residue_keys, tier1_keys = _compute_residue_keys(asym_ids, seq_ids, auth_asym_ids, auth_seq_ids)

    records = [
        _AtomRecord(row_index=i, residue_key=residue_keys[i], atom_name=atom_ids[i], alt_id=alt_ids[i])
        for i in range(n)
    ]
    return records, tier1_keys


def _initial_groups(
    atoms: list[_AtomRecord],
) -> tuple[_UnionFind, dict[tuple[tuple[str, str], str], _GroupClusters]]:
    """Step 2: provisional groups + intra-residue split.

    Args:
        atoms: Parsed atom records.

    Returns:
        (UnionFind over row indices, group cluster representatives).
    """
    uf = _UnionFind(len(atoms))

    residue_atom_names: dict[tuple[str, str], set[str]] = {}
    residue_alt_rows: dict[tuple[tuple[str, str], str], list[int]] = {}

    for atom in atoms:
        residue_atom_names.setdefault(atom.residue_key, set()).add(atom.atom_name)
        if atom.alt_id != ".":
            residue_alt_rows.setdefault((atom.residue_key, atom.alt_id), []).append(atom.row_index)

    group_clusters: dict[tuple[tuple[str, str], str], _GroupClusters] = {}

    for (res_key, alt_id), row_indices in residue_alt_rows.items():
        all_names = residue_atom_names[res_key]
        has_backbone = bool(all_names & _BACKBONE_SENTINELS)

        if not has_backbone:
            for ri in row_indices[1:]:
                uf.union(row_indices[0], ri)
            group_clusters[(res_key, alt_id)] = _GroupClusters(ca_side_rep=row_indices[0], n_side_rep=None)
            continue

        group_names = {atoms[ri].atom_name for ri in row_indices}
        has_n = "N" in group_names
        has_ca = "CA" in group_names
        has_n_adj = bool(group_names & _N_ADJACENT_ATOMS)
        has_non_n_adj = bool(group_names - _N_ADJACENT_ATOMS)

        if (not has_n) and (not has_ca) and has_n_adj and has_non_n_adj:
            n_side = [ri for ri in row_indices if atoms[ri].atom_name in _N_ADJACENT_ATOMS]
            ca_side = [ri for ri in row_indices if atoms[ri].atom_name not in _N_ADJACENT_ATOMS]
            for ri in n_side[1:]:
                uf.union(n_side[0], ri)
            for ri in ca_side[1:]:
                uf.union(ca_side[0], ri)
            group_clusters[(res_key, alt_id)] = _GroupClusters(ca_side_rep=ca_side[0], n_side_rep=n_side[0])
        else:
            for ri in row_indices[1:]:
                uf.union(row_indices[0], ri)
            group_clusters[(res_key, alt_id)] = _GroupClusters(
                ca_side_rep=row_indices[0],
                n_side_rep=None,
                all_n_adj=has_n_adj and not has_non_n_adj,
            )

    return uf, group_clusters


def _apply_merges(
    atoms: list[_AtomRecord],
    uf: _UnionFind,
    group_clusters: dict[tuple[tuple[str, str], str], _GroupClusters],
    tier1_keys: set[tuple[str, str]],
) -> None:
    """Step 3: inter-residue backbone merging via Union-Find.

    Args:
        atoms: Parsed atom records.
        uf: Union-Find to update in place.
        group_clusters: Group cluster representatives from Step 2.
        tier1_keys: Residue keys that participate in cross-residue merging.
    """
    residue_alts: dict[tuple[str, str], set[str]] = {}
    res_atom_alts: dict[tuple[tuple[str, str], str], set[str]] = {}
    residue_has_backbone: dict[tuple[str, str], bool] = {}

    for atom in atoms:
        rk = atom.residue_key
        if rk not in residue_has_backbone:
            residue_has_backbone[rk] = False
        if atom.atom_name in _BACKBONE_SENTINELS:
            residue_has_backbone[rk] = True
        if atom.alt_id != ".":
            residue_alts.setdefault(rk, set()).add(atom.alt_id)
            res_atom_alts.setdefault((rk, atom.atom_name), set()).add(atom.alt_id)

    chain_residues: dict[str, list[tuple[int, tuple[str, str]]]] = {}
    for rk in tier1_keys:
        if not residue_has_backbone.get(rk, False):
            continue
        try:
            seq_num = int(rk[1])
        except ValueError:
            continue
        chain_residues.setdefault(rk[0], []).append((seq_num, rk))

    for res_list in chain_residues.values():
        res_list.sort()
        for i in range(len(res_list) - 1):
            r_seq, r_key = res_list[i]
            r1_seq, r1_key = res_list[i + 1]
            if r1_seq != r_seq + 1:
                continue
            common_alts = residue_alts.get(r_key, set()) & residue_alts.get(r1_key, set())

            for alt_id in common_alts:
                r_gc = group_clusters.get((r_key, alt_id))
                r1_gc = group_clusters.get((r1_key, alt_id))
                if r_gc is None or r1_gc is None:
                    continue

                r_c_alt = alt_id in res_atom_alts.get((r_key, "C"), set()) or alt_id in res_atom_alts.get(
                    (r_key, "O"), set()
                )
                r1_n_alt = alt_id in res_atom_alts.get((r1_key, "N"), set())
                r1_h_alt = alt_id in res_atom_alts.get((r1_key, "H"), set()) or alt_id in res_atom_alts.get(
                    (r1_key, "HN"), set()
                )

                if r_c_alt and r1_n_alt:
                    uf.union(r_gc.ca_side_rep, r1_gc.ca_side_rep)
                elif r_c_alt and not r1_n_alt and r1_h_alt:
                    if r1_gc.n_side_rep is not None:
                        uf.union(r_gc.ca_side_rep, r1_gc.n_side_rep)
                    elif r1_gc.all_n_adj:
                        uf.union(r_gc.ca_side_rep, r1_gc.ca_side_rep)


def _build_tree(
    atoms: list[_AtomRecord],
    uf: _UnionFind,
) -> tuple[HierarchyTree, dict[int, str]]:
    """Step 4: build HierarchyTree and root-to-state-ID mapping.

    Args:
        atoms: Parsed atom records.
        uf: Finalized Union-Find.

    Returns:
        (HierarchyTree, dict mapping UF root → state ID).
    """
    root_to_earliest: dict[int, int] = {}
    for atom in atoms:
        if atom.alt_id == ".":
            continue
        root = uf.find(atom.row_index)
        if root not in root_to_earliest or atom.row_index < root_to_earliest[root]:
            root_to_earliest[root] = atom.row_index

    sorted_roots = sorted(root_to_earliest, key=lambda r: root_to_earliest[r])

    states: list[HierarchyState] = [HierarchyState(id="Base", name="base_state", parent=None, details=None)]
    gen = HierarchyIdGenerator()
    root_to_state_id: dict[int, str] = {}

    for root in sorted_roots:
        sid = next(gen)
        states.append(HierarchyState(id=sid, name=f"state_{sid}", parent="Base", details=None))
        root_to_state_id[root] = sid

    return HierarchyTree(states=states), root_to_state_id


def _assign(
    atoms: list[_AtomRecord],
    uf: _UnionFind,
    root_to_state_id: dict[int, str],
) -> list[str]:
    """Step 5: produce per-atom state assignment list.

    Args:
        atoms: Parsed atom records.
        uf: Finalized Union-Find.
        root_to_state_id: Mapping from UF root to state ID.

    Returns:
        One state ID per atom, in row order.
    """
    return ["Base" if atom.alt_id == "." else root_to_state_id[uf.find(atom.row_index)] for atom in atoms]


def assign_from_alt_ids(
    source: Path | gemmi.cif.Block,
) -> tuple[HierarchyTree, list[str]]:
    """Infer a HierarchyTree and per-atom assignments from _atom_site.label_alt_id.

    Args:
        source: Path to an mmCIF file or an already-loaded gemmi Block.

    Returns:
        (HierarchyTree, list[str]) — one state ID per atom_site row, in row order.
        Atoms with label_alt_id == "." are assigned to "Base".

    Raises:
        PdbxParseError: If atom_site is missing or malformed.
        FileNotFoundError: If source is a Path and the file does not exist.
    """
    block = _resolve_block(source)
    atoms, tier1_keys = _parse_atoms(block)

    base_tree = HierarchyTree(states=[HierarchyState(id="Base", name="base_state", parent=None, details=None)])

    if atoms is None:
        return base_tree, ["Base"] * count_atom_site_rows(block)

    if all(atom.alt_id == "." for atom in atoms):
        return base_tree, ["Base"] * len(atoms)

    uf, group_clusters = _initial_groups(atoms)
    _apply_merges(atoms, uf, group_clusters, tier1_keys)
    tree, root_to_state_id = _build_tree(atoms, uf)
    return tree, _assign(atoms, uf, root_to_state_id)
