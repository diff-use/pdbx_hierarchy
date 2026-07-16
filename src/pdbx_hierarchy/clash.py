"""Steric clash detection with merge / coexistence mitigation.

This module detects van der Waals overlaps between atoms and, for clashes between
hierarchy states that could be present simultaneously, proposes a mitigation:
either merging two states into one coupled conformation, or adding a ``NOT``
coexistence rule marking them mutually exclusive.

The analysis is *hierarchy-driven*, not altloc-driven. gemmi's ``ContactSearch``
silently skips atom pairs with differing ``label_alt_id``, and hierarchy states do
not map one-to-one onto altloc letters, so detection runs on an altloc-sanitized
copy of the model and relies entirely on ``pdbx_heterogeneity_id`` to distinguish
states. The output mmCIF (written elsewhere) keeps its original ``label_alt_id``.

Bond handling uses inferred connectivity: a covalent-radii window identifies
bonded (1-2) pairs, from which angle (1-3) pairs are derived; both are excluded
from clash reporting. Because the bond window is centred on the expected covalent
length, gross overlaps far shorter than any real bond are *not* mistaken for bonds
and are reported as clashes.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gemmi

from pdbx_hierarchy.exceptions import ClashAnalysisError, PdbxParseError
from pdbx_hierarchy.io.reader import _resolve_block, read_atom_site_heterogeneity_ids
from pdbx_hierarchy.models.coexistence import CoexistenceRule

if TYPE_CHECKING:
    from collections.abc import Callable

    from pdbx_hierarchy.models.coexistence import CoexistenceTable
    from pdbx_hierarchy.models.hierarchy import HierarchyTree

logger = logging.getLogger(__name__)

#: Current clash-report schema version. ``apply`` rejects other versions.
REPORT_VERSION = 1

#: Root/base state id; atoms here are present in every conformation.
_BASE_ID = "Base"

#: Search radius (Å) for the clash contact search (~2 * max vdW radius).
_CLASH_SEARCH_RADIUS = 4.0
#: Search radius (Å) for covalent-bond inference (covers metal coordination).
_BOND_SEARCH_RADIUS = 3.2
#: A pair is treated as covalently bonded when its distance falls within
#: ``[covalent_sum - _BOND_LOWER_TOL, covalent_sum + _BOND_UPPER_TOL]``. The
#: lower bound keeps gross overlaps (far shorter than a real bond) reportable.
_BOND_LOWER_TOL = 0.45
_BOND_UPPER_TOL = 0.50

#: Clash categories assigned by :func:`classify_clashes`.
CATEGORY_ACTIONABLE = "actionable"
CATEGORY_SAME_STATE = "same_state"
CATEGORY_ANCESTOR = "ancestor"
CATEGORY_BENIGN = "benign"


@dataclass
class ClashAtom:
    """One atom involved in a clash, with its hierarchy assignment."""

    serial: int
    chain: str
    seq: str
    atom_name: str
    het_id: str

    @property
    def residue_key(self) -> tuple[str, str]:
        """Return the ``(chain, seq)`` residue key."""
        return (self.chain, self.seq)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "serial": self.serial,
            "chain": self.chain,
            "seq": self.seq,
            "atom_name": self.atom_name,
            "het_id": self.het_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClashAtom:
        """Reconstruct from a dict produced by :meth:`to_dict`."""
        return cls(
            serial=int(data["serial"]),
            chain=str(data["chain"]),
            seq=str(data["seq"]),
            atom_name=str(data["atom_name"]),
            het_id=str(data["het_id"]),
        )


@dataclass
class Clash:
    """A detected van der Waals overlap between two atoms.

    Attributes:
        atom1: The first atom.
        atom2: The second atom.
        distance: Inter-atomic distance in Å.
        overlap: vdW overlap in Å (sum of vdW radii minus distance).
        image_idx: 0 for a same-asymmetric-unit contact, non-zero for a crystal
            symmetry-mate contact.
        category: Classification assigned by :func:`classify_clashes`.
        note: Optional human-readable explanation of the category.
    """

    atom1: ClashAtom
    atom2: ClashAtom
    distance: float
    overlap: float
    image_idx: int = 0
    category: str = ""
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "atom1": self.atom1.to_dict(),
            "atom2": self.atom2.to_dict(),
            "distance": self.distance,
            "overlap": self.overlap,
            "image_idx": self.image_idx,
            "category": self.category,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Clash:
        """Reconstruct from a dict produced by :meth:`to_dict`."""
        return cls(
            atom1=ClashAtom.from_dict(data["atom1"]),
            atom2=ClashAtom.from_dict(data["atom2"]),
            distance=float(data["distance"]),
            overlap=float(data["overlap"]),
            image_idx=int(data.get("image_idx", 0)),
            category=str(data.get("category", "")),
            note=data.get("note"),
        )


@dataclass
class MergeProposal:
    """A proposal to merge states into one coupled conformation (target first)."""

    states: list[str]
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {"action": "merge", "states": list(self.states), "enabled": self.enabled}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MergeProposal:
        """Reconstruct from a dict produced by :meth:`to_dict`."""
        return cls(states=[str(s) for s in data["states"]], enabled=bool(data.get("enabled", True)))


@dataclass
class NotProposal:
    """A proposal to add a NOT coexistence rule between two states."""

    states: list[str]
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {"action": "not", "states": list(self.states), "enabled": self.enabled}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NotProposal:
        """Reconstruct from a dict produced by :meth:`to_dict`."""
        return cls(states=[str(s) for s in data["states"]], enabled=bool(data.get("enabled", True)))


@dataclass
class ClashReport:
    """The result of classifying clashes: proposed mitigations plus context.

    Attributes:
        clashes: Every detected clash, each tagged with a category.
        merges: Proposed state merges (each with an ``enabled`` toggle).
        not_rules: Proposed NOT rules (each with an ``enabled`` toggle).
        conflicts: Human-readable notes where a proposal could not be made
            unambiguously (e.g. a pair asked to both merge and stay mutually
            exclusive).
        version: Report schema version.
    """

    clashes: list[Clash] = field(default_factory=list)
    merges: list[MergeProposal] = field(default_factory=list)
    not_rules: list[NotProposal] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    version: int = REPORT_VERSION

    def clashes_in(self, category: str) -> list[Clash]:
        """Return all clashes with the given category."""
        return [c for c in self.clashes if c.category == category]

    def to_json(self) -> str:
        """Serialize the report to an indented JSON string."""
        return json.dumps(
            {
                "version": self.version,
                "clashes": [c.to_dict() for c in self.clashes],
                "merges": [m.to_dict() for m in self.merges],
                "not_rules": [n.to_dict() for n in self.not_rules],
                "conflicts": list(self.conflicts),
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> ClashReport:
        """Parse a report from JSON, validating the schema version.

        Args:
            text: JSON produced by :meth:`to_json`.

        Returns:
            The parsed ClashReport.

        Raises:
            ClashAnalysisError: If the JSON is malformed or the version is unsupported.
        """
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ClashAnalysisError(f"clash report is not valid JSON: {exc}") from exc
        version = data.get("version")
        if version != REPORT_VERSION:
            raise ClashAnalysisError(f"unsupported clash report version {version!r}; expected {REPORT_VERSION}")
        try:
            return cls(
                clashes=[Clash.from_dict(c) for c in data.get("clashes", [])],
                merges=[MergeProposal.from_dict(m) for m in data.get("merges", [])],
                not_rules=[NotProposal.from_dict(n) for n in data.get("not_rules", [])],
                conflicts=[str(x) for x in data.get("conflicts", [])],
                version=version,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ClashAnalysisError(f"malformed clash report: {exc}") from exc


@dataclass
class _AtomData:
    """Row-aligned _atom_site data used by detection and classification."""

    serials: list[int]
    residue_keys: list[tuple[str, str]]
    atom_names: list[str]
    het_ids: list[str]
    serial_to_row: dict[int, int]


def _read_atom_data(block: gemmi.cif.Block, *, use_auth: bool) -> _AtomData:
    """Read row-aligned _atom_site columns needed for clash analysis.

    Args:
        block: The block to read.
        use_auth: If True, residue keys use ``auth_*`` numbering; otherwise ``label_*``.

    Returns:
        Parallel per-row lists plus a serial->row lookup.

    Raises:
        PdbxParseError: If a required column is missing or lengths disagree.
        HierarchyNotFoundError: If the pdbx_heterogeneity_id column is absent.
    """
    het_ids = read_atom_site_heterogeneity_ids(block)

    asym_tag, seq_tag = ("auth_asym_id", "auth_seq_id") if use_auth else ("label_asym_id", "label_seq_id")
    tbl = block.find("_atom_site.", [asym_tag, seq_tag, "label_atom_id", "id"])
    if not tbl:
        raise PdbxParseError(f"_atom_site is missing one of {asym_tag}, {seq_tag}, label_atom_id, id")

    residue_keys: list[tuple[str, str]] = []
    atom_names: list[str] = []
    serials: list[int] = []
    serial_to_row: dict[int, int] = {}
    for row_index, row in enumerate(tbl):
        residue_keys.append((row[0], row[1]))
        atom_names.append(row[2])
        try:
            serial = int(row[3])
        except ValueError as exc:
            raise PdbxParseError(f"non-integer _atom_site.id {row[3]!r}") from exc
        serials.append(serial)
        serial_to_row[serial] = row_index

    if not (len(het_ids) == len(residue_keys) == len(atom_names) == len(serials)):
        raise PdbxParseError(
            f"_atom_site column length mismatch: het_ids={len(het_ids)}, residue_keys={len(residue_keys)}, "
            f"atom_names={len(atom_names)}, serials={len(serials)}"
        )
    return _AtomData(serials, residue_keys, atom_names, het_ids, serial_to_row)


def _sanitize_altlocs(model: gemmi.Model) -> None:
    """Clear every atom's altloc so contact search is not gated on label_alt_id."""
    for chain in model:
        for residue in chain:
            for atom in residue:
                atom.altloc = "\0"


def _isolate_asymmetric_unit(structure: gemmi.Structure) -> None:
    """Remove crystal periodicity so only intra-ASU contacts are found.

    gemmi's neighbor search uses the minimum-image convention, which folds
    lattice-translation (crystal-packing) contacts into ``image_idx == 0`` and so
    cannot be filtered out by image index alone. To search the asymmetric unit in
    isolation, the model is translated into a P1 box large enough that no atom sees
    a periodic image of another within the search radius. Coordinates are mutated on
    this in-memory copy only.

    Args:
        structure: The structure to isolate (mutated in place).
    """
    model = structure[0]
    positions = [atom.pos for chain in model for residue in chain for atom in residue]
    if not positions:
        return
    margin = _CLASH_SEARCH_RADIUS + 1.0
    min_x, min_y, min_z = (min(p.x for p in positions), min(p.y for p in positions), min(p.z for p in positions))
    max_x, max_y, max_z = (max(p.x for p in positions), max(p.y for p in positions), max(p.z for p in positions))
    dx, dy, dz = margin - min_x, margin - min_y, margin - min_z
    for chain in model:
        for residue in chain:
            for atom in residue:
                p = atom.pos
                atom.pos = gemmi.Position(p.x + dx, p.y + dy, p.z + dz)
    structure.spacegroup_hm = "P 1"
    structure.cell = gemmi.UnitCell(
        (max_x - min_x) + 2 * margin,
        (max_y - min_y) + 2 * margin,
        (max_z - min_z) + 2 * margin,
        90.0,
        90.0,
        90.0,
    )
    structure.setup_cell_images()


def _bond_graph(
    ns: gemmi.NeighborSearch,
    data: _AtomData,
    same_conformer: Callable[[str, str], bool],
) -> dict[int, set[int]]:
    """Infer covalent connectivity as a serial->neighbors adjacency map.

    A same-asymmetric-unit contact is treated as a bond when (a) both atoms could
    occupy one physical conformation — i.e. their hierarchy states lie on a single
    ancestor path — and (b) its distance falls in a window centred on the sum of
    covalent radii. Connectivity therefore relies entirely on the hierarchy labels,
    never on input altlocs, so atoms of mutually exclusive states are never bonded.
    The window's lower bound keeps a gross overlap (far shorter than any real bond)
    reportable rather than silently absorbed as a bond.

    Args:
        ns: A populated neighbor search over the (sanitized) model.
        data: Row-aligned atom data, for the serial->state lookup.
        same_conformer: Predicate telling whether two states can share a conformation.

    Returns:
        Mapping from atom serial to the set of covalently bonded atom serials.
    """
    graph: dict[int, set[int]] = defaultdict(set)
    cs = gemmi.ContactSearch(_BOND_SEARCH_RADIUS)
    cs.ignore = gemmi.ContactSearch.Ignore.Nothing
    cs.min_occupancy = 0.0
    for contact in cs.find_contacts(ns):
        if contact.image_idx != 0:
            continue
        a1, a2 = contact.partner1.atom, contact.partner2.atom
        s1, s2 = a1.serial, a2.serial
        if s1 == s2:
            continue
        if not same_conformer(data.het_ids[data.serial_to_row[s1]], data.het_ids[data.serial_to_row[s2]]):
            continue
        cov = a1.element.covalent_r + a2.element.covalent_r
        if (cov - _BOND_LOWER_TOL) <= contact.dist <= (cov + _BOND_UPPER_TOL):
            graph[s1].add(s2)
            graph[s2].add(s1)
    return graph


def _excluded_pairs(graph: dict[int, set[int]]) -> set[frozenset[int]]:
    """Return the set of 1-2 (bonded) and 1-3 (angle) atom-serial pairs to ignore."""
    excluded: set[frozenset[int]] = set()
    for center, neighbors in graph.items():
        for nb in neighbors:
            excluded.add(frozenset((center, nb)))
        ordered = sorted(neighbors)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                excluded.add(frozenset((ordered[i], ordered[j])))
    return excluded


def _make_clash(data: _AtomData, s1: int, s2: int, dist: float, overlap: float, image_idx: int) -> Clash:
    """Build a Clash from two atom serials, resolving per-atom hierarchy data."""
    try:
        r1, r2 = data.serial_to_row[s1], data.serial_to_row[s2]
    except KeyError as exc:
        raise PdbxParseError(f"contact references unknown _atom_site serial {exc}") from exc
    atom1 = ClashAtom(s1, data.residue_keys[r1][0], data.residue_keys[r1][1], data.atom_names[r1], data.het_ids[r1])
    atom2 = ClashAtom(s2, data.residue_keys[r2][0], data.residue_keys[r2][1], data.atom_names[r2], data.het_ids[r2])
    return Clash(atom1=atom1, atom2=atom2, distance=round(dist, 3), overlap=round(overlap, 3), image_idx=image_idx)


def detect_clashes(
    source: Path | gemmi.cif.Block,
    tree: HierarchyTree,
    *,
    tolerance: float = 0.4,
    symmetry: bool = False,
    use_auth: bool = False,
) -> list[Clash]:
    """Detect van der Waals clashes between atoms in a structure.

    Runs on an altloc-sanitized copy of the model so hierarchy states, not
    ``label_alt_id``, determine which atoms are compared. Covalently bonded (1-2)
    and angle (1-3) pairs are excluded via connectivity inferred from the hierarchy
    (atoms of mutually exclusive states are never treated as bonded).

    Args:
        source: Path to an mmCIF file or an already-loaded gemmi Block.
        tree: The hierarchy tree, used to decide which states can share a conformation.
        tolerance: Minimum vdW overlap (Å) for a pair to count as a clash.
        symmetry: If True, include crystal symmetry-mate contacts; otherwise only
            same-asymmetric-unit contacts.
        use_auth: If True, residue keys use ``auth_*`` numbering.

    Returns:
        Detected clashes (unclassified). Call :func:`classify_clashes` next.

    Raises:
        PdbxParseError: If _atom_site is missing or malformed.
        HierarchyNotFoundError: If the pdbx_heterogeneity_id column is absent.
    """
    block = _resolve_block(source)
    data = _read_atom_data(block, use_auth=use_auth)

    structure = gemmi.make_structure_from_block(block)
    if len(structure) == 0:
        return []
    model = structure[0]
    _sanitize_altlocs(model)
    if symmetry:
        # Keep the real cell/spacegroup and populate symmetry images so crystal
        # symmetry-mate and lattice-neighbour contacts are included.
        structure.setup_cell_images()
    else:
        _isolate_asymmetric_unit(structure)

    anc_cache: dict[str, set[str]] = {}

    def same_conformer(j: str, k: str) -> bool:
        # Two states can share a conformation iff one is an ancestor of the other
        # (Base is an ancestor of everything, so it is compatible with all states).
        return j == k or j in _ancestors_inclusive(tree, k, anc_cache) or k in _ancestors_inclusive(tree, j, anc_cache)

    ns = gemmi.NeighborSearch(model, structure.cell, _CLASH_SEARCH_RADIUS).populate()
    excluded = _excluded_pairs(_bond_graph(ns, data, same_conformer))

    cs = gemmi.ContactSearch(_CLASH_SEARCH_RADIUS)
    cs.ignore = gemmi.ContactSearch.Ignore.SameResidue
    cs.min_occupancy = 0.0

    clashes: list[Clash] = []
    for contact in cs.find_contacts(ns):
        a1, a2 = contact.partner1.atom, contact.partner2.atom
        overlap = a1.element.vdw_r + a2.element.vdw_r - contact.dist
        if overlap <= tolerance:
            continue
        if frozenset((a1.serial, a2.serial)) in excluded:
            continue
        clashes.append(_make_clash(data, a1.serial, a2.serial, contact.dist, overlap, contact.image_idx))
    return clashes


class _StateUnionFind:
    """Minimal Union-Find over hierarchy state ids for accumulating merges."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def _add(self, x: str) -> None:
        if x not in self._parent:
            self._parent[x] = x

    def find(self, x: str) -> str:
        """Return the representative of x's component (adds x if unseen)."""
        self._add(x)
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, x: str, y: str) -> None:
        """Merge the components containing x and y."""
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self._parent[ry] = rx

    def components(self) -> list[set[str]]:
        """Return the current components as a list of id sets."""
        groups: dict[str, set[str]] = defaultdict(set)
        for member in self._parent:
            groups[self.find(member)].add(member)
        return list(groups.values())


def _ancestors_inclusive(tree: HierarchyTree, state_id: str, cache: dict[str, set[str]]) -> set[str]:
    """Return ``{state_id} | ancestors | {Base}`` (cached), or ``{state_id}`` if unknown."""
    if state_id in cache:
        return cache[state_id]
    try:
        result = {state_id} | {s.id for s in tree.get_ancestors(state_id)}
    except KeyError:
        result = {state_id}
    cache[state_id] = result
    return result


def _decide_pair(
    j: str,
    rk_j: tuple[str, str],
    k: str,
    rk_k: tuple[str, str],
    alt: dict[tuple[str, str], set[str]],
    *,
    not_only: bool,
) -> tuple[str, list[tuple[str, str]]]:
    """Decide how to mitigate an actionable clash between states J and K.

    Returns ``("merge", [(a, b), ...])`` for a binary-complement merge, or
    ``("not", [(J, K)])`` otherwise (or always, when ``not_only``).
    """
    if not_only:
        return ("not", [(j, k)])
    alt_j, alt_k = alt.get(rk_j, set()), alt.get(rk_k, set())
    # Clean binary-complement case: each side has exactly two disjoint alternatives.
    if len(alt_j) == 2 and len(alt_k) == 2 and j in alt_j and k in alt_k and k not in alt_j and j not in alt_k:
        other_j = (alt_j - {j}).pop()
        other_k = (alt_k - {k}).pop()
        return ("merge", [(j, other_k), (other_j, k)])
    return ("not", [(j, k)])


def classify_clashes(
    clashes: list[Clash],
    source: Path | gemmi.cif.Block,
    tree: HierarchyTree,
    coexistence: CoexistenceTable | None,
    *,
    not_only: bool = False,
    use_auth: bool = False,
) -> ClashReport:
    """Classify clashes and propose merge / NOT mitigations.

    A clash between atom A (state J) and atom B (state K) is *benign* if a NOT rule
    links ``anc*(J)`` and ``anc*(K)``, or the same atom name at A's residue is also
    assigned to a state in ``anc*(K)`` (or symmetrically for B) — where ``anc*(S)``
    is S plus its ancestors up to and including Base. Clashes where an atom is Base,
    or where J and K lie on one ancestor line, are warnings only. Everything else is
    actionable and reduces to a merge or NOT proposal.

    Args:
        clashes: Clashes from :func:`detect_clashes` (mutated in place with categories).
        source: The same mmCIF file/Block the clashes came from.
        tree: The hierarchy tree.
        coexistence: Existing coexistence rules, or None.
        not_only: If True, propose a NOT rule for every actionable clash.
        use_auth: If True, residue keys use ``auth_*`` numbering (match detection).

    Returns:
        A ClashReport with per-clash categories and deduplicated proposals.
    """
    block = _resolve_block(source)
    data = _read_atom_data(block, use_auth=use_auth)

    alt: dict[tuple[str, str], set[str]] = defaultdict(set)
    atom_states: dict[tuple[tuple[str, str], str], set[str]] = defaultdict(set)
    for row in range(len(data.serials)):
        rk, name, hid = data.residue_keys[row], data.atom_names[row], data.het_ids[row]
        atom_states[(rk, name)].add(hid)
        if hid != _BASE_ID:
            alt[rk].add(hid)

    not_pairs: set[frozenset[str]] = set()
    if coexistence is not None:
        for rule in coexistence.rules:
            if rule.rule == CoexistenceRule.NOT:
                for related in rule.heterogeneity_ids:
                    if rule.heterogeneity_id != related:
                        not_pairs.add(frozenset((rule.heterogeneity_id, related)))

    anc_cache: dict[str, set[str]] = {}

    def not_linked(j: str, k: str) -> bool:
        aj = _ancestors_inclusive(tree, j, anc_cache)
        ak = _ancestors_inclusive(tree, k, anc_cache)
        return any(p != q and frozenset((p, q)) in not_pairs for p in aj for q in ak)

    merge_edges: set[frozenset[str]] = set()
    not_edges: set[frozenset[str]] = set()

    for clash in clashes:
        j, k = clash.atom1.het_id, clash.atom2.het_id
        if j == k:
            clash.category = CATEGORY_SAME_STATE
            clash.note = f"both atoms are in state {j!r}"
            continue

        aj = _ancestors_inclusive(tree, j, anc_cache)
        ak = _ancestors_inclusive(tree, k, anc_cache)
        if j in ak or k in aj:
            clash.category = CATEGORY_ANCESTOR
            clash.note = "Base or ancestor/descendant states always coexist; review manually"
            continue

        if not_linked(j, k):
            clash.category = CATEGORY_BENIGN
            clash.note = "states are already mutually exclusive (NOT rule)"
            continue
        if atom_states.get((clash.atom1.residue_key, clash.atom1.atom_name), set()) & ak:
            clash.category = CATEGORY_BENIGN
            clash.note = f"{clash.atom1.atom_name} has an alternative conformation in {k!r}'s branch"
            continue
        if atom_states.get((clash.atom2.residue_key, clash.atom2.atom_name), set()) & aj:
            clash.category = CATEGORY_BENIGN
            clash.note = f"{clash.atom2.atom_name} has an alternative conformation in {j!r}'s branch"
            continue

        clash.category = CATEGORY_ACTIONABLE
        action, pairs = _decide_pair(j, clash.atom1.residue_key, k, clash.atom2.residue_key, alt, not_only=not_only)
        for a, b in pairs:
            if a == b:
                continue
            (merge_edges if action == "merge" else not_edges).add(frozenset((a, b)))

    return _build_report(clashes, merge_edges, not_edges)


def _build_report(
    clashes: list[Clash],
    merge_edges: set[frozenset[str]],
    not_edges: set[frozenset[str]],
) -> ClashReport:
    """Assemble merge components and NOT proposals, recording contradictions."""
    uf = _StateUnionFind()
    for edge in merge_edges:
        a, b = tuple(edge)
        uf.union(a, b)

    merges = [MergeProposal(states=sorted(component)) for component in uf.components() if len(component) > 1]

    conflicts: list[str] = []
    not_rules: list[NotProposal] = []
    for edge in sorted(not_edges, key=lambda e: sorted(e)):
        a, b = sorted(edge)
        if merge_edges and uf.find(a) == uf.find(b):
            conflicts.append(f"states {a!r} and {b!r} are proposed for both a merge and a NOT rule; resolve manually")
            continue
        not_rules.append(NotProposal(states=[a, b]))

    return ClashReport(
        clashes=clashes,
        merges=sorted(merges, key=lambda m: m.states),
        not_rules=not_rules,
        conflicts=conflicts,
    )
