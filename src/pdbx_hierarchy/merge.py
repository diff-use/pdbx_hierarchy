"""Merge a ground-state and a changed-state model into one hierarchical file.

A crystallographer holding two separately refined models of the same crystal — a
ground state and a changed state — gets back one extended PDBx/mmCIF file whose
hierarchy tree is ``Base`` -> {``Ground``, ``Changed``} and whose atoms are the
union of both inputs, each assigned to a state under the appropriate branch.

Unlike the rest of the toolbox, assembly happens at gemmi's ``Structure`` level
rather than at the ``cif.Block`` level. Copying one input's ``_entity`` /
``_struct_asym`` family while keeping both inputs' atom rows silently corrupts
the file: the same ``label_entity_id`` routinely names different components in
two independently deposited models, so preserving those tables would relabel one
input's atoms as the other's ligand. Reading both inputs as structures and
letting gemmi regenerate the entity bookkeeping from the coordinate content
solves that, at the cost of sharing no code with the block-level commands. That
cost is recorded deliberately; an integrated version needs it reconciled.

The regenerated ``_entity`` and ``_struct_asym`` are written out with the
coordinates, because ``_atom_site.label_entity_id`` and ``label_asym_id`` name
them on every row. Dropping the two tables while keeping those columns is the
other half of the same trap: not a file that mislabels its components, but a file
that cannot say what its components are.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, cast

import gemmi

from pdbx_hierarchy import altloc, occupancy
from pdbx_hierarchy.assignment import assign_from_alt_ids
from pdbx_hierarchy.exceptions import HierarchyNotFoundError, PdbxParseError, PdbxValidationError
from pdbx_hierarchy.io.reader import (
    count_atom_site_rows,
    has_hierarchy,
    read_atom_site_heterogeneity_ids,
    read_hierarchy,
    read_mmcif,
)
from pdbx_hierarchy.io.writer import write_atom_site_heterogeneity_ids, write_hierarchy, write_mmcif
from pdbx_hierarchy.models.hierarchy import HierarchyState, HierarchyTree
from pdbx_hierarchy.models.id_generator import HierarchyIdGenerator, is_canonical_id

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

BASE_STATE_ID = "Base"
BASE_STATE_NAME = "base_state"
GROUND_STATE_ID = "Ground"
GROUND_STATE_NAME = "ground_state"
CHANGED_STATE_ID = "Changed"
CHANGED_STATE_NAME = "changed_state"

#: The only categories the merged file carries. Everything else from both inputs
#: is dropped: experimental and refinement metadata, ``_struct_conn``, and the
#: polymer sequence tables.
#:
#: ``_entity`` and ``_struct_asym`` are here because ``_atom_site`` points at
#: them: every row carries a ``label_entity_id`` and a ``label_asym_id``, and
#: without the two tables those columns would name nothing. They are safe to
#: write only because they are regenerated from the merged coordinates rather
#: than copied from an input — see the module docstring.
OUTPUT_CATEGORIES = (
    "_entry.",
    "_cell.",
    "_symmetry.",
    "_chem_comp.",
    "_entity.",
    "_struct_asym.",
    "_atom_site.",
)

#: Columns compared to confirm that a walk over a Structure visits atoms in the
#: same order as the rows of the corresponding _atom_site loop.
_ORDER_CHECK_COLUMNS = ("label_comp_id", "label_atom_id", "label_alt_id")


class AltlocMap(NamedTuple):
    """One input's altloc relabelling, for the caller to print in full.

    Attributes:
        source_name: The input file's name.
        mapping: Old -> new ``label_alt_id``, with ``.`` for the blank label.
    """

    source_name: str
    mapping: dict[str, str]


class _ResidueKey(NamedTuple):
    """What makes a residue distinct when both inputs' residues are pooled.

    Component id is part of the key on purpose: a ligand in one input and a water
    in the other, sharing a residue number, must stay two residues rather than
    fusing into one residue with two "conformations".
    """

    seq_num: int
    insertion_code: str
    comp_id: str


@dataclass
class SourceModel:
    """One input model, with its hierarchy tree and per-atom state assignment.

    Attributes:
        path: The file this model was read from.
        block: The input's sole data block, kept for its ``_chem_comp`` table.
        structure: The coordinates, as a gemmi Structure.
        tree: The hierarchy tree, either read from the file or inferred.
        atom_state_ids: One state id per atom, in Structure walk order.
        reused_hierarchy: True when the tree came from the file rather than
            inference.
    """

    path: Path
    block: gemmi.cif.Block
    structure: gemmi.Structure
    tree: HierarchyTree
    atom_state_ids: list[str]
    reused_hierarchy: bool


@dataclass
class MergeReport:
    """What a merge did, for the caller to report to the user.

    Attributes:
        output_path: Where the merged file was written.
        tree: The merged hierarchy tree.
        changed_id_map: Old -> new state id for every non-root state of the
            changed input's tree.
        altloc_maps: Each input's altloc relabelling, ground first.
        notes: Human-readable observations worth printing.
    """

    output_path: Path
    tree: HierarchyTree
    changed_id_map: dict[str, str]
    altloc_maps: list[AltlocMap] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def merge_states(
    *,
    ground_path: Path,
    changed_path: Path,
    occ: float,
    output_path: Path,
) -> MergeReport:
    """Merge a ground and a changed model into one hierarchical file.

    Args:
        ground_path: The ground-state input.
        changed_path: The changed-state input.
        occ: The changed state's fraction of the crystal, ``0 < occ < 1``.
            Ground occupancies are scaled by ``1 - occ`` and changed occupancies
            by ``occ``; the value is also recorded as provenance.
        output_path: Where to write the merged file. Its stem becomes both the
            data block name and ``_entry.id``.

    Returns:
        A MergeReport describing what was written.

    Raises:
        FileNotFoundError: If either input does not exist.
        PdbxParseError: If an input is malformed or contains no atoms.
        PdbxValidationError: If either input has an atom position whose
            occupancies already sum above ``1.00``, if the two inputs need more
            altloc labels between them than the alphabet has, if the two inputs
            define the same chemical component with differing formulae, or if the
            assembled file's atom order cannot be reconciled with the hierarchy
            assignments.
    """
    ground = load_source(ground_path)
    changed = load_source(changed_path)

    _rename_root(ground, GROUND_STATE_ID, GROUND_STATE_NAME)
    _rename_root(changed, CHANGED_STATE_ID, CHANGED_STATE_NAME)
    changed_id_map = _deconflict_state_ids(ground.tree, changed)
    altloc_maps = _partition_altlocs(ground, changed)

    occupancy_notes = _scale_occupancies(ground, changed, occ)
    tree = _build_merged_tree(ground, changed, occ)
    structure, atom_state_ids = _assemble_structure(ground, changed, name=output_path.stem)
    chem_comp = _union_chem_comp([ground, changed])

    doc = _build_document(structure, tree, atom_state_ids, chem_comp)
    write_mmcif(doc, output_path)

    notes = [
        f"{source.path.name}: reused the hierarchy already in the file"
        for source in (ground, changed)
        if source.reused_hierarchy
    ]
    notes.extend(occupancy_notes)
    return MergeReport(
        output_path=output_path,
        tree=tree,
        changed_id_map=changed_id_map,
        altloc_maps=altloc_maps,
        notes=notes,
    )


# === Reading an input ===


def load_source(path: Path) -> SourceModel:
    """Read one input and obtain a hierarchy tree and per-atom assignment for it.

    A file that already carries a hierarchy has that tree reused, so hand-tuned
    work on a branch is not thrown away. Otherwise a tree is inferred from
    ``label_alt_id``. Inference deliberately runs here, on the input's original
    altloc labels: later stages rewrite blanks into real letters, and inferring
    afterwards would read almost every atom in a typical model as a conformer.

    Args:
        path: The mmCIF file to read.

    Returns:
        The loaded SourceModel.

    Raises:
        FileNotFoundError: If the file does not exist.
        PdbxParseError: If the file is malformed, contains no atoms, or stores
            its atoms in an order a Structure walk does not reproduce.
        PdbxValidationError: If the file holds more than one model, or has a
            hierarchy but no per-atom assignments — leaving nowhere to put its
            atoms.
    """
    block = read_mmcif(path)
    if count_atom_site_rows(block) == 0:
        raise PdbxParseError(f"{path.name} contains no atoms")

    if has_hierarchy(block):
        tree = read_hierarchy(block)
        try:
            atom_state_ids = read_atom_site_heterogeneity_ids(block)
        except HierarchyNotFoundError as exc:
            raise PdbxValidationError(
                f"{path.name} has a hierarchy table but no _atom_site.pdbx_heterogeneity_id "
                f"column, so its atoms cannot be placed in the merged tree"
            ) from exc
        reused = True
    else:
        tree, atom_state_ids = assign_from_alt_ids(block)
        reused = False

    structure = gemmi.make_structure_from_block(block)
    # Only the first model is assembled, so a multi-model input would lose atoms.
    if len(structure) > 1:
        raise PdbxValidationError(
            f"{path.name} contains {len(structure)} models; merge-states reads single-model files"
        )
    _check_walk_matches_rows(path.name, block, structure)

    # Everything downstream reasons about occupancy in whole hundredths, so the
    # input's values are put on that grid here rather than at the point of use.
    occupancy.round_onto_grid(_iter_atoms(structure))

    walk_length = sum(1 for _ in _iter_atoms(structure))
    if walk_length != len(atom_state_ids):
        raise PdbxParseError(
            f"{path.name}: {walk_length} atom(s) in the structure but {len(atom_state_ids)} assignment(s)"
        )

    return SourceModel(
        path=path,
        block=block,
        structure=structure,
        tree=tree,
        atom_state_ids=atom_state_ids,
        reused_hierarchy=reused,
    )


def _iter_atoms(structure: gemmi.Structure) -> Iterator[tuple[gemmi.Chain, gemmi.Residue, gemmi.Atom]]:
    """Walk the first model's atoms in the order gemmi writes them to _atom_site."""
    for chain in structure[0]:
        for residue in chain:
            for atom in residue:
                yield chain, residue, atom


def _check_walk_matches_rows(label: str, block: gemmi.cif.Block, structure: gemmi.Structure) -> None:
    """Confirm a Structure walk visits atoms in _atom_site row order.

    Per-atom state assignments are indexed by row, and the merge carries them
    across a Structure. gemmi builds a Structure by appending rows in order and
    writes _atom_site by walking one, so the two agree for every real file — but
    a file interleaving one residue's rows with another's would break the
    correspondence silently, which is worth a check rather than a comment.

    Args:
        label: What to call the file in the error message.
        block: The data block holding the _atom_site loop.
        structure: The Structure to walk.

    Raises:
        PdbxParseError: If the orders disagree.
    """
    columns = [tag for tag in _ORDER_CHECK_COLUMNS if block.find_loop(f"_atom_site.{tag}")]
    if not columns:
        return
    table = block.find("_atom_site.", columns)
    from_rows = [tuple(row[i] for i in range(len(columns))) for row in table]
    from_walk = [
        tuple(_walk_value(name, residue, atom) for name in columns) for _, residue, atom in _iter_atoms(structure)
    ]
    if from_rows != from_walk:
        raise PdbxParseError(
            f"{label}: _atom_site rows are not in the order gemmi reads them into a structure, "
            f"so per-atom state assignments cannot be carried across the merge"
        )


def _walk_value(column: str, residue: gemmi.Residue, atom: gemmi.Atom) -> str:
    """Return the value a Structure walk yields for one of the order-check columns."""
    if column == "label_comp_id":
        return residue.name
    if column == "label_atom_id":
        return atom.name
    return altloc.read_altloc(atom)


# === Tree surgery ===


def _apply_id_map(source: SourceModel, id_map: dict[str, str], rename: Callable[[HierarchyState], str]) -> None:
    """Rewrite a source's state ids in place, following its atoms and children.

    Args:
        source: The model to rewrite; its tree and its per-atom assignments are
            both carried through the map.
        id_map: Old -> new state id. Ids absent from the map are left alone.
        rename: Returns the new name for a state, given the state as it was
            before the remap.

    Raises:
        InvalidHierarchyError: If the result violates a tree invariant — a new id
            or name colliding with another state in the same tree, say.
    """
    source.tree = HierarchyTree(
        states=[
            HierarchyState(
                id=id_map.get(state.id, state.id),
                name=rename(state),
                parent=id_map.get(state.parent, state.parent) if state.parent is not None else None,
                details=state.details,
            )
            for state in source.tree.states
        ]
    )
    source.atom_state_ids = [id_map.get(state_id, state_id) for state_id in source.atom_state_ids]


def _rename_root(source: SourceModel, new_id: str, new_name: str) -> None:
    """Rename a source's root state to ``Ground`` or ``Changed``, in place.

    Args:
        source: The model whose root is renamed.
        new_id: The root's new id.
        new_name: The root's new name.
    """
    old = source.tree.get_root()
    if old.id == new_id and old.name == new_name:
        return
    _apply_id_map(source, {old.id: new_id}, lambda state: new_name if state.id == old.id else state.name)


def _deconflict_state_ids(ground_tree: HierarchyTree, changed: SourceModel) -> dict[str, str]:
    """Move the changed tree's non-root state ids above the ground tree's highest.

    Both trees come out of inference using the same ``A``, ``B``, ... sequence, so
    merging them unchanged would produce duplicate ids and a tree the model
    rejects. The ground model is usually the source of truth, so it keeps its ids
    and only the changed tree is renumbered. Names are regenerated as
    ``state_{new_id}`` because ``name`` is unique in the format and would
    otherwise collide exactly as ids do.

    Args:
        ground_tree: The ground tree, left untouched.
        changed: The changed model; its tree and per-atom assignments are
            rewritten.

    Returns:
        Old -> new id for every non-root state of the changed tree.
    """
    root = changed.tree.get_root()
    non_root = [state.id for state in changed.tree.get_descendants(root.id)]
    taken = {state.id for state in ground_tree.states} | {BASE_STATE_ID, GROUND_STATE_ID, CHANGED_STATE_ID}
    # A reused ground hierarchy can already hold a name of the state_X form, so an
    # id is only free if the name it generates is free too.
    taken_names = {state.name for state in ground_tree.states} | {BASE_STATE_NAME, GROUND_STATE_NAME}
    new_ids = _next_free_ids(taken, len(non_root), reject=lambda candidate: _state_name(candidate) in taken_names)
    mapping = dict(zip(non_root, new_ids, strict=True))

    _apply_id_map(
        changed,
        mapping,
        lambda state: _state_name(mapping[state.id]) if state.id in mapping else state.name,
    )
    return mapping


def _state_name(state_id: str) -> str:
    """Return the regenerated name for a reassigned state."""
    return f"state_{state_id}"


def _next_free_ids(
    taken: set[str],
    count: int,
    *,
    reject: Callable[[str], bool] | None = None,
) -> list[str]:
    """Return ``count`` canonical ids, all above every canonical id in ``taken``.

    "Above" means later in the ``A``, ``B``, ..., ``Z``, ``AA``, ... sequence the
    id generator produces, so a gap in the ids already used is skipped past
    rather than filled.

    Args:
        taken: Ids that must not be produced, canonical or not.
        count: How many ids to return.
        reject: Optional extra veto on a candidate id — used to skip an id whose
            generated state name would collide.

    Returns:
        The requested ids, in sequence order.
    """
    generator = HierarchyIdGenerator()
    remaining = {state_id for state_id in taken if is_canonical_id(state_id)}
    while remaining:
        remaining.discard(next(generator))

    ids: list[str] = []
    while len(ids) < count:
        candidate = next(generator)
        if candidate not in taken and not (reject is not None and reject(candidate)):
            ids.append(candidate)
    return ids


def _build_merged_tree(ground: SourceModel, changed: SourceModel, occ: float) -> HierarchyTree:
    """Insert a new ``Base`` root above both branches.

    ``Base`` owns no atoms: a future extension collapsing similar states between
    the two branches would give it some, but nothing here does.

    Args:
        ground: The ground model, root already renamed and ids untouched.
        changed: The changed model, root renamed and non-root ids deconflicted.
        occ: The changed state's fraction of the crystal, recorded as provenance.

    Returns:
        The merged tree: ``Base`` -> {``Ground``, ``Changed``} with each branch's
        own states beneath it.
    """
    ground_token = _provenance_token(ground.path)
    changed_token = _provenance_token(changed.path)
    base = HierarchyState(
        id=BASE_STATE_ID,
        name=BASE_STATE_NAME,
        parent=None,
        details=f"ground={ground_token};changed={changed_token};occ={occ:.2f}",
    )
    states = [base]
    states.extend(_reparented(ground, f"source={ground_token};occupancy_factor={1 - occ:.2f}"))
    states.extend(_reparented(changed, f"source={changed_token};occupancy_factor={occ:.2f}"))
    return HierarchyTree(states=states)


def _reparented(source: SourceModel, root_details: str) -> list[HierarchyState]:
    """Return a source's states with its root hung under ``Base`` and given provenance."""
    root_id = source.tree.get_root().id
    return [
        state.model_copy(update={"parent": BASE_STATE_ID, "details": root_details}) if state.id == root_id else state
        for state in source.tree.states
    ]


def _provenance_token(path: Path) -> str:
    """Return a whitespace-free form of a filename, safe as an unquoted mmCIF value."""
    return re.sub(r"\s+", "_", path.name)


# === Altloc labels ===


def _partition_altlocs(ground: SourceModel, changed: SourceModel) -> list[AltlocMap]:
    """Give every atom in both inputs a real altloc label, ground's first.

    Sits here in the pipeline for two reasons: after state assignment, which is
    what the ``altloc`` module's ordering note is about, and before assembly,
    which pools the two inputs' atoms into shared residues — coherent only once
    no label means two things. See that module for the alphabet and its limits.

    Args:
        ground: The ground model, relabelled from the start of the alphabet.
        changed: The changed model, relabelled above ground's highest label.

    Returns:
        Each input's old -> new mapping, ground first, for the caller to print.

    Raises:
        PdbxValidationError: If the two inputs need more than the 62 available
            labels between them.
    """
    sources = (ground, changed)
    # Materialised because each input's atoms are read twice: once for their
    # labels, once to write the new ones back.
    atoms = [[atom for _, _, atom in _iter_atoms(source.structure)] for source in sources]
    mappings = altloc.partition([altloc.original_labels(group) for group in atoms])
    for group, mapping in zip(atoms, mappings, strict=True):
        altloc.relabel(group, mapping)
    return [AltlocMap(source.path.name, mapping) for source, mapping in zip(sources, mappings, strict=True)]


# === Occupancy ===


def _scale_occupancies(ground: SourceModel, changed: SourceModel, occ: float) -> list[str]:
    """Give each input its share of the crystal, in place.

    Runs before assembly, on each input separately, because that is where an
    atom's provenance still says which factor it gets. Scaling the assembled
    structure instead would need the ground/changed split threaded through it,
    and each input's own occupancies are what the input validation is about.

    Doing it per input is also what makes the merged bound hold without any
    cross-input arithmetic: each input group is held to the floor of its exact
    share, and the two shares sum to the whole. See the ``occupancy`` module.

    Args:
        ground: The ground model, scaled by ``1 - occ``.
        changed: The changed model, scaled by ``occ``.
        occ: The changed state's fraction of the crystal.

    Returns:
        Notes worth reporting to the user — at most one, carrying the count of
        zero-occupancy atoms across both inputs.

    Raises:
        PdbxValidationError: If either input has an atom position whose
            occupancies already sum above ``1.00``.
    """
    ground_factor, changed_factor = occupancy.split_factors(occ)
    zero_count = sum(
        occupancy.scale_to_share(_iter_atoms(source.structure), factor=factor, label=source.path.name)
        for source, factor in ((ground, ground_factor), (changed, changed_factor))
    )

    if zero_count == 0:
        return []
    return [f"{zero_count} atom(s) carry zero occupancy in the inputs and are written as 0.00"]


# === Coordinate assembly ===


def _assemble_structure(ground: SourceModel, changed: SourceModel, *, name: str) -> tuple[gemmi.Structure, list[str]]:
    """Build one Structure holding both inputs' atoms, ground first.

    Chains are matched by their authored name — ``auth_asym_id`` does correspond
    between two models of the same crystal even where label identity does not.
    Residues are matched by number, insertion code, and component id, so a
    polymer residue present in both inputs becomes one residue carrying both
    inputs' atoms, while a ligand and a water sharing a number stay separate.

    Args:
        ground: The ground model.
        changed: The changed model.
        name: The entry name, used for the block name and ``_entry.id``.

    Returns:
        Tuple of (assembled Structure, one state id per atom in walk order).

    Raises:
        PdbxParseError: If a residue carries no sequence number, leaving nothing
            to key it by.
    """
    # Accumulated in pure Python and materialised in one pass at the end: holding
    # a reference to a gemmi Residue while more residues are added to its chain
    # dangles when the underlying vector reallocates.
    chain_order: list[str] = []
    residue_order: dict[str, list[_ResidueKey]] = {}
    residue_atoms: dict[tuple[str, _ResidueKey], list[tuple[gemmi.Atom, str]]] = {}

    for source in (ground, changed):
        walk = zip(_iter_atoms(source.structure), source.atom_state_ids, strict=True)
        for (chain, residue, atom), state_id in walk:
            if chain.name not in residue_order:
                chain_order.append(chain.name)
                residue_order[chain.name] = []
            if residue.seqid.num is None:
                raise PdbxParseError(f"{source.path.name}: residue {residue.name!r} has no sequence number")
            key = _ResidueKey(residue.seqid.num, residue.seqid.icode, residue.name)
            if (chain.name, key) not in residue_atoms:
                residue_order[chain.name].append(key)
                residue_atoms[(chain.name, key)] = []
            residue_atoms[(chain.name, key)].append((atom.clone(), state_id))

    structure = gemmi.Structure()
    structure.name = name
    structure.cell = ground.structure.cell
    structure.spacegroup_hm = ground.structure.spacegroup_hm

    model = gemmi.Model(1)
    atom_state_ids: list[str] = []
    for chain_name in chain_order:
        chain = gemmi.Chain(chain_name)
        for key in residue_order[chain_name]:
            residue = gemmi.Residue()
            residue.seqid = gemmi.SeqId(key.seq_num, key.insertion_code)
            residue.name = key.comp_id
            for atom, state_id in residue_atoms[(chain_name, key)]:
                residue.add_atom(atom)
                atom_state_ids.append(state_id)
            chain.add_residue(residue)
        model.add_chain(chain)
    structure.add_model(model)

    # Regenerate the entity bookkeeping from the coordinates. This is the call
    # that makes the merged file self-consistent instead of carrying one input's
    # label_* meanings over the other input's atoms.
    structure.setup_entities()
    structure.assign_label_seq_id(True)
    _number_entities(structure)
    return structure, atom_state_ids


def _number_entities(structure: gemmi.Structure) -> None:
    """Rename the regenerated entities to ``1``, ``2``, ... in order.

    gemmi names a regenerated entity after the content it found — ``water``, or
    ``EDO!`` for a non-polymer — and writes that name into
    ``_atom_site.label_entity_id``, where the convention is an integer. The
    merged file carries ``_entity`` itself, so the same name lands in
    ``_entity.id`` too; numbering keeps both columns conventional rather than
    exposing gemmi's internal naming.

    Args:
        structure: The assembled structure, after ``setup_entities``.
    """
    for number, entity in enumerate(structure.entities, start=1):
        entity.name = str(number)


def _union_chem_comp(sources: Sequence[SourceModel]) -> dict[str, list[str]]:
    """Union both inputs' ``_chem_comp`` tables, keyed by component id.

    A component id is a globally meaningful key needing no remapping, which is
    what makes this table safe to merge when the entity family is not. A ligand
    present in only one input is still defined in the merged file.

    Args:
        sources: The input models, in output order.

    Returns:
        The unioned table as a raw mmCIF category, empty if neither input has one.

    Raises:
        PdbxValidationError: If a component id appears in both inputs with
            differing formulae — a merged file must never claim one ligand's
            atoms belong to a different ligand.
    """
    columns: list[str] = []
    rows: dict[str, dict[str, str]] = {}
    origin: dict[str, Path] = {}

    for source in sources:
        category = source.block.get_mmcif_category("_chem_comp.", raw=True)
        if not category or "id" not in category:
            continue
        for column in category:
            if column not in columns:
                columns.append(column)
        for index in range(len(category["id"])):
            row = {column: values[index] for column, values in category.items()}
            comp_id = row["id"]
            previous = rows.get(comp_id)
            if previous is None:
                rows[comp_id] = row
                origin[comp_id] = source.path
                continue
            old, new = previous.get("formula"), row.get("formula")
            if old is not None and new is not None and old != new:
                raise PdbxValidationError(
                    f"_chem_comp {comp_id!r} is defined as {old} in {origin[comp_id].name} "
                    f"and as {new} in {source.path.name}"
                )
            previous.update({column: value for column, value in row.items() if column not in previous})

    if not rows:
        return {}
    ordered = ["id", *[column for column in columns if column != "id"]]
    return {column: [row.get(column, "?") for row in rows.values()] for column in ordered}


def _build_document(
    structure: gemmi.Structure,
    tree: HierarchyTree,
    atom_state_ids: list[str],
    chem_comp: dict[str, list[str]],
) -> gemmi.cif.Document:
    """Turn the assembled structure and tree into the document to write.

    Args:
        structure: The assembled coordinates.
        tree: The merged hierarchy tree.
        atom_state_ids: One state id per atom, in Structure walk order.
        chem_comp: The unioned ``_chem_comp`` category; when empty, the table
            gemmi derived from the coordinates is kept instead.

    Returns:
        The document to write.

    Raises:
        PdbxParseError: If the written atom order does not match the walk the
            assignments were built from.
        PdbxValidationError: If the assignment count does not match the written
            row count.
    """
    doc = structure.make_mmcif_document()
    block = doc.sole_block()

    for category in _categories(block):
        if category not in OUTPUT_CATEGORIES or (category == "_chem_comp." and chem_comp):
            block.find_mmcif_category(category).erase()
    if chem_comp:
        block.set_mmcif_category("_chem_comp.", chem_comp, raw=True)

    _check_walk_matches_rows("the merged file", block, structure)
    _write_two_decimal_occupancies(block)
    write_hierarchy(block, tree, overwrite=True)
    write_atom_site_heterogeneity_ids(block, atom_state_ids, overwrite=True)
    return doc


def _write_two_decimal_occupancies(block: gemmi.cif.Block) -> None:
    """Rewrite the occupancy column with two decimal places on every row.

    gemmi trims a trailing zero, writing ``0.4`` where every other coordinate
    file in a crystallographer's directory says ``0.40``. The scaled values are
    whole hundredths by construction, so this only restores the notation; it
    changes no number.

    Args:
        block: The merged file's block, after the coordinates are written.
    """
    column = block.find_loop("_atom_site.occupancy")
    for index, value in enumerate(list(column)):
        # gemmi writes this column from the structure, so every cell is a number.
        # An unknown marker is left as it is rather than crashing on float().
        if value in {".", "?"}:
            continue
        column[index] = f"{float(value):.2f}"


def _categories(block: gemmi.cif.Block) -> list[str]:
    """Return the mmCIF category prefixes present in a block, in file order."""
    found: list[str] = []
    for item in block:
        # gemmi types Item.pair as object and Item.loop as non-optional; at runtime
        # exactly one of the two is set for the items we care about.
        pair = cast("tuple[str, str] | None", item.pair)
        if pair is not None:
            category = pair[0].split(".")[0] + "."
        elif item.loop is not None:
            category = item.loop.tags[0].split(".")[0] + "."
        else:
            continue
        if category not in found:
            found.append(category)
    return found
