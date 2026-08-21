#!/usr/bin/env python
"""Widen a file's ``label_alt_id`` labels out of the single-character alphabet.

    uv run python scripts/widen_altlocs.py merged.cif [-o merged_widened.cif]

For a user whose downstream reader folds case and so cannot tell ``a`` from
``A``. Every label becomes one of ``A``, ``B``, ..., ``Z``, ``AA``, ``AB``, ...,
which stay distinct under that fold. See :mod:`pdbx_hierarchy.altloc` for the
trade this makes.

A script rather than a ``pdbx-hierarchy`` subcommand, deliberately: it is
prototype-grade, it is a separate transformation from any command, and its output
cannot be read back into a gemmi ``Structure``, which every other write path in
the toolbox depends on. Being outside the packaged CLI keeps that one-way
conversion something a user has to reach for by name.

``argparse`` rather than the typer the toolbox's real CLI uses, for the same
reason. typer is how ``pdbx-hierarchy``'s commands are written, so a second typer
app is the thing most likely to be mistaken for one of them — and this must not
be. Nothing here needs more than the standard library gives.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pdbx_hierarchy.altloc import WIDENING_NAMESPACE_NOTE, widen_file
from pdbx_hierarchy.exceptions import PdbxHierarchyError

#: Appended to the input's stem when no output path is given. Never in place: the
#: conversion is one-way, so overwriting the input would destroy the only copy
#: that still round-trips through gemmi.
_OUTPUT_SUFFIX = "_widened"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Arguments to parse, or None to read ``sys.argv``.

    Returns:
        The parsed arguments: ``source``, ``output``, ``force``.
    """
    parser = argparse.ArgumentParser(
        description="Widen a file's label_alt_id labels out of the single-character alphabet.",
        allow_abbrev=False,
    )
    parser.add_argument("source", type=Path, help="the mmCIF file to widen")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=f"where to write the widened file (default: <stem>{_OUTPUT_SUFFIX}.cif beside the input)",
    )
    parser.add_argument("-f", "--force", action="store_true", help="overwrite the output file if it exists")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Widen one file's altloc labels and print the mapping.

    Args:
        argv: Command-line arguments, or None to read ``sys.argv``.

    Returns:
        A process exit status: 0 on success, 1 on a bad input or a refused
        overwrite.
    """
    args = _parse_args(argv)
    output = args.output or args.source.with_name(f"{args.source.stem}{_OUTPUT_SUFFIX}.cif")

    if output == args.source:
        print(f"error: {output} is the input; widening is one-way and writes a new file", file=sys.stderr)
        return 1
    if output.exists() and not args.force:
        print(f"error: {output} already exists; pass --force to overwrite", file=sys.stderr)
        return 1

    try:
        mapping = widen_file(args.source, output)
    except (OSError, PdbxHierarchyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {output}")
    print(f"label_alt_id mapping ({len(mapping)} label(s)):")
    for old, new in mapping.items():
        print(f"  {old} -> {new}")
    print(f"note: {WIDENING_NAMESPACE_NOTE}")
    print("note: this is one-way; gemmi cannot read a multi-character label_alt_id back into an atom")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
