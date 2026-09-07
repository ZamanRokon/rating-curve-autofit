"""Unified entry point; each method keeps its own input and output contract."""
from __future__ import annotations

import argparse
import sys

from . import __version__


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="rating-curve",
        description="Fit stage-discharge curves with additive controls or smooth nested validation.",
        epilog="Use 'rating-curve additive --help' or 'rating-curve validated --help' for method settings.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("method", choices=["additive", "validated"], nargs="?",
                        help="additive: 1–3 controls with bootstrap; validated: 1–2 smooth regimes and daily estimates")
    if not argv or argv[0] in {"-h", "--help", "--version"}:
        parser.parse_args(argv)
        if not argv:
            parser.print_help()
        return
    method = parser.parse_args(argv[:1]).method
    if method == "additive":
        from .additive import main as run
    else:
        from .validated import main as run
    run(argv[1:])
