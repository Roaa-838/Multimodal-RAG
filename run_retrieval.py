"""
run_retrieval.py
CLI: embed all pages in a manifest and save the ColPali index.

Usage
-----
    python run_retrieval.py
    python run_retrieval.py --manifest data/extracted/manifest.json --output data/index
    python run_retrieval.py --batch-size 2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ── correct import ────────────────────────────────────────────
from retrieval.build_index import build_index


DEFAULT_MANIFEST = "data/extracted/manifest.json"
DEFAULT_OUTPUT   = "data/index"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ColPali multi-vector index from an ingestion manifest."
    )
    parser.add_argument("--manifest",   default=DEFAULT_MANIFEST)
    parser.add_argument("--output",     default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    manifest_path = Path(args.manifest)
    output_dir    = Path(args.output)

    if not manifest_path.exists():
        print(
            f"[error] Manifest not found: {manifest_path}\n"
            "        Run ingestion first:\n"
            "            python -m ingestion.run_ingestion --pdf-dir data/pdfs"
        )
        return 1

    print("=" * 60)
    print("  DSAI 413 — ColPali Indexing Pipeline")
    print(f"  manifest  : {manifest_path}")
    print(f"  output    : {output_dir}")
    print("=" * 60)

    t0 = time.time()

    try:
        out_path = build_index(
            manifest_path = manifest_path,
            output_dir    = output_dir,
            batch_size    = args.batch_size,
        )
    except Exception as exc:
        print(f"\n[fatal] Indexing failed: {exc}")
        return 1

    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"  ✓  INDEXING COMPLETE  ({elapsed:.1f}s)")
    print(f"     index → {out_path}")
    print(f"{'=' * 60}")
    print(f"\nNext step:")
    print(f"  python run_qa.py --manifest {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())