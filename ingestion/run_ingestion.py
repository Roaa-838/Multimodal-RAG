"""
run_ingestion.py
Master ingestion pipeline — works with ANY PDF dataset.

Two modes:
  1. --pdf-dir  : scan a folder and ingest every PDF found (recommended)
  2. --local    : explicit file list (for fine-grained control)

Usage
-----
    # Ingest everything in data/pdfs/
    python -m ingestion.run_ingestion --pdf-dir data/pdfs

    # Ingest specific files with custom IDs
    python -m ingestion.run_ingestion \\
        --local path/to/report.pdf annual_report \\
        --local path/to/policy.pdf policy_doc

    # Custom output directory
    python -m ingestion.run_ingestion --pdf-dir data/pdfs --output data/extracted
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ingestion.pdf_loader    import load_pdfs_from_dir, load_pdfs_from_list
from ingestion.page_extractor import extract_pages
from ingestion.manifest       import create_manifest


# ── Config ────────────────────────────────────────────────────
DEFAULT_PDF_DIR = "data/pdfs"
DEFAULT_OUTPUT  = "data/extracted"


# ── Core pipeline ─────────────────────────────────────────────

def run(pdfs: dict[str, Path], output_dir: str | Path = DEFAULT_OUTPUT) -> dict:
    """
    Ingest a dict of {doc_id: pdf_path} and return the manifest.

    Args:
        pdfs       : mapping of doc_id → PDF path (from any loader).
        output_dir : root directory for extracted images + manifest.

    Returns:
        Manifest dict.
    """

    output_dir = Path(output_dir)

    print("=" * 60)
    print("  DSAI 413 — Multi-Modal RAG Ingestion Pipeline")
    print(f"  docs   : {len(pdfs)}")
    print(f"  output : {output_dir}")
    print("=" * 60)

    all_pages:    list[list[dict]] = []
    doc_metadata: dict[str, dict]  = {}

    for doc_id, pdf_path in pdfs.items():
        print(f"\n[ingesting] {doc_id}  ←  {pdf_path.name}")

        pages = extract_pages(
            pdf_path      = pdf_path,
            doc_id        = doc_id,
            output_dir    = output_dir,
            use_pdf2image = False,        # PyMuPDF fallback — no poppler needed
        )

        all_pages.append(pages)

        doc_metadata[doc_id] = {
            "description": doc_id.replace("_", " ").title(),
            "filename":    pdf_path.name,
            "url":         "",
        }

    if not all_pages:
        print("[error] No pages extracted. Exiting.")
        sys.exit(1)

    manifest = create_manifest(
        all_pages    = all_pages,
        doc_metadata = doc_metadata,
        output_dir   = output_dir,
    )

    print("\n✓  INGESTION COMPLETE")
    print(f"   manifest → {output_dir / 'manifest.json'}")
    print(f"   images   → {output_dir}/")
    print("\nNext step:")
    print(f"   python run_retrieval.py --manifest {output_dir / 'manifest.json'}")

    return manifest


# ── CLI ───────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-Modal RAG Ingestion — works with any PDF dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan a whole folder (easiest):
  python -m ingestion.run_ingestion --pdf-dir data/pdfs

  # Pick specific files:
  python -m ingestion.run_ingestion \\
      --local reports/q1.pdf q1_report \\
      --local reports/q2.pdf q2_report
        """,
    )

    source = parser.add_mutually_exclusive_group(required=True)

    source.add_argument(
        "--pdf-dir",
        metavar="DIR",
        help="Folder to scan for .pdf files (all PDFs will be ingested)",
    )
    source.add_argument(
        "--local",
        nargs=2,
        action="append",
        metavar=("PATH", "DOC_ID"),
        help="Explicit PDF: --local path/to/file.pdf my_doc_id  (repeatable)",
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output directory for images + manifest  (default: {DEFAULT_OUTPUT})",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # ── Resolve PDF sources ───────────────────────────────────
    if args.pdf_dir:
        try:
            pdfs = load_pdfs_from_dir(args.pdf_dir)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[error] {exc}")
            return 1
    else:
        # --local pairs: [(path, doc_id), ...]
        try:
            pdfs = load_pdfs_from_list([p for p, _ in args.local])
            # Override auto-generated doc_ids with user-supplied ones
            pdfs = {
                doc_id: list(pdfs.values())[i]
                for i, (_, doc_id) in enumerate(args.local)
                if i < len(pdfs)
            }
        except ValueError as exc:
            print(f"[error] {exc}")
            return 1

    run(pdfs, output_dir=args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())