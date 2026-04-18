"""
pdf_loader.py
Dynamically discovers and validates PDFs from a folder or explicit paths.

No hardcoded filenames — works with ANY PDF dataset.

Usage:
    # Scan a folder
    pdfs = load_pdfs_from_dir("data/pdfs")

    # Load explicit files
    pdfs = load_pdfs_from_list(["path/a.pdf", "path/b.pdf"])

    # CLI test
    python -m ingestion.pdf_loader --dir data/pdfs
"""

from __future__ import annotations

import argparse
from pathlib import Path


# ── Public API ────────────────────────────────────────────────

def load_pdfs_from_dir(pdf_dir: str | Path) -> dict[str, Path]:
    """
    Scan *pdf_dir* and return all valid PDFs as {doc_id: Path}.
    doc_id is the filename stem, e.g. "annual_report_2023".

    Args:
        pdf_dir: folder containing .pdf files (searched non-recursively).

    Returns:
        Ordered dict mapping doc_id → resolved Path.

    Raises:
        FileNotFoundError: when the directory does not exist.
        ValueError: when no valid PDFs are found.
    """

    pdf_dir = Path(pdf_dir)

    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    candidates = sorted(pdf_dir.glob("*.pdf"))

    if not candidates:
        raise ValueError(
            f"No .pdf files found in: {pdf_dir}\n"
            "Place your PDF files there and re-run."
        )

    result: dict[str, Path] = {}

    for path in candidates:
        if _is_valid_pdf(path):
            doc_id = _safe_doc_id(path.stem)
            result[doc_id] = path.resolve()
        else:
            print(f"[warn] Skipping invalid PDF: {path.name}")

    if not result:
        raise ValueError(f"No valid PDFs found in: {pdf_dir}")

    print(f"[pdf_loader] Found {len(result)} PDFs in '{pdf_dir}':")
    for doc_id, path in result.items():
        print(f"  {doc_id:<40} {path.name}")

    return result


def load_pdfs_from_list(paths: list[str | Path]) -> dict[str, Path]:
    """
    Load PDFs from an explicit list of file paths.

    Args:
        paths: list of file paths (str or Path).

    Returns:
        dict mapping doc_id → resolved Path.
    """

    result: dict[str, Path] = {}

    for p in paths:
        path = Path(p).resolve()

        if not path.exists():
            print(f"[warn] File not found: {path}")
            continue

        if not _is_valid_pdf(path):
            print(f"[warn] Not a valid PDF: {path.name}")
            continue

        doc_id = _safe_doc_id(path.stem)
        result[doc_id] = path

    if not result:
        raise ValueError("No valid PDFs could be loaded from the provided list.")

    return result


def load_all_pdfs(pdf_dir: str | Path = "data/pdfs") -> dict[str, Path]:
    """
    Convenience wrapper — loads all PDFs from the default folder.
    Drop-in replacement for the old hardcoded version.
    """
    return load_pdfs_from_dir(pdf_dir)


# ── Internal helpers ──────────────────────────────────────────

def _is_valid_pdf(path: Path) -> bool:
    """Check the PDF magic bytes (%PDF header)."""
    try:
        with open(path, "rb") as f:
            return f.read(5).startswith(b"%PDF")
    except OSError:
        return False


def _safe_doc_id(stem: str) -> str:
    """
    Convert a filename stem into a safe doc_id:
    spaces and special chars → underscores, lowercased.
    e.g. 'IMF Article IV 2023' → 'imf_article_iv_2023'
    """
    import re
    return re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List discoverable PDFs.")
    parser.add_argument("--dir", default="data/pdfs", help="PDF folder to scan")
    args = parser.parse_args()

    pdfs = load_pdfs_from_dir(args.dir)
    print(f"\nTotal: {len(pdfs)} PDFs ready for ingestion.")