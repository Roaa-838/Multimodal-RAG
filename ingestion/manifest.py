"""
manifest.py
Central index for the multi-modal RAG system.

Stores:
- document metadata
- page-level image paths
- extracted text
- citation information

Used by:
- embedding pipeline (ColPali)
- FAISS indexing
- retrieval + QA system
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


MANIFEST_FILENAME = "manifest.json"


# ─────────────────────────────────────────────────────────────
# MAIN BUILDER
# ─────────────────────────────────────────────────────────────
def create_manifest(
    all_pages: list[list[dict]],
    doc_metadata: dict[str, dict] | None,
    output_dir: str | Path,
) -> dict:
    """
    Build unified manifest from extracted page data.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    doc_metadata = doc_metadata or {}

    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "version": "1.0",
        "documents": {},
        "pages": [],
        "total_pages": 0,
        "total_docs": 0,
    }

    global_idx = 0

    # ─────────────────────────────────────────────
    # Build document + page structure
    # ─────────────────────────────────────────────
    for page_list in all_pages:
        if not page_list:
            continue

        doc_id = page_list[0]["doc_id"]
        meta = doc_metadata.get(doc_id, {})

        manifest["documents"][doc_id] = {
            "doc_id": doc_id,
            "description": meta.get("description", doc_id),
            "url": meta.get("url", ""),
            "filename": meta.get("filename", ""),
            "page_count": len(page_list),
            "first_global_idx": global_idx,
            "last_global_idx": global_idx + len(page_list) - 1,
        }

        for page in page_list:
            manifest["pages"].append({
                # identity
                "global_idx": global_idx,
                "doc_id": page["doc_id"],
                "page_num": page["page_num"],

                # multimodal content
                "image_path": page["image_path"],
                "text": page.get("text", ""),

                # metadata
                "text_length": len(page.get("text", "")),
                "width": page.get("width", 0),
                "height": page.get("height", 0),
                "dpi": page.get("dpi", 0),

                # retrieval helper
                "citation": _format_citation(page, meta),
            })

            global_idx += 1

    manifest["total_pages"] = global_idx
    manifest["total_docs"] = len(manifest["documents"])

    # ─────────────────────────────────────────────
    # Save manifest
    # ─────────────────────────────────────────────
    manifest_path = output_dir / MANIFEST_FILENAME

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n[manifest] saved → {manifest_path}")
    print(f"[summary] docs={manifest['total_docs']} pages={manifest['total_pages']}")
    _print_summary(manifest)

    return manifest


# ─────────────────────────────────────────────────────────────
# LOADERS / UTILITIES
# ─────────────────────────────────────────────────────────────
def load_manifest(manifest_path: str | Path) -> dict:
    """Load manifest from disk."""

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    required = {"documents", "pages", "total_pages", "total_docs"}

    missing = required - set(manifest.keys())
    if missing:
        raise ValueError(f"Manifest missing fields: {missing}")

    return manifest


def get_page_by_global_idx(manifest: dict, global_idx: int) -> dict:
    """Fetch page by global index."""

    if not (0 <= global_idx < manifest["total_pages"]):
        raise IndexError("global_idx out of range")

    return manifest["pages"][global_idx]


def get_pages_for_doc(manifest: dict, doc_id: str) -> list[dict]:
    """Return all pages for a document."""
    return [p for p in manifest["pages"] if p["doc_id"] == doc_id]


def get_all_image_paths(manifest: dict) -> list[str]:
    """Return image paths in correct retrieval order."""
    return [p["image_path"] for p in manifest["pages"]]


def get_all_texts(manifest: dict) -> list[str]:
    """Return all page texts."""
    return [p.get("text", "") for p in manifest["pages"]]


# ─────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────
def _format_citation(page: dict, meta: dict) -> str:
    """
    Create citation string for LLM grounding.
    """

    desc = meta.get("description", page["doc_id"])

    return f"{desc} — Page {page['page_num']}"


def _print_summary(manifest: dict) -> None:
    """Pretty print document summary."""

    print("\nDocuments:")
    print("-" * 60)

    for doc_id, info in manifest["documents"].items():
        print(f"{doc_id:<15} | pages: {info['page_count']:<4} | {info['description'][:40]}")


# ─────────────────────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/extracted/manifest.json"

    m = load_manifest(path)

    print(f"\nLoaded: {m['total_docs']} docs | {m['total_pages']} pages")
    print("\nFirst page:")
    print(json.dumps(m["pages"][0], indent=2))