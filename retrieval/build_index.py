"""
build_index.py
Standalone script that loads a manifest, embeds all pages with ColPali,
and writes the index to disk.

This is the retrieval-layer counterpart to ingestion/run_ingestion.py.
It can be imported as a module OR run directly as a script.

Usage (module):
    from retrieval.build_index import build_index
    build_index("data/extracted/manifest.json", "data/index")

Usage (script):
    python -m retrieval.build_index
    python -m retrieval.build_index --manifest data/extracted/manifest.json --output data/index
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from retrieval.embedder import ColPaliEmbedder


# ── Defaults ──────────────────────────────────────────────────
DEFAULT_MANIFEST = "data/extracted/manifest.json"
DEFAULT_OUTPUT   = "data/index"


# ── Public API ────────────────────────────────────────────────

def build_index(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    output_dir:    str | Path = DEFAULT_OUTPUT,
    batch_size:    int | None = None,
    overwrite:     bool       = False,
) -> Path:
    """
    Embed every page image in *manifest_path* with ColPali and save
    the updated manifest (with ``embedding_path`` fields) to *output_dir*.

    Args:
        manifest_path : path to manifest.json from the ingestion pipeline.
        output_dir    : directory for .pt embedding files and output manifest.
        batch_size    : images per forward pass (auto when None).
        overwrite     : if False, skip pages whose .pt file already exists.

    Returns:
        Path to the saved ``manifest_with_embeddings.json``.
    """

    manifest_path = Path(manifest_path)
    output_dir    = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            "Run ingestion first:  python -m ingestion.run_ingestion --local <PDF> <ID>"
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    pages = manifest["pages"]
    n     = len(pages)

    if batch_size is None:
        batch_size = 4 if torch.cuda.is_available() else 1

    print(f"\n[build_index] {n} pages | batch={batch_size} | output={output_dir}")

    embedder = ColPaliEmbedder()
    t0       = time.time()
    done     = 0
    skipped  = 0

    for i in tqdm(range(0, n, batch_size), desc="Embedding"):
        batch = pages[i : i + batch_size]

        images:      list[Image.Image] = []
        valid_batch: list[dict]        = []

        for page in batch:
            emb_path = output_dir / f"emb_{page['doc_id']}_{page['page_num']:04d}.pt"

            # Skip if already embedded and overwrite=False
            if emb_path.exists() and not overwrite:
                page["embedding_path"] = str(emb_path)
                skipped += 1
                continue

            img_path = Path(page["image_path"])
            if not img_path.exists():
                tqdm.write(f"[warn] missing image: {img_path}")
                continue

            images.append(Image.open(img_path).convert("RGB"))
            valid_batch.append((page, emb_path))

        if not images:
            continue

        embeddings = embedder.embed_batch(images)

        for (page, emb_path), emb in zip(valid_batch, embeddings):
            torch.save(emb, emb_path)
            page["embedding_path"] = str(emb_path)
            done += 1

    # ── Save updated manifest ─────────────────────────────────
    out_path = output_dir / "manifest_with_embeddings.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"\n[build_index] ✓  embedded={done}  skipped={skipped}  time={elapsed:.1f}s")
    print(f"[build_index] saved → {out_path}")

    return out_path


# ── CLI ───────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build ColPali index from manifest.")
    p.add_argument("--manifest",   default=DEFAULT_MANIFEST)
    p.add_argument("--output",     default=DEFAULT_OUTPUT)
    p.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    p.add_argument("--overwrite",  action="store_true",
                   help="Re-embed pages even if .pt files already exist")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_index(
        manifest_path = args.manifest,
        output_dir    = args.output,
        batch_size    = args.batch_size,
        overwrite     = args.overwrite,
    )