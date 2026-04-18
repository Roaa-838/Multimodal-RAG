"""
search.py
High-level search API.  Wraps embedder + MultiVectorIndex into a single
callable used by both the CLI (run_qa.py) and the Streamlit app (app.py).

Separating search logic here keeps query_engine.py and app.py thin.

Usage:
    from retrieval.search import Searcher

    searcher = Searcher(manifest)          # loads all embeddings into RAM
    results  = searcher.search("What is the inflation forecast?", top_k=4)
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from retrieval.embedder     import ColPaliEmbedder
from retrieval.vector_store import MultiVectorIndex
from retrieval.citations    import deduplicate_pages


# ── Public class ──────────────────────────────────────────────

class Searcher:
    """
    Stateful searcher — load once, query many times.

    Parameters
    ----------
    manifest : dict
        Manifest dict (must have ``"pages"`` list with ``"embedding_path"``
        on every entry — i.e. the output of build_index).
    embedder : ColPaliEmbedder | None
        Pass an existing embedder to avoid loading the model twice.
        A new one is created when None (default).
    """

    def __init__(self, manifest: dict, embedder: ColPaliEmbedder | None = None):
        self.manifest = manifest
        self.embedder = embedder or ColPaliEmbedder()
        self.index    = self._build_index()

    # ── Setup ─────────────────────────────────────────────────

    def _build_index(self) -> MultiVectorIndex:
        index = MultiVectorIndex()

        for page in self.manifest["pages"]:
            emb_path = page.get("embedding_path")

            if not emb_path or not Path(emb_path).exists():
                raise FileNotFoundError(
                    f"Embedding not found for page {page.get('page_num')} "
                    f"of {page.get('doc_id')}.\n"
                    "Run build_index first."
                )

            emb = torch.load(emb_path, weights_only=True)
            index.add(emb, page)

        return index

    # ── Public ────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 4, deduplicate: bool = True) -> list[dict]:
        """
        Embed *query* and return the top-k most relevant pages.

        Args:
            query       : natural-language question.
            top_k       : number of pages to return.
            deduplicate : remove duplicate (doc_id, page_num) pairs.

        Returns:
            List of page metadata dicts, each augmented with a ``"score"`` key.
        """

        q_emb   = self.embedder.embed_query(query)
        raw     = self.index.score(q_emb)

        results = [
            {**meta, "score": float(score), "query_embedding": q_emb}
            for score, meta in raw[:top_k]
        ]

        if deduplicate:
            results = deduplicate_pages(results)

        return results

    def search_with_query_emb(self, query: str, top_k: int = 4):
        """
        Like :meth:`search` but also returns the raw query embedding tensor —
        useful for downstream similarity-map visualisation.

        Returns:
            (results, q_emb)
        """

        q_emb   = self.embedder.embed_query(query)
        raw     = self.index.score(q_emb)

        results = [
            {**meta, "score": float(score)}
            for score, meta in raw[:top_k]
        ]

        results = deduplicate_pages(results)

        return results, q_emb


# ── Convenience function ──────────────────────────────────────

def load_searcher(
    manifest_path: str | Path,
    embedder: ColPaliEmbedder | None = None,
) -> Searcher:
    """
    Load a :class:`Searcher` from a ``manifest_with_embeddings.json`` path.

    Args:
        manifest_path : path to the index manifest.
        embedder      : optional pre-loaded embedder.

    Returns:
        Ready-to-use :class:`Searcher` instance.
    """

    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Index manifest not found: {manifest_path}\n"
            "Run:  python -m retrieval.build_index"
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    return Searcher(manifest, embedder=embedder)