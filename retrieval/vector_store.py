"""
retrieval/vector_store.py
In-memory multi-vector index using ColBERT-style MaxSim scoring.

Why not FAISS?
  FAISS works with single flat vectors.
  ColPali produces *multi-vector* embeddings (one per image patch).
  Retrieval requires MaxSim: for each query token, find the max similarity
  across all document patch vectors, then sum over query tokens.
  This can't be expressed as a single ANN lookup — we compute it exactly.

For an assignment-scale dataset (≤ 500 pages), exact MaxSim is fast enough.
For production, use Qdrant's native multi-vector support instead.
"""

import torch
from dataclasses import dataclass, field


@dataclass
class IndexEntry:
    embedding: torch.Tensor   # [N_patches, 128]  on CPU
    metadata:  dict           # page record from the manifest


class MultiVectorIndex:
    """
    Stores (embedding, metadata) pairs and ranks them with MaxSim scoring.

    Usage:
        index = MultiVectorIndex()
        index.add(page_embedding, page_metadata)   # repeat for each page
        results = index.score(query_embedding)      # [(score, metadata), ...]
    """

    def __init__(self):
        self._entries: list[IndexEntry] = []

    # ── Building the index ────────────────────────────────────────────────────

    def add(self, embedding: torch.Tensor, metadata: dict) -> None:
        """
        Add a single page embedding to the index.

        Args:
            embedding: [N_patches, 128] tensor (CPU, float32 or bfloat16)
            metadata:  Page record dict from manifest (must include image_path etc.)
        """
        # Normalise to float32 for consistent scoring
        self._entries.append(
            IndexEntry(
                embedding=embedding.to(dtype=torch.float32),
                metadata=metadata,
            )
        )

    def add_batch(self, embeddings: list[torch.Tensor], metadatas: list[dict]) -> None:
        """Add multiple pages at once."""
        if len(embeddings) != len(metadatas):
            raise ValueError("embeddings and metadatas must have the same length")
        for emb, meta in zip(embeddings, metadatas):
            self.add(emb, meta)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def score(
        self,
        query_embedding: torch.Tensor,
        top_k: int | None = None,
    ) -> list[tuple[float, dict]]:
        """
        Rank all pages against a query using MaxSim (ColBERT late interaction).

        Score formula per document d:
            score(q, d) = Σ_{i ∈ query_tokens}  max_{j ∈ doc_patches}  cos_sim(q_i, d_j)

        Args:
            query_embedding: [N_query_tokens, 128] tensor
            top_k:           Return only the top-k results. None = return all.

        Returns:
            List of (score, metadata) sorted descending by score.
        """
        if not self._entries:
            return []

        q = query_embedding.to(dtype=torch.float32)   # [Q, 128]
        q = torch.nn.functional.normalize(q, dim=-1)  # L2-normalise query

        scores = []
        for entry in self._entries:
            d = torch.nn.functional.normalize(entry.embedding, dim=-1)  # [P, 128]

            # Similarity matrix: [Q, P]
            sim_matrix = torch.matmul(q, d.T)

            # MaxSim: for each query token take best-matching patch, then sum
            score = sim_matrix.max(dim=-1).values.sum().item()
            scores.append(score)

        # Sort descending
        ranked = sorted(
            zip(scores, [e.metadata for e in self._entries]),
            key=lambda x: x[0],
            reverse=True,
        )

        return ranked[:top_k] if top_k else ranked

    # ── Introspection ─────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"MultiVectorIndex({len(self._entries)} pages)"

    def get_all_metadata(self) -> list[dict]:
        return [e.metadata for e in self._entries]