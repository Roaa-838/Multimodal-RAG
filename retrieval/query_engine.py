import torch
from PIL import Image

from retrieval.embedder import ColPaliEmbedder
from retrieval.vector_store import MultiVectorIndex


class QueryEngine:
    def __init__(self, manifest):
        self.manifest = manifest
        self.embedder = ColPaliEmbedder()
        self.index = MultiVectorIndex()

        self._load_index()

    def _load_index(self):
        """
        Load all page embeddings into memory
        """
        for page in self.manifest["pages"]:
            emb = torch.load(page["embedding_path"])
            self.index.add(emb, page)

    def retrieve(self, query: str, top_k: int = 5):
        """
        Embed query and run MaxSim retrieval
        """

        q_emb = self.embedder.embed_query(query)

        results = self.index.score(q_emb)

        top_results = results[:top_k]

        formatted = []
        for score, meta in top_results:
            formatted.append({
                "score": float(score),
                "doc_id": meta["doc_id"],
                "page_num": meta["page_num"],
                "image_path": meta["image_path"],
                "citation": meta["citation"],
            })

        return formatted