"""
retrieval/embedder.py
ColQwen2 multi-vector embedder — loads from local HuggingFace cache
without any network calls.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Set offline env vars BEFORE any transformers import ──────────
os.environ["TRANSFORMERS_OFFLINE"]  = "1"
os.environ["HF_DATASETS_OFFLINE"]   = "1"

import torch
from PIL import Image


# ── Resolve local model path ──────────────────────────────────────
def _find_local_snapshot(hf_home: str, repo: str) -> str:
    """
    Finds the local snapshot directory for a HuggingFace repo.
    Returns the snapshot path if found, otherwise the repo name.
    """
    cache_name = "models--" + repo.replace("/", "--")

    for base in [hf_home, r"C:\Users\roaar\.cache\huggingface",
                 os.path.expanduser("~/.cache/huggingface")]:
        snapshots_dir = Path(base) / "hub" / cache_name / "snapshots"
        if snapshots_dir.exists():
            hashes = [d for d in snapshots_dir.iterdir() if d.is_dir()]
            if hashes:
                snap = str(hashes[0])
                print(f"[embedder] Found local snapshot: {snap}")
                return snap

    print(f"[embedder] WARNING: No local snapshot found for {repo}, will try network")
    return repo


HF_HOME    = os.getenv("HF_HOME", r"C:\Users\roaar\.cache\huggingface")
MODEL_REPO = "vidore/colqwen2-v1.0"
MODEL_PATH = _find_local_snapshot(HF_HOME, MODEL_REPO)


class ColPaliEmbedder:
    """
    Wraps ColQwen2 for batch image and query embedding.
    Loads entirely from local disk cache — no network required.
    """

    def __init__(self, model_path: str = MODEL_PATH):
        from colpali_engine.models import ColQwen2, ColQwen2Processor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        print(f"[embedder] Loading from: {model_path}")
        print(f"[embedder] Device: {self.device}")

        self.model = ColQwen2.from_pretrained(
            model_path,
            torch_dtype      = dtype,
            device_map       = self.device,
            local_files_only = True,
        ).eval()

        self.processor = ColQwen2Processor.from_pretrained(
            model_path,
            local_files_only = True,
        )

        print("[embedder] Ready ✔")

    def embed_batch(self, images: list) -> list:
        """List of PIL images -> list of [N_patches, 128] tensors on CPU."""
        if not images:
            return []
        batch = self.processor.process_images(images).to(self.device)
        with torch.no_grad():
            embeddings = self.model(**batch)
        return [emb.cpu() for emb in embeddings]

    def embed_query(self, query: str):
        """Text query -> [N_tokens, 128] tensor on CPU."""
        batch = self.processor.process_queries([query]).to(self.device)
        with torch.no_grad():
            embeddings = self.model(**batch)
        return embeddings[0].cpu()

    def embed_image_file(self, image_path) -> object:
        img = Image.open(image_path).convert("RGB")
        return self.embed_batch([img])[0]

    @property
    def embedding_dim(self) -> int:
        return 128