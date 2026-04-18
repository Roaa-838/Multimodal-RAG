"""
retrieval/embedder.py
ColQwen2 multi-vector embedder for page images and text queries.
"""
# Add at the very top of embedder.py, BEFORE any other imports
import os
from dotenv import load_dotenv
load_dotenv()
os.environ.setdefault("HF_HOME", os.getenv("HF_HOME", r"D:\hf_cache"))

from pathlib import Path

import torch
from PIL import Image


MODEL_NAME = "vidore/colqwen2-v1.0"


class ColPaliEmbedder:
    """
    Wraps ColQwen2 (colpali-engine) for batch image and query embedding.
    Uses HF_TOKEN from .env for authenticated model download.
    """

    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        from colpali_engine.models import ColQwen2, ColQwen2Processor

        # handles both HF_TOKEN and HF_token casings
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HF_token")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        print(f"[embedder] Loading {model_name} on {self.device} ...")
        if hf_token:
            print(f"[embedder] HF token found ✔ (authenticated download)")
        else:
            print(f"[embedder] ⚠ No HF token — add HF_token=hf_xxx to your .env file")

        self.model = ColQwen2.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=self.device,
            token=hf_token,
        ).eval()

        self.processor = ColQwen2Processor.from_pretrained(
            model_name,
            token=hf_token,
        )

        print("[embedder] Ready ✔")

    def embed_batch(self, images: list) -> list:
        """List of PIL images → list of [N_patches, 128] tensors on CPU."""
        if not images:
            return []
        batch = self.processor.process_images(images).to(self.device)
        with torch.no_grad():
            embeddings = self.model(**batch)
        return [emb.cpu() for emb in embeddings]

    def embed_query(self, query: str):
        """Text query → [N_tokens, 128] tensor on CPU."""
        batch = self.processor.process_queries([query]).to(self.device)
        with torch.no_grad():
            embeddings = self.model(**batch)
        return embeddings[0].cpu()

    def embed_image_file(self, image_path) -> object:
        """Convenience: load image from disk and embed."""
        img = Image.open(image_path).convert("RGB")
        return self.embed_batch([img])[0]

    @property
    def embedding_dim(self) -> int:
        return 128