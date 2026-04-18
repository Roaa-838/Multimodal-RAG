"""
retrieval/generator.py
Gemini VLM answer generator using the current google-genai SDK.
"""

import os
import base64
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class GeminiVLMGenerator:
    def __init__(self, api_key: str | None = None):
        from google import genai
        from google.genai import types

        self._types = types
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY not set. Pass api_key= or add it to .env")

        self.client = genai.Client(api_key=key)
        self.model  = "gemini-flash-latest"
        print(f"[generator] Gemini ready → {self.model}")

    def answer(
        self,
        question: str,
        image_paths: list[str],
        retrieved_pages: list[dict],
    ) -> str:
        """
        Send question + retrieved page images to Gemini and return the answer.
        """

        system_prompt = (
            "You are a strict document question-answering assistant.\n"
            "Rules:\n"
            "- Answer ONLY from the provided document pages.\n"
            "- Do NOT use external knowledge.\n"
            "- If the answer is not visible in the pages, say: "
            "'Not found in the provided pages.'\n"
            "- Always cite the page number when referencing information.\n"
            "- Be precise and factual."
        )

        # Build content parts: system prompt + question + images
        parts = [system_prompt, f"Question: {question}"]

        for img_path in image_paths:
            path = Path(img_path)
            if not path.exists():
                continue
            image_bytes = path.read_bytes()
            mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            parts.append(
                self._types.Part.from_bytes(data=image_bytes, mime_type=mime)
            )

        response = self.client.models.generate_content(
            model    = self.model,
            contents = parts,
        )

        return response.text