"""
retrieval/context_builder.py
Assembles the multi-modal context passed to the generator LLM.

Responsibilities:
  - Load PIL Images for retrieved pages
  - Combine page images + extracted text into a structured prompt
  - Token-budget management (trim text if too many pages retrieved)

This is the bridge between retrieval and generation.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image


# ── Constants ─────────────────────────────────────────────────────────────────
MAX_TEXT_CHARS_PER_PAGE = 1_500   # truncate page text beyond this for prompt
MAX_PAGES_IN_CONTEXT    = 5       # hard cap — beyond this, quality drops


# ── Image loading ─────────────────────────────────────────────────────────────

def load_page_images(retrieved_pages: list[dict]) -> list[Image.Image]:
    """
    Load PIL Images for every page in *retrieved_pages*.
    Missing images are replaced with a white placeholder to avoid crashes.

    Args:
        retrieved_pages: List of page dicts (must have 'image_path')

    Returns:
        List of RGB PIL Images, same length as retrieved_pages
    """
    images = []
    for page in retrieved_pages:
        img_path = Path(page.get("image_path", ""))
        if img_path.exists():
            images.append(Image.open(img_path).convert("RGB"))
        else:
            # Placeholder — keeps index alignment intact
            placeholder = Image.new("RGB", (512, 700), color=(240, 240, 240))
            images.append(placeholder)
    return images


# ── Text context ──────────────────────────────────────────────────────────────

def build_text_context(retrieved_pages: list[dict], max_chars: int = MAX_TEXT_CHARS_PER_PAGE) -> str:
    """
    Build a structured text context block from page text extracts.
    Used as supplementary context for text-grounded questions.

    Format:
        --- Page 14 | Egypt 2024 Article IV ---
        <extracted text, truncated to max_chars>
        ...

    Args:
        retrieved_pages: List of page dicts (with 'text', 'page_num', 'doc_id')
        max_chars:       Max characters per page text block

    Returns:
        Combined text string
    """
    if not retrieved_pages:
        return "(No pages retrieved)"

    blocks = []
    for page in retrieved_pages[:MAX_PAGES_IN_CONTEXT]:
        doc_label = page.get("doc_id", "unknown").replace("_", " ").title()
        page_num  = page.get("page_num", "?")
        text      = page.get("text", "").strip()

        if len(text) > max_chars:
            text = text[:max_chars] + "\n… [truncated]"

        header = f"--- Page {page_num} | {doc_label} ---"
        blocks.append(f"{header}\n{text}")

    return "\n\n".join(blocks)


# ── Full context package ──────────────────────────────────────────────────────

def build_context_package(
    question: str,
    retrieved_pages: list[dict],
    include_text: bool = True,
) -> dict:
    """
    Build a complete context package for the generator.

    Returns a dict with:
        images       — list of PIL Images (primary visual context for VLM)
        text_context — combined page text (supplementary)
        image_paths  — list of image file paths
        page_labels  — human-readable label per page
        question     — the original question
        n_pages      — number of context pages

    Args:
        question:        User's question string
        retrieved_pages: Retrieved page dicts from the vector store
        include_text:    Whether to include text context (True by default)
    """
    pages = retrieved_pages[:MAX_PAGES_IN_CONTEXT]

    images      = load_page_images(pages)
    image_paths = [p.get("image_path", "") for p in pages]
    page_labels = [
        f"Page {p.get('page_num', '?')} — "
        f"{p.get('doc_id', 'document').replace('_', ' ').title()}"
        for p in pages
    ]

    text_context = build_text_context(pages) if include_text else ""

    return {
        "question"    : question,
        "images"      : images,
        "image_paths" : image_paths,
        "text_context": text_context,
        "page_labels" : page_labels,
        "n_pages"     : len(pages),
    }


# ── Prompt assembly ───────────────────────────────────────────────────────────

def build_generation_prompt(question: str, page_labels: list[str]) -> str:
    """
    Build the text portion of the multi-modal generation prompt.
    Images are passed separately as PIL / bytes to the VLM.

    Args:
        question:    User question
        page_labels: List of human-readable page labels shown alongside images

    Returns:
        Prompt string
    """
    label_block = "\n".join(f"  Image {i+1}: {lbl}" for i, lbl in enumerate(page_labels))

    return (
        f"You are analyzing the following {len(page_labels)} document page(s):\n"
        f"{label_block}\n\n"
        f"Question: {question}\n\n"
        "Instructions:\n"
        "- Answer strictly from the provided document pages.\n"
        "- If you reference a table or chart, describe what it shows.\n"
        "- Cite the page number for every specific claim.\n"
        "- If the answer is not in the pages, say: 'Not found in retrieved pages.'"
    )