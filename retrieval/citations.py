"""
retrieval/citations.py
Citation formatting and page deduplication utilities.

Used by:
  - run_qa.py     (CLI interface)
  - app.py        (format_citations — Markdown version is inline there)
"""

from __future__ import annotations


# ── Formatting ────────────────────────────────────────────────────────────────

def format_citations_plain(retrieved_pages: list[dict]) -> str:
    """
    Build a plain-text citation block from retrieved page records.

    Example output:
        Sources:
          [1] IMF Country Report — Egypt 2024 Article IV, Page 14  (score: 0.847)
          [2] IMF Country Report — Egypt 2024 Article IV, Page 15  (score: 0.831)

    Args:
        retrieved_pages: List of page dicts (must have 'citation', 'page_num', 'score')

    Returns:
        Formatted string
    """
    if not retrieved_pages:
        return "Sources: (none)"

    lines = ["Sources:"]
    for i, page in enumerate(retrieved_pages, start=1):
        citation = page.get("citation", f"Page {page.get('page_num', '?')}")
        score    = page.get("score", 0.0)
        lines.append(f"  [{i}] {citation}  (score: {score:.3f})")

    return "\n".join(lines)


def format_citations_markdown(retrieved_pages: list[dict]) -> str:
    """
    Build a Markdown citation block for Streamlit / report rendering.

    Example output:
        **Sources:**
        - IMF Country Report — Egypt 2024 Article IV, Page 14
        - IMF Country Report — France 2024 Article IV, Page 3

    Args:
        retrieved_pages: List of page dicts

    Returns:
        Markdown-formatted string (with leading newlines for spacing)
    """
    if not retrieved_pages:
        return "\n\n**Sources:** (none)"

    lines = ["\n\n**Sources:**"]
    for page in retrieved_pages:
        citation = page.get("citation", f"Page {page.get('page_num', '?')}")
        lines.append(f"- {citation}")

    return "\n".join(lines)


def format_citations_structured(retrieved_pages: list[dict]) -> list[dict]:
    """
    Return citations as a structured list for programmatic use
    (e.g., evaluation scripts, JSON output).

    Each item:
        {
          "rank":     1,
          "doc_id":   "egypt_2024_article_iv",
          "page_num": 14,
          "citation": "IMF Country Report — Egypt 2024 Article IV, Page 14",
          "score":    0.847
        }
    """
    return [
        {
            "rank":     i + 1,
            "doc_id":   page.get("doc_id", "unknown"),
            "page_num": page.get("page_num", 0),
            "citation": page.get("citation", ""),
            "score":    round(page.get("score", 0.0), 4),
        }
        for i, page in enumerate(retrieved_pages)
    ]


# ── Deduplication ─────────────────────────────────────────────────────────────

def deduplicate_pages(pages: list[dict]) -> list[dict]:
    """
    Remove duplicate page entries (same doc_id + page_num).
    Keeps the highest-scoring entry when duplicates exist.

    Duplicates can occur when multiple retrieval passes are combined
    (e.g., hybrid text + image search) or when top-k > corpus size.

    Args:
        pages: List of page dicts (each must have 'doc_id' and 'page_num')

    Returns:
        Deduplicated list, preserving original score-descending order.
    """
    seen: set[tuple[str, int]] = set()
    unique: list[dict] = []

    for page in pages:
        key = (page.get("doc_id", ""), page.get("page_num", -1))
        if key not in seen:
            seen.add(key)
            unique.append(page)

    return unique


def sort_by_score(pages: list[dict]) -> list[dict]:
    """Return pages sorted descending by 'score' field."""
    return sorted(pages, key=lambda p: p.get("score", 0.0), reverse=True)


def filter_by_threshold(pages: list[dict], min_score: float = 0.5) -> list[dict]:
    """
    Remove pages below a minimum similarity score.
    Useful for avoiding low-relevance context being sent to the LLM.

    Args:
        pages:     List of page dicts with 'score' key
        min_score: Minimum acceptable MaxSim score (0–1 range)

    Returns:
        Filtered list
    """
    filtered = [p for p in pages if p.get("score", 0.0) >= min_score]
    if not filtered:
        # Always return at least the top-1 to avoid empty context
        return pages[:1] if pages else []
    return filtered