"""
run_qa.py
Interactive CLI question-answering interface for the Multi-Modal RAG system.

Usage
-----
    # Interactive session
    python run_qa.py --manifest data/index/manifest_with_embeddings.json --api-key YOUR_KEY

    # Single question
    python run_qa.py --manifest data/index/manifest_with_embeddings.json --api-key YOUR_KEY \
                     --question "What is the main topic?"

    # Evaluation benchmark
    python run_qa.py --manifest data/index/manifest_with_embeddings.json --api-key YOUR_KEY \
                     --eval eval/benchmark_queries.json --out eval/results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

from retrieval.search    import load_searcher       # ← correct import
from retrieval.generator import GeminiVLMGenerator
from retrieval.citations import format_citations_plain


DEFAULT_MANIFEST = "data/index/manifest_with_embeddings.json"
DEFAULT_TOP_K    = 4


# ── Session ───────────────────────────────────────────────────

class QASession:
    def __init__(self, manifest_path: str | Path, api_key: str, top_k: int = DEFAULT_TOP_K):
        print("[qa] Loading index …", end=" ", flush=True)
        self.top_k     = top_k
        self.searcher  = load_searcher(manifest_path)
        self.generator = GeminiVLMGenerator(api_key=api_key)
        print("ready.\n")

    def ask(self, question: str) -> dict:
        t0 = time.time()

        retrieved = self.searcher.search(question, top_k=self.top_k)

        image_paths = [p["image_path"] for p in retrieved]
        answer = self.generator.answer(
            question        = question,
            image_paths     = image_paths,
            retrieved_pages = retrieved,
        )

        citations = format_citations_plain(retrieved)
        latency   = time.time() - t0

        return {
            "question" : question,
            "answer"   : answer,
            "retrieved": retrieved,
            "citations": citations,
            "latency_s": round(latency, 2),
        }


# ── Evaluation ────────────────────────────────────────────────

def run_eval(session: QASession, benchmark_path: str | Path, out_path: str | Path) -> None:
    benchmark_path = Path(benchmark_path)
    out_path       = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(benchmark_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    print(f"\n[eval] Running {len(queries)} benchmark queries …\n")
    results = []

    for i, item in enumerate(queries, start=1):
        question = item["question"]
        print(f"  [{i:02d}/{len(queries)}] {question[:70]}")
        result = session.ask(question)
        result["category"] = item.get("category", "general")
        results.append(result)
        print(f"         → {result['latency_s']}s")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    avg = sum(r["latency_s"] for r in results) / len(results)
    print(f"\n[eval] ✓  {len(results)} answers → {out_path}  (avg {avg:.2f}s)")


# ── Interactive loop ──────────────────────────────────────────

def interactive_loop(session: QASession) -> None:
    print("Multi-Modal RAG — Interactive QA")
    print("Type a question and press Enter.  Ctrl-C or 'exit' to quit.\n")
    print("-" * 60)

    while True:
        try:
            question = input("\n❓ Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[exit]")
            break

        if not question or question.lower() in {"exit", "quit", "q"}:
            print("[exit]")
            break

        result = session.ask(question)
        print(f"\n💬 Answer:\n{result['answer']}")
        print(f"\n{result['citations']}")
        print(f"\n⏱  {result['latency_s']}s  |  retrieved {len(result['retrieved'])} pages")
        print("-" * 60)


# ── CLI ───────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-Modal RAG — CLI QA Interface")
    parser.add_argument("--manifest",   default=DEFAULT_MANIFEST)
    parser.add_argument("--api-key",    default=os.getenv("GEMINI_API_KEY", ""), dest="api_key")
    parser.add_argument("--top-k",      type=int, default=DEFAULT_TOP_K, dest="top_k")
    parser.add_argument("--question",   default=None)
    parser.add_argument("--eval",       default=None, metavar="BENCHMARK_JSON")
    parser.add_argument("--out",        default="eval/results.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.api_key:
        print("[error] Gemini API key required. Pass --api-key or set GEMINI_API_KEY env var.")
        return 1

    try:
        session = QASession(manifest_path=args.manifest, api_key=args.api_key, top_k=args.top_k)
    except FileNotFoundError as exc:
        print(f"[error] {exc}")
        return 1

    if args.question:
        result = session.ask(args.question)
        print(f"\n💬 Answer:\n{result['answer']}")
        print(f"\n{result['citations']}")
        print(f"\n⏱  {result['latency_s']}s")
        return 0

    if args.eval:
        run_eval(session, args.eval, args.out)
        return 0

    interactive_loop(session)
    return 0


if __name__ == "__main__":
    sys.exit(main())