# DSAI 413 — Multi-Modal Document Intelligence (RAG-Based QA System)

A complete **Retrieval-Augmented Generation (RAG)** pipeline that ingests
complex PDF documents (text, tables, charts, figures) and answers questions
over them using **ColPali (ColQwen2)** visual embeddings + **Gemini 2.0 Flash**
as the answer generator.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          INGESTION PIPELINE                         │
│   PDF → PyMuPDF → Page Images (PNG, 300 DPI) + Text → manifest.json│
└─────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│                          INDEXING PIPELINE                          │
│   Page Images → ColQwen2 → Multi-Vector .pt files                  │
│                          → manifest_with_embeddings.json            │
└─────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│                          RETRIEVAL + QA                             │
│   Query → ColQwen2 → MaxSim → Top-K pages → Gemini 2.0 Flash       │
│                                           → Answer + Citations      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
assignment1/
├── app.py                      # Streamlit chat UI
├── run_retrieval.py            # CLI: build ColPali index
├── run_qa.py                   # CLI: interactive / eval QA
├── trim_manifest.py            # Utility: reduce manifest to N pages
├── list_models.py              # Utility: list available Gemini models
├── requirements.txt
├── .env                        # API keys (not committed)
│
├── ingestion/
│   ├── pdf_loader.py           # Scan folder or load explicit PDFs
│   ├── page_extractor.py       # PDF → PNG images + text per page
│   ├── manifest.py             # Build / load manifest.json
│   └── run_ingestion.py        # Master ingestion entry-point
│
├── retrieval/
│   ├── embedder.py             # ColQwen2 image & query embeddings
│   ├── vector_store.py         # In-memory MaxSim index
│   ├── build_index.py          # Build & save .pt embeddings
│   ├── indexer.py              # Indexer module (used by run_retrieval)
│   ├── search.py               # High-level Searcher class
│   ├── query_engine.py         # Query engine wrapper
│   ├── context_builder.py      # Load/resize images for VLM
│   ├── generator.py            # Gemini 2.0 Flash VLM integration
│   └── citations.py            # Citation formatting utilities
│
├── data/
│   ├── documents/              # Raw PDFs (not committed)
│   ├── extracted/              # Page images + manifest.json
│   └── index/                  # .pt embeddings + manifest_with_embeddings.json
│
└── eval/
    ├── benchmark_queries.json
    └── results.json
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API keys

Create a `.env` file:
```dotenv
GEMINI_API_KEY=your_gemini_key_here
HF_token=your_huggingface_token_here
HF_HOME=D:\hf_cache
```
- Gemini key: https://aistudio.google.com/app/apikey
- HuggingFace token: https://huggingface.co/settings/tokens

---

## Quick Start

### Step 1 — Ingest PDFs
```bash
python -m ingestion.run_ingestion --pdf-dir data/documents --output data/extracted
```

### Step 2 — Build index (trim to 50 pages on CPU)
```bash
python trim_manifest.py
python run_retrieval.py --manifest data/extracted/manifest_small.json --output data/index
```

### Step 3a — Streamlit app
```bash
streamlit run app.py
```

### Step 3b — CLI single question
```bash
python run_qa.py --manifest data/index/manifest_with_embeddings.json --question "What is the GDP growth forecast?"
```

### Step 3c — Evaluation benchmark
```bash
python run_qa.py --manifest data/index/manifest_with_embeddings.json --eval eval/benchmark_queries.json --out eval/results.json
```

---

## Key Design Decisions

| Component | Choice | Rationale |
|---|---|---|
| Page representation | Full-page PNG (300 DPI) | Preserves tables, charts and layout |
| Embedding model | ColQwen2 (ColPali) | Late-interaction multi-vector document retrieval |
| Similarity scoring | MaxSim (token-level) | Matches ColPali training objective |
| Answer generator | Gemini 2.0 Flash | Native multi-image input with citations |
| Index storage | In-memory + `.pt` files | No external dependencies; skips already-embedded pages |

---

## Windows Notes

- Set `HF_HOME=D:\hf_cache` in `.env` to avoid filling C: drive (~9GB model)
- ColQwen2 UNEXPECTED/MISSING key warnings are safe to ignore
- Use `%GEMINI_API_KEY%` (not `$`) in Windows CMD

---

## Authors

DSAI 413 — Multi-Modal Deep Learning
Zewail City of Science and Technology · Spring 2025
