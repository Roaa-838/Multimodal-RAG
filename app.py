import json
from pathlib import Path
from typing import List, Dict

import streamlit as st
import torch
from PIL import Image

from ingestion.page_extractor import extract_pages
from ingestion.manifest import create_manifest, load_manifest
from retrieval.embedder import ColPaliEmbedder
from retrieval.vector_store import MultiVectorIndex
from retrieval.generator import GeminiVLMGenerator

try:
    from colpali_engine.interpretability import plot_all_similarity_maps
    HAS_INTERPRET = True
except Exception:
    HAS_INTERPRET = False


# ── App config ────────────────────────────────────────────────
st.set_page_config(page_title="Multi-Modal RAG QA", layout="wide")
st.title("📄 Multi-Modal RAG (ColPali + Gemini)")

DATA_ROOT     = Path("data/streamlit")
EXTRACTED_DIR = DATA_ROOT / "extracted"
INDEX_DIR     = DATA_ROOT / "index"
MANIFEST_PATH = INDEX_DIR / "manifest_with_embeddings.json"


# ── Cached loaders (load ONCE per session) ────────────────────
@st.cache_resource(show_spinner="⏳ Loading ColPali model (first time only ~1 min)...")
def get_embedder():
    return ColPaliEmbedder()


@st.cache_resource(show_spinner="⏳ Loading Gemini...")
def get_generator(api_key: str):
    return GeminiVLMGenerator(api_key=api_key)


@st.cache_resource(show_spinner="⏳ Loading index from disk...")
def get_index(manifest_path: str):
    """
    Cached — keyed on manifest_path string.
    Call st.cache_resource.clear() after rebuilding to force a reload.
    """
    manifest = load_manifest(manifest_path)
    _embedder = get_embedder()          # reuses cached embedder
    idx = MultiVectorIndex()
    for page in manifest["pages"]:
        emb = torch.load(page["embedding_path"], weights_only=True)
        idx.add(emb, page)
    return idx, manifest


# ── Helpers ───────────────────────────────────────────────────
def save_uploaded_pdf(uploaded_file) -> Path:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    dst = DATA_ROOT / uploaded_file.name
    with open(dst, "wb") as f:
        f.write(uploaded_file.read())
    return dst


def embed_pages_with_progress(pages: List[Dict], progress_bar) -> List[Dict]:
    embedder   = get_embedder()
    batch_size = 4 if torch.cuda.is_available() else 1
    total      = len(pages)

    for i in range(0, total, batch_size):
        batch  = pages[i : i + batch_size]
        images = [Image.open(p["image_path"]).convert("RGB") for p in batch]
        embs   = embedder.embed_batch(images)

        for p, emb in zip(batch, embs):
            emb_path = INDEX_DIR / f"emb_{p['doc_id']}_{p['page_num']}.pt"
            emb_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(emb, emb_path)
            p["embedding_path"] = str(emb_path)

        progress_bar.progress(min((i + batch_size) / total, 1.0))

    return pages


def retrieve(query: str, embedder, index, top_k: int = 4):
    q_emb   = embedder.embed_query(query)
    results = index.score(q_emb)

    formatted = [
        {
            "score":          float(score),
            "doc_id":         meta["doc_id"],
            "page_num":       meta["page_num"],
            "image_path":     meta["image_path"],
            "citation":       meta["citation"],
            "embedding_path": meta["embedding_path"],
        }
        for score, meta in results[:top_k]
    ]
    return formatted, q_emb


def format_citations(retrieved):
    lines = "\n".join(
        f"- {p['citation']} (Page {p['page_num']})" for p in retrieved
    )
    return f"\n\n**Sources:**\n{lines}"


# ─────────────────────────────────────────────────────────────
# SIDEBAR — Upload & Index
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📥 Upload & Index")

    api_key      = st.text_input("Gemini API Key", type="password")
    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

    if st.button("Build Index"):
        if not uploaded_pdf:
            st.warning("Please upload a PDF first.")
        else:
            with st.spinner("Extracting pages..."):
                pdf_path = save_uploaded_pdf(uploaded_pdf)
                doc_id   = Path(pdf_path).stem

                pages = extract_pages(
                    pdf_path      = pdf_path,
                    doc_id        = doc_id,
                    output_dir    = EXTRACTED_DIR,
                    use_pdf2image = False,
                )
                st.success(f"Extracted {len(pages)} pages")

            st.write("Embedding pages (this takes a minute on CPU)...")
            progress_bar = st.progress(0.0)
            pages = embed_pages_with_progress(pages, progress_bar)

            with st.spinner("Saving manifest..."):
                manifest = create_manifest(
                    all_pages    = [pages],
                    doc_metadata = {
                        doc_id: {
                            "description": doc_id,
                            "filename":    pdf_path.name,
                        }
                    },
                    output_dir = INDEX_DIR,
                )
                with open(MANIFEST_PATH, "w") as f:
                    json.dump(manifest, f, indent=2)

            # ── Clear cached index so it reloads with the new manifest ──
            get_index.clear()
            st.success("✅ Index built and saved!")
            st.rerun()

    if MANIFEST_PATH.exists():
        st.success("Index ready ✔")
    else:
        st.info("Upload a PDF and click 'Build Index' to start.")


# ─────────────────────────────────────────────────────────────
# LOAD INDEX  (stop early if not built yet)
# ─────────────────────────────────────────────────────────────
if not MANIFEST_PATH.exists():
    st.info("👈 Upload a PDF and build the index to get started.")
    st.stop()

# These are cached — fast after the first call
index, manifest = get_index(str(MANIFEST_PATH))
embedder        = get_embedder()


# ─────────────────────────────────────────────────────────────
# CHAT INTERFACE
# ─────────────────────────────────────────────────────────────
st.header("💬 Ask Questions")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask a question about the document...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # ── Retrieval ─────────────────────────────────────────────
    with st.spinner("Searching document..."):
        retrieved, q_emb = retrieve(query, embedder, index)

    # ── Show retrieved pages in sidebar ───────────────────────
    with st.sidebar:
        st.subheader("📄 Retrieved Pages")
        for p in retrieved:
            st.image(
                p["image_path"],
                caption        = f"{p['doc_id']} — Page {p['page_num']}  (score: {p['score']:.3f})",
                use_column_width = True,
            )

    # ── Similarity maps (bonus) ───────────────────────────────
    if HAS_INTERPRET:
        st.subheader("🔥 Similarity Maps")
        for p in retrieved[:2]:
            doc_emb = torch.load(p["embedding_path"], weights_only=True)
            img     = Image.open(p["image_path"]).convert("RGB")
            fig     = plot_all_similarity_maps(
                query_embeddings    = q_emb,
                document_embeddings = doc_emb,
                image               = img,
            )
            st.pyplot(fig)

    # ── Generate answer ───────────────────────────────────────
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar.")
        st.stop()

    generator   = get_generator(api_key)
    image_paths = [p["image_path"] for p in retrieved]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = generator.answer(
                question        = query,
                image_paths     = image_paths,
                retrieved_pages = retrieved,
            )

        final_answer = answer + format_citations(retrieved)
        st.markdown(final_answer)

    st.session_state.messages.append({"role": "assistant", "content": final_answer})