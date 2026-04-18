"""
page_extractor.py
Converts each page of a PDF into:
  - High-res image (300 DPI, ≥1024px width)
  - Raw extracted text
  - Structured page-level manifest

Supports:
  - pdf2image (best quality, optional)
  - PyMuPDF (fallback, default)
"""

import json
import shutil
from pathlib import Path

import fitz
from PIL import Image
from tqdm import tqdm


# ── Configuration ─────────────────────────────────────────────────────────────
DPI = 300
MIN_WIDTH = 1024
IMAGE_FORMAT = "PNG"
IMAGE_EXT = ".png"


# ── Main extraction pipeline ──────────────────────────────────────────────────
def extract_pages(pdf_path: Path, doc_id: str, output_dir: Path, use_pdf2image: bool = True) -> list[dict]:
    """
    Extract images + text per page and save to disk.
    """

    output_dir = Path(output_dir)
    img_dir = output_dir / doc_id / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[extract] {doc_id}")
    print(f"PDF  : {pdf_path}")
    print(f"OUT  : {img_dir}")

    # ── Choose backend ─────────────────────────────────────────────
    if use_pdf2image and _pdf2image_available():
        images = _rasterize_pdf2image(pdf_path)
    else:
        if use_pdf2image:
            print("[warn] pdf2image not available → using PyMuPDF")
        images = _rasterize_pymupdf(pdf_path)

    texts = _extract_text_pymupdf(pdf_path)

    # safety check
    n_pages = min(len(images), len(texts))

    pages = []

    for idx in tqdm(range(n_pages), desc=f"Processing {doc_id}"):
        page_num = idx + 1

        img = _ensure_min_width(images[idx], MIN_WIDTH)
        text = texts[idx].strip()

        img_filename = f"page_{page_num:04d}{IMAGE_EXT}"
        img_path = img_dir / img_filename

        img.save(img_path, format=IMAGE_FORMAT)

        pages.append({
            "doc_id": doc_id,
            "page_num": page_num,
            "image_path": str(img_path.resolve()),
            "text": text,
            "width": img.width,
            "height": img.height,
            "dpi": DPI,
        })

    # ── Save manifest (IMPORTANT FOR ASSIGNMENT) ────────────────
    manifest_path = output_dir / f"{doc_id}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, indent=2)

    print(f"[ok] Extracted {len(pages)} pages")
    print(f"[saved] manifest → {manifest_path}")

    return pages


# ── Backends ─────────────────────────────────────────────────────────────────
def _pdf2image_available() -> bool:
    return shutil.which("pdftoppm") is not None


def _rasterize_pdf2image(pdf_path: Path):
    from pdf2image import convert_from_path

    return convert_from_path(
        str(pdf_path),
        dpi=DPI,
        fmt="RGB",
        thread_count=2,
    )


def _rasterize_pymupdf(pdf_path: Path):
    doc = fitz.open(str(pdf_path))

    zoom = DPI / 72.0
    mat = fitz.Matrix(zoom, zoom)

    images = []

    for page in doc:
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    doc.close()
    return images


# ── Text extraction ───────────────────────────────────────────────────────────
def _extract_text_pymupdf(pdf_path: Path):
    doc = fitz.open(str(pdf_path))
    texts = [page.get_text("text") for page in doc]
    doc.close()
    return texts


# ── Utilities ────────────────────────────────────────────────────────────────
def _ensure_min_width(img: Image.Image, min_width: int) -> Image.Image:
    if img.width < min_width:
        scale = min_width / img.width
        new_size = (min_width, int(img.height * scale))

        return img.resize(new_size, Image.Resampling.LANCZOS)

    return img


def load_page_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


# ── Local test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from ingestion.pdf_loader import load_all_pdfs

    output_root = Path("data/extracted")

    pdfs = load_all_pdfs()

    for doc_id, pdf_path in pdfs.items():
        pages = extract_pages(pdf_path, doc_id, output_root)

        print(f"\nSample page from {doc_id}:")
        print(json.dumps(pages[0], indent=2))