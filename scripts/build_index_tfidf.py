# scripts/build_index_tfidf.py
import pickle
from pathlib import Path

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUT_DIR = ROOT_DIR / "index"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    text = " ".join((text or "").split())
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def read_pdf(path: Path):
    reader = PdfReader(str(path))
    out = []
    for i, page in enumerate(reader.pages):
        t = (page.extract_text() or "").strip()
        if t:
            out.append((t, i + 1))
    return out


def read_txt(path: Path):
    t = path.read_text(encoding="utf-8", errors="ignore").strip()
    return [(t, 1)] if t else []


def iter_documents():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"'{DATA_DIR}' yok. data/ içine pdf veya txt koy.")

    for p in DATA_DIR.rglob("*"):
        if p.is_dir() or p.name.startswith("."):
            continue

        ext = p.suffix.lower()
        if ext == ".pdf":
            for page_text, page_no in read_pdf(p):
                yield {"source": str(p.relative_to(DATA_DIR)), "page": page_no, "text": page_text}
        elif ext in [".txt", ".md"]:
            for page_text, page_no in read_txt(p):
                yield {"source": str(p.relative_to(DATA_DIR)), "page": page_no, "text": page_text}


def build_index():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    docs = list(iter_documents())
    chunks = []
    corpus = []

    for d in docs:
        parts = chunk_text(d["text"])
        for idx, part in enumerate(parts):
            chunks.append(
                {
                    "source": d["source"],
                    "page": d["page"],
                    "chunk_id": idx,
                    "text": part,
                }
            )
            corpus.append(part)

    if not corpus:
        raise RuntimeError("Hiç içerik bulunamadı. data/ içine pdf/txt koyduğuna emin ol.")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=50000,
        ngram_range=(1, 2),
        stop_words=None,
    )
    X = vectorizer.fit_transform(corpus)

    with open(OUT_DIR / "tfidf_matrix.pkl", "wb") as f:
        pickle.dump(X, f)
    with open(OUT_DIR / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(OUT_DIR / "metadata.pkl", "wb") as f:
        pickle.dump(chunks, f)

    return len(chunks)


if __name__ == "__main__":
    n = build_index()
    print(f"✅ TF-IDF index oluşturuldu. Toplam chunk: {n}")
