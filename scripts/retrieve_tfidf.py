# scripts/retrieve_tfidf.py
import pickle
from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity


ROOT_DIR = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT_DIR / "index"

VEC_PATH = INDEX_DIR / "tfidf_vectorizer.pkl"
MAT_PATH = INDEX_DIR / "tfidf_matrix.pkl"
META_PATH = INDEX_DIR / "metadata.pkl"


def _load():
    with open(VEC_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(MAT_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    return vectorizer, tfidf_matrix, metadata


def search(query: str, top_k: int = 5):
    vectorizer, tfidf_matrix, metadata = _load()

    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, tfidf_matrix)[0]

    top_idx = sims.argsort()[-top_k:][::-1]

    results = []
    for i in top_idx:
        m = metadata[i]
        results.append(
            {
                "score": float(sims[i]),
                "source": m.get("source"),
                "page": m.get("page"),
                "text": m.get("text", ""),  # <-- kritik fix
                "chunk_id": m.get("chunk_id"),
            }
        )
    return results
