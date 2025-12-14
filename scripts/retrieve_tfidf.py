import pickle
from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity


INDEX_DIR = Path("index")


def load_index():
    with open(INDEX_DIR / "tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open(INDEX_DIR / "tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)

    with open(INDEX_DIR / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    return vectorizer, tfidf_matrix, metadata


def search(query: str, top_k: int = 5):
    vectorizer, tfidf_matrix, metadata = load_index()
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, tfidf_matrix)[0]
    top_idx = sims.argsort()[-top_k:][::-1]

    results = []
    for i in top_idx:
        m = metadata[int(i)]
        results.append(
            {
                "score": float(sims[int(i)]),
                "source": m["source"],
                "page": m["page"],
                "text": m["text"],
            }
        )
    return results


def main():
    print("Soru (çıkmak için boş): ", end="", flush=True)
    while True:
        q = input().strip()
        if not q:
            break
        results = search(q, top_k=5)
        for r in results:
            print("\n---")
            print(f'{r["source"]} | sayfa {r["page"]} | score={r["score"]:.3f}')
            print(r["text"][:800] + ("..." if len(r["text"]) > 800 else ""))
        print("\nSoru (çıkmak için boş): ", end="", flush=True)


if __name__ == "__main__":
    main()