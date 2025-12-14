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

def search(query, vectorizer, tfidf_matrix, metadata, top_k=5):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, tfidf_matrix)[0]
    top_idx = sims.argsort()[-top_k:][::-1]

    results = []
    for i in top_idx:
        results.append({
            "score": float(sims[i]),
            "source": metadata[i]["source"],
            "page": metadata[i]["page"],
            "text": metadata[i].get("text", ""),
        })
    return results

def main():
    vectorizer, tfidf_matrix, metadata = load_index()

    while True:
        q = input("\nSoru (çıkmak için boş): ").strip()
        if not q:
            break

        results = search(q, vectorizer, tfidf_matrix, metadata, top_k=5)
        for r in results:
            print("\n---")
            print(f'{r["source"]} | sayfa {r["page"]} | skor={r["score"]:.4f}')
            print((r["text"][:500] + "…") if len(r["text"]) > 500 else r["text"])

if __name__ == "__main__":
    main()
