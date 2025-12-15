import pickle
from sklearn.metrics.pairwise import cosine_similarity


def load_index():
    with open("index/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open("index/tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)

    with open("index/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    return vectorizer, tfidf_matrix, metadata


def search(query: str, top_k: int = 5):
    vectorizer, tfidf_matrix, metadata = load_index()

    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, tfidf_matrix)[0]
    top_idx = sims.argsort()[-top_k:][::-1]

    results = []
    for i in top_idx:
        results.append(
            {
                "score": float(sims[i]),
                "source": metadata[i]["source"],
                "page": metadata[i]["page"],
                "chunk_id": metadata[i].get("chunk_id"),
                "text": metadata[i]["text"],  # ✅ EN KRİTİK: None değil, gerçek metin
            }
        )
    return results


def main():
    print("\nSoru (çıkmak için boş): ", end="")
    while True:
        q = input().strip()
        if not q:
            break
        res = search(q, top_k=5)
        print("\n---")
        for r in res:
            print(f"{r['source']} | sayfa {r['page']} | score={r['score']:.4f}")
            print(r["text"][:600])
            print("---")
        print("\nSoru (çıkmak için boş): ", end="")


if __name__ == "__main__":
    main()
