import os
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

INDEX_DIR = Path("index")

MODEL = "gpt-4o-mini"   # ucuz
MAX_TOKENS = 220        # kısa tut
TOP_K = 5

def load_index():
    with open(INDEX_DIR / "tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(INDEX_DIR / "tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open(INDEX_DIR / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return vectorizer, tfidf_matrix, metadata

def retrieve(query, vectorizer, tfidf_matrix, metadata, top_k=TOP_K):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, tfidf_matrix)[0]
    top_idx = sims.argsort()[-top_k:][::-1]

    chunks = []
    sources = []
    for i in top_idx:
        m = metadata[i]
        chunks.append(f"[{m['source']} | sayfa {m['page']}] {m.get('text','')}")
        sources.append({"source": m["source"], "page": m["page"]})
    return chunks, sources

def answer_with_gpt(query, context_chunks):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY ortam değişkeni yok.")

    client = OpenAI(api_key=api_key)

    context = "\n\n".join(context_chunks[:TOP_K])

    prompt = f"""Sen İstanbul Üniversitesi İşletme Fakültesi mevzuat/SSS dokümanlarına göre cevap veren bir asistansın.
Cevabı Türkçe yaz. Eğer bağlamda cevap yoksa “Bu dokümanlarda net bir madde bulamadım” de ve en yakın ilgili parçayı söyle.

Soru: {query}

Bağlam:
{context}
"""

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Kısa, net, kaynak referanslı cevap ver."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

def main():
    vectorizer, tfidf_matrix, metadata = load_index()

    while True:
        q = input("\nSoru (çıkmak için boş): ").strip()
        if not q:
            break

        chunks, sources = retrieve(q, vectorizer, tfidf_matrix, metadata)
        ans = answer_with_gpt(q, chunks)

        print("\n=== CEVAP ===")
        print(ans)
        print("\n=== KAYNAKLAR ===")
        for s in sources:
            print(f"- {s['source']} | sayfa {s['page']}")

if __name__ == "__main__":
    main()
