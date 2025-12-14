import os
import pickle
from pathlib import Path

import streamlit as st
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

INDEX_DIR = Path("index")

MODEL = "gpt-4o-mini"
TOP_K = 5
MAX_TOKENS = 220

def build_index_if_missing():
    needed = [
        INDEX_DIR / "tfidf_vectorizer.pkl",
        INDEX_DIR / "tfidf_matrix.pkl",
        INDEX_DIR / "metadata.pkl",
    ]
    if all(p.exists() for p in needed):
        return

    # build fonksiyonunu direkt import edip çalıştır
    from scripts.build_index_tfidf import main as build_main
    build_main()

@st.cache_resource
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
        sources.append((m["source"], m["page"]))
    return chunks, sources

def get_api_key():
    # Streamlit Cloud -> Secrets
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    # local -> env
    return os.getenv("OPENAI_API_KEY", "")

def ask_gpt(query, context_chunks, api_key):
    client = OpenAI(api_key=api_key)
    context = "\n\n".join(context_chunks[:TOP_K])

    prompt = f"""Sen İstanbul Üniversitesi İşletme Fakültesi mevzuat/SSS dokümanlarına göre cevap veren bir asistansın.
Cevabı Türkçe yaz. Eğer bağlamda cevap yoksa “Bu dokümanlarda net bir madde bulamadım” de.

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

st.set_page_config(page_title="İÜ İşletme Bot", page_icon="🎓", layout="centered")
st.title("🎓 İÜ İşletme Bot")
st.caption("TF-IDF retrieval + GPT (ucuz mod). Kaynak sayfa bilgisiyle cevaplar.")

# index garanti
try:
    build_index_if_missing()
except Exception as e:
    st.error(f"Index build edilemedi: {e}")
    st.stop()

vectorizer, tfidf_matrix, metadata = load_index()

api_key = get_api_key()
gpt_enabled = bool(api_key)

with st.sidebar:
    st.subheader("Ayarlar")
    st.write(f"GPT aktif: {'✅' if gpt_enabled else '❌'}")
    st.write("GPT kapalıysa sadece ilgili parçaları gösteririm.")
    st.write(f"Model: {MODEL}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

q = st.chat_input("Sorunu yaz…")
if q:
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    chunks, sources = retrieve(q, vectorizer, tfidf_matrix, metadata)

    with st.chat_message("assistant"):
        if gpt_enabled:
            try:
                ans = ask_gpt(q, chunks, api_key)
                st.markdown(ans)
            except Exception as e:
                st.warning(f"GPT çağrısı başarısız: {e}")
                st.markdown("İlgili doküman parçalarını aşağıya bırakıyorum:")

        st.markdown("**Kaynak parçalar:**")
        for (src, pg), ch in zip(sources, chunks):
            with st.expander(f"{src} | sayfa {pg}"):
                st.write(ch)

    # assistant mesajı olarak kaydet (gpt yoksa retrieval özetini kaydet)
    st.session_state.messages.append({"role": "assistant", "content": "Cevap üretildi (aşağıda kaynaklar var)."})
