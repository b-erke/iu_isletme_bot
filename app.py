import os
import pickle
import streamlit as st
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import subprocess

BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "index"

VEC_PATH = INDEX_DIR / "tfidf_vectorizer.pkl"
MAT_PATH = INDEX_DIR / "tfidf_matrix.pkl"
META_PATH = INDEX_DIR / "metadata.pkl"

def ensure_index():
    if not VEC_PATH.exists():
        print("🔧 Index bulunamadı, yeniden oluşturuluyor...")
        subprocess.run(
            ["python", "scripts/build_index_tfidf.py"],
            check=True
        )

ensure_index()

# Optional: OpenAI (kota biterse app yine TF-IDF ile çalışsın)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

load_dotenv()

APP_TITLE = "İÜ İşletme Bot (TF-IDF + GPT)"
INDEX_DIR = "index"

def load_index():
    with open(os.path.join(INDEX_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(INDEX_DIR, "tfidf_matrix.pkl"), "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open(os.path.join(INDEX_DIR, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    return vectorizer, tfidf_matrix, metadata

def tfidf_search(query, vectorizer, tfidf_matrix, metadata, top_k=5, min_sim=0.0):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, tfidf_matrix)[0]
    idx = sims.argsort()[::-1]

    results = []
    for i in idx[:top_k]:
        score = float(sims[i])
        if score < min_sim:
            continue
        m = metadata[i]
        results.append({
            "score": score,
            "source": m.get("source", ""),
            "page": m.get("page", None),
            "text": m.get("text", "")  # build_index_tfidf.py bunu doldurmalı
        })
    return results

def format_sources(results):
    lines = []
    for r in results:
        p = f"sayfa {r['page']}" if r["page"] is not None else "sayfa ?"
        lines.append(f"- **{r['source']}** | {p} | skor={r['score']:.3f}")
    return "\n".join(lines) if lines else "_Kaynak bulunamadı._"

def build_prompt(user_q, contexts):
    # contextleri kısa tut (bütçe!)
    ctx_blocks = []
    for c in contexts[:5]:
        txt = (c.get("text") or "").strip()
        if not txt:
            continue
        # aşırı uzamasın
        if len(txt) > 1200:
            txt = txt[:1200] + "..."
        ctx_blocks.append(f"[KAYNAK: {c['source']} | sayfa {c['page']}]\n{txt}")

    ctx = "\n\n".join(ctx_blocks) if ctx_blocks else "Yeterli kaynak metni yok."

    return f"""
Sen İstanbul Üniversitesi İşletme Fakültesi yönergeleri/yönetmelikleri hakkında soru cevaplayan bir asistansın.
Sadece verilen kaynak metnine dayan. Uydurma yapma.
Cevabı Türkçe yaz, kısa ve net yaz.
Cevabın sonunda 'Kaynaklar' başlığı altında hangi PDF/TXT ve sayfa kullanıldığını madde madde belirt.

KULLANILABİLİR KAYNAK METİNLER:
{ctx}

SORU:
{user_q}
""".strip()

def stream_openai_answer(prompt):
    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    max_out = int(os.getenv("MAX_OUTPUT_TOKENS", "280"))

    if not api_key or OpenAI is None:
        raise RuntimeError("OPENAI_API_KEY yok ya da openai paketi yok.")

    client = OpenAI(api_key=api_key)

    # Chat Completions streaming
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Kaynaklara dayalı, uydurmayan bir yardımcı ol."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=max_out,
        stream=True,
    )

    for event in stream:
        delta = event.choices[0].delta.content
        if delta:
            yield delta

# ---------------- UI ----------------
st.set_page_config(page_title=APP_TITLE, page_icon="🎓", layout="centered")
st.title("🎓 İÜ İşletme Bot")
st.caption("TF-IDF ile kaynak bulur; istersen GPT ile cevap üretir (kota biterse sadece TF-IDF çalışır).")

# Sidebar controls
with st.sidebar:
    st.header("Ayarlar")
    use_gpt = st.toggle("GPT ile cevap üret", value=True)
    top_k = st.slider("Top-K kaynak", 1, 10, int(os.getenv("TOP_K", "5")))
    min_sim = st.slider("Min benzerlik eşiği", 0.0, 0.30, float(os.getenv("MIN_SIM", "0.08")))
    st.markdown("---")
    st.write("**Not:** API key’i repoya koyma. Streamlit Cloud’da Secrets kullan.")

# Load index once
if "index_loaded" not in st.session_state:
    try:
        st.session_state.vectorizer, st.session_state.tfidf_matrix, st.session_state.metadata = load_index()
        st.session_state.index_loaded = True
    except Exception as e:
        st.session_state.index_loaded = False
        st.error(f"Index bulunamadı. Önce `python scripts/build_index_tfidf.py` çalıştır. Hata: {e}")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_q = st.chat_input("Sorunu yaz (örn: Mazeret sınavı hangi maddeye göre yapılır?)")

if user_q and st.session_state.index_loaded:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    # Retrieve
    results = tfidf_search(
        user_q,
        st.session_state.vectorizer,
        st.session_state.tfidf_matrix,
        st.session_state.metadata,
        top_k=top_k,
        min_sim=min_sim,
    )

    # Always show sources
    with st.chat_message("assistant"):
        st.markdown("### 📌 Bulunan Kaynaklar")
        st.markdown(format_sources(results))

    # GPT answer (optional)
    if use_gpt:
        prompt = build_prompt(user_q, results)

        with st.chat_message("assistant"):
            st.markdown("### 🤖 Cevap")
            placeholder = st.empty()
            acc = ""

            try:
                for token in stream_openai_answer(prompt):
                    acc += token
                    placeholder.markdown(acc)
            except Exception as e:
                # Kota bitti / key yok / vs.
                st.warning(f"GPT çağrısı başarısız (muhtemelen kota/anahtar). TF-IDF kaynakları üstte. Hata: {e}")

            if acc.strip():
                st.session_state.messages.append({"role": "assistant", "content": acc})
    else:
        with st.chat_message("assistant"):
            st.info("GPT kapalı. Yalnızca kaynaklar gösterildi.")
