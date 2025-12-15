# app.py
import os
from pathlib import Path

import streamlit as st
from openai import OpenAI

from scripts.retrieve_tfidf import search


MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
INDEX_DIR = Path(__file__).resolve().parent / "index"
NEEDED = ["tfidf_vectorizer.pkl", "tfidf_matrix.pkl", "metadata.pkl"]

client = OpenAI()


def ensure_index_exists():
    missing = [f for f in NEEDED if not (INDEX_DIR / f).exists()]
    if missing:
        st.error(
            "Index dosyaları bulunamadı.\n\n"
            f"Eksikler: {', '.join(missing)}\n\n"
            "Çözüm: index/ klasörünü repoya ekle (3 pkl), ya da localde build edip pushla."
        )
        st.stop()


def build_context(results, max_chars=6000):
    parts = []
    total = 0
    for r in results:
        body = (r.get("text") or "").strip()
        if not body:
            continue
        header = f"[KAYNAK: {r['source']} | sayfa {r['page']} | skor {r['score']:.4f}]\n"
        chunk = header + body + "\n\n"
        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total += len(chunk)
    return "".join(parts).strip()


def ask_gpt(question: str, results):
    context = build_context(results)

    # bağlam boşsa API'ye gitme (para yakma)
    if not context:
        return "Bu dokümanlarda net bir madde bulamadım."

    system = (
        "Sen İstanbul Üniversitesi İşletme Fakültesi için doküman tabanlı bir asistansın. "
        "Sadece verilen BAĞLAM'a dayanarak cevap ver. "
        "BAĞLAM'da yoksa kesinlikle uydurma ve şu cümleyi ver: 'Bu dokümanlarda net bir madde bulamadım.' "
        "Cevapta mümkünse madde numarası ve sayfa belirt. Kısa ve net yaz."
    )

    user = f"SORU: {question}\n\nBAĞLAM:\n{context}"

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()


def render_debug(results):
    st.sidebar.write("Top sonuçlar:")
    for r in results[:5]:
        st.sidebar.write(f"{r['score']:.4f} | {r['source']} | s.{r['page']}")
        t = (r.get("text") or "").strip()
        if t:
            st.sidebar.caption(t[:300].replace("\n", " "))
        else:
            st.sidebar.warning("⚠️ text alanı boş")


def main():
    st.set_page_config(page_title="İÜ İşletme Bot", page_icon="🎓", layout="centered")
    st.title("🎓 İÜ İşletme Doküman Botu")
    st.caption("Sorulara dokümanlardan cevap verir. (Bağlam yoksa uydurmaz.)")

    st.session_state["debug"] = st.sidebar.checkbox("Debug modu", value=False)
    st.sidebar.write(f"Model: `{MODEL}`")

    # Streamlit Cloud’da key: Settings > Secrets
    # OPENAI_API_KEY = "..."
    ensure_index_exists()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    q = st.chat_input("Sorunu yaz…")
    if not q:
        return

    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        results = search(q, top_k=5)

        if st.session_state.get("debug"):
            render_debug(results)

        # düşük skor varsa boş cevap dön (para yakma)
        best = results[0]["score"] if results else 0.0
        threshold = float(st.sidebar.slider("Min skor (API çağrısı)", 0.0, 0.4, 0.05, 0.01))

        if best < threshold:
            answer = "Bu dokümanlarda net bir madde bulamadım."
        else:
            answer = ask_gpt(q, results)

        sources = []
        for r in results[:3]:
            sources.append(f"- {r['source']} (s.{r['page']}) skor={r['score']:.4f}")
        footer = "\n\n**Kaynaklar (en yakın eşleşmeler):**\n" + "\n".join(sources) if sources else ""

        out = answer + footer
        st.markdown(out)
        st.session_state.messages.append({"role": "assistant", "content": out})


if __name__ == "__main__":
    main()
