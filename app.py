import os
from pathlib import Path

import streamlit as st
from openai import OpenAI

from scripts.retrieve_tfidf import search

# index builder'ı subprocess yerine import ederek çağırıyoruz
from scripts.build_index_tfidf import build_index


MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # düşük maliyet
client = OpenAI()

INDEX_DIR = Path("index")
NEEDED = ["tfidf_vectorizer.pkl", "tfidf_matrix.pkl", "metadata.pkl"]


def ensure_index():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    missing = [f for f in NEEDED if not (INDEX_DIR / f).exists()]
    if missing:
        build_index()


def build_context(results, max_chars=6000):
    parts = []
    total = 0
    for r in results:
        header = f"[KAYNAK: {r['source']} | sayfa {r['page']} | skor {r['score']:.4f}]\n"
        body = (r.get("text") or "").strip()
        chunk = header + body + "\n\n"
        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total += len(chunk)
    return "".join(parts).strip()


def ask_gpt(question: str, results):
    context = build_context(results)

    system = (
        "Sen İstanbul Üniversitesi İşletme Fakültesi için doküman tabanlı bir asistan botsun. "
        "Sadece verilen bağlamdan cevap ver. Bağlamda yoksa 'Bu dokümanlarda net bir madde bulamadım.' de. "
        "Cevapta mümkünse madde numarası/sayfa belirt. Kısa ve net yaz."
    )

    user = f"SORU: {question}\n\nBAĞLAM:\n{context if context else '(boş)'}"

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
    if not st.session_state.get("debug"):
        return
    st.sidebar.write("Top sonuçlar:")
    for r in results[:5]:
        st.sidebar.write(f"{r['score']:.4f} | {r['source']} | s.{r['page']}")
        if r.get("text"):
            st.sidebar.caption(r["text"][:300].replace("\n", " "))
        else:
            st.sidebar.warning("⚠️ text alanı boş/None")


def main():
    st.set_page_config(page_title="İÜ İşletme Bot", page_icon="🎓", layout="centered")
    st.title("🎓 İÜ İşletme Doküman Botu")
    st.caption("Dokümanlara göre cevap verir. Debug moduyla retrieval’ı kontrol edebilirsin.")

    st.session_state["debug"] = st.sidebar.checkbox("Debug modu", value=False)
    st.sidebar.write(f"Model: `{MODEL}`")

    # index garanti
    try:
        ensure_index()
    except Exception as e:
        st.error(f"Index oluşturulamadı: {e}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # geçmişi bas
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
    try:
        results = search(q, top_k=5)
        render_debug(results)

        # 🔑 EN ÖNEMLİ KISIM: skor eşiği
        best = max([r["score"] for r in results], default=0.0)

        if best < 0.05:
            answer = "Bu dokümanlarda sorunuza doğrudan karşılık gelen net bir madde bulamadım."
        else:
            answer = ask_gpt(q, results)

        # kaynakları footer olarak göster
        sources = []
        for r in results[:3]:
            sources.append(
                f"- {r['source']} (s.{r['page']}) skor={r['score']:.4f}"
            )

        footer = "\n\n**Kaynaklar (en yakın eşleşmeler):**\n" + "\n".join(sources)

        st.markdown(answer + footer)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer + footer}
        )

    except Exception as e:
        st.error(f"Hata: {e}")


if __name__ == "__main__":
    main()
