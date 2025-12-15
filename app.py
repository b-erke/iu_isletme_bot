import os
from pathlib import Path

import streamlit as st
from openai import OpenAI

from scripts.retrieve_tfidf import search

# ⚠️ Indexler GİTHUB’DA VARSA build_index KULLANMIYORUZ
# from scripts.build_index_tfidf import build_index


# =====================
# AYARLAR
# =====================
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # ucuz model
client = OpenAI()

INDEX_DIR = Path("index")
NEEDED = ["tfidf_vectorizer.pkl", "tfidf_matrix.pkl", "metadata.pkl"]


# =====================
# INDEX KONTROL
# =====================
def ensure_index():
    missing = [f for f in NEEDED if not (INDEX_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Index eksik: {missing}. "
            "Index dosyaları repo içinde olmalı."
        )


# =====================
# CONTEXT OLUŞTURMA
# =====================
def build_context(results, max_chars=6000):
    parts = []
    total = 0

    for r in results:
        text = (r.get("text") or "").strip()
        if not text:
            continue

        header = (
            f"[KAYNAK: {r['source']} | sayfa {r['page']} | skor {r['score']:.4f}]\n"
        )
        chunk = header + text + "\n\n"

        if total + len(chunk) > max_chars:
            break

        parts.append(chunk)
        total += len(chunk)

    return "".join(parts).strip()


# =====================
# GPT SORU
# =====================
def ask_gpt(question: str, results):
    context = build_context(results)

    system = (
        "Sen İstanbul Üniversitesi İşletme Fakültesi için doküman tabanlı bir asistansın. "
        "SADECE verilen bağlamdan cevap ver. "
        "Bağlamda yoksa aynen şu cümleyi yaz: "
        "'Bu dokümanlarda sorunuza doğrudan karşılık gelen net bir madde bulamadım.' "
        "Varsa madde numarası ve sayfa belirt. Kısa ve net yaz."
    )

    user = f"SORU:\n{question}\n\nBAĞLAM:\n{context if context else '(boş)'}"

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
    )

    return resp.choices[0].message.content.strip()


# =====================
# DEBUG
# =====================
def render_debug(results):
    if not st.session_state.get("debug"):
        return

    st.sidebar.markdown("### 🔎 Retrieval Debug")
    for r in results:
        st.sidebar.write(
            f"{r['score']:.4f} | {r['source']} | s.{r['page']}"
        )
        if r.get("text"):
            st.sidebar.caption(r["text"][:300].replace("\n", " "))
        else:
            st.sidebar.warning("Text boş")


# =====================
# STREAMLIT APP
# =====================
def main():
    st.set_page_config(
        page_title="İÜ İşletme Bot",
        page_icon="🎓",
        layout="centered",
    )

    st.title("🎓 İÜ İşletme Doküman Botu")
    st.caption("Sadece resmi dokümanlara göre cevap verir.")

    st.session_state["debug"] = st.sidebar.checkbox("Debug modu", value=False)
    st.sidebar.write(f"Model: `{MODEL}`")

    # Index kontrol
    try:
        ensure_index()
    except Exception as e:
        st.error(str(e))
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Geçmişi göster
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    q = st.chat_input("Sorunu yaz…")
    if not q:
        return

    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    # =====================
    # ASISTAN
    # =====================
    with st.chat_message("assistant"):
        try:
            results = search(q, top_k=5)
            render_debug(results)

            best_score = max([r["score"] for r in results], default=0.0)

            if best_score < 0.05:
                answer = (
                    "Bu dokümanlarda sorunuza doğrudan karşılık gelen "
                    "net bir madde bulamadım."
                )
            else:
                answer = ask_gpt(q, results)

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
