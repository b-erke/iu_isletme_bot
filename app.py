import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from scripts.retrieve_tfidf import search


load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def get_api_key():
    # Streamlit secrets öncelikli, yoksa env
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    return os.getenv("OPENAI_API_KEY", "")


def ensure_index():
    if not Path("index/tfidf_vectorizer.pkl").exists():
        from scripts.build_index_tfidf import main as build_main

        build_main()


SYSTEM_PROMPT = """Sen İstanbul Üniversitesi İşletme Fakültesi için yönetmelik/SSS gibi metinlerden yanıt veren bir asistansın.
Kurallar:
- Yanıt dili Türkçe.
- Sadece verilen kaynak parçalarına dayan.
- Eğer kaynak parçalarında net cevap yoksa "Bu belgelerde net bilgi yok" de ve hangi belgenin hangi sayfasına baktığını belirt.
- En sonda kaynakları madde madde yaz: (dosya | sayfa).
"""


def format_context(hits):
    parts = []
    for h in hits:
        parts.append(f"[Kaynak: {h['source']} | sayfa {h['page']} | skor {h['score']:.3f}]")
        parts.append(h["text"])
        parts.append("")
    return "\n".join(parts).strip()


st.set_page_config(page_title="İÜ İşletme Bot", page_icon="🎓", layout="centered")
st.title("🎓 İÜ İşletme Bot")
st.caption("TF-IDF ile doküman bulur, GPT ile cevaplar. Kaynak gösterir.")

ensure_index()

api_key = get_api_key()
if not api_key:
    st.error("OPENAI_API_KEY bulunamadı. Lokal için .env, Streamlit Cloud için Secrets ekle.")
    st.stop()

client = OpenAI(api_key=api_key)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Merhaba. Sorunu yaz, ilgili belge parçalarını bulup kaynaklı cevap vereyim."}
    ]


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


q = st.chat_input("Soru yaz (ör: Mazeret sınavı hangi maddeye göre yapılır?)")
if q:
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    hits = search(q, top_k=5)
    context = format_context(hits)

    user_prompt = f"""Soru: {q}

Aşağıdaki kaynak parçalarını kullanarak cevap ver:
{context}
"""

    with st.chat_message("assistant"):
        out = st.empty()
        full = ""

        try:
            stream = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=350,
                stream=True,
            )

            for evt in stream:
                delta = evt.choices[0].delta.content or ""
                if delta:
                    full += delta
                    out.markdown(full)

        except Exception as e:
            # Quota / 429 vs. her şey için: fallback
            full = "❌ GPT çağrısı başarısız oldu. Şimdilik sadece doküman parçalarını gösterebiliyorum.\n\n"
            full += f"Hata: `{str(e)}`\n\n"
            full += "### Bulunan parçalar\n"
            for h in hits:
                full += f"- **{h['source']}**, sayfa **{h['page']}** (skor {h['score']:.3f})\n"
            out.markdown(full)

    st.session_state.messages.append({"role": "assistant", "content": full})