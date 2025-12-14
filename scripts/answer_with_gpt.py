import os
import textwrap
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from scripts.retrieve_tfidf import search


load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI()


SYSTEM_PROMPT = """Sen İstanbul Üniversitesi İşletme Fakültesi için yönetmelik/SSS gibi metinlerden yanıt veren bir asistansın.
Kurallar:
- Yanıt dili Türkçe.
- Sadece verilen kaynak parçalarına dayan.
- Eğer kaynak parçalarında net cevap yoksa "Bu belgelerde net bilgi yok" de ve hangi belgenin hangi sayfasına baktığını belirt.
- En sonda kaynakları madde madde yaz: (dosya | sayfa).
"""


def build_context(hits):
    lines = []
    for h in hits:
        header = f"[Kaynak: {h['source']} | sayfa {h['page']} | skor {h['score']:.3f}]"
        lines.append(header)
        lines.append(h["text"])
        lines.append("")
    return "\n".join(lines).strip()


def answer(question: str, top_k: int = 5):
    hits = search(question, top_k=top_k)
    context = build_context(hits)

    user_prompt = f"""Soru: {question}

Aşağıdaki kaynak parçalarını kullanarak cevap ver:
{context}
"""

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=350,
    )

    text = resp.choices[0].message.content.strip()
    return text, hits


def main():
    # index yoksa oluştur
    if not Path("index/tfidf_vectorizer.pkl").exists():
        print("⚠️ index/ yok. Önce index oluşturuyorum...")
        from scripts.build_index_tfidf import main as build_main

        build_main()

    print("Soru (çıkmak için boş): ", end="", flush=True)
    while True:
        q = input().strip()
        if not q:
            break

        try:
            ans, hits = answer(q)
            print("\n" + ans + "\n")
        except Exception as e:
            print("\n❌ GPT çağrısı başarısız:", str(e))
            print("➡️ Sadece TF-IDF sonuçlarını gösteriyorum.\n")
            hits = search(q, top_k=5)

        print("Kaynak parçaları:")
        for h in hits:
            print(f"- {h['source']} | sayfa {h['page']} | skor={h['score']:.3f}")
        print("\nSoru (çıkmak için boş): ", end="", flush=True)


if __name__ == "__main__":
    main()