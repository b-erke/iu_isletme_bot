import os
from openai import OpenAI

from scripts.retrieve_tfidf import search

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # düşük maliyet için
client = OpenAI()


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


def answer(question: str):
    results = search(question, top_k=5)
    context = build_context(results)

    system = (
        "Sen İstanbul Üniversitesi İşletme Fakültesi için doküman tabanlı bir asistan botsun. "
        "Sadece verilen bağlamdan cevap ver. Bağlamda yoksa 'Bu dokümanlarda net bir madde bulamadım.' de. "
        "Cevapta mümkünse madde numarası/sayfa belirt."
    )

    user = f"SORU: {question}\n\nBAĞLAM:\n{context if context else '(boş)'}"

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
        )
        text = resp.choices[0].message.content.strip()

    except Exception as e:
        # ✅ quota / rate limit vs. olursa GPT olmadan geri dön
        text = (
            "Şu an GPT çağrısı yapılamadı (kota/bağlantı hatası olabilir). "
            "Aşağıda dokümanlardan en yakın eşleşen parçaları veriyorum:\n\n"
        )
        for r in results[:3]:
            text += f"- {r['source']} (sayfa {r['page']}), skor={r['score']:.4f}\n  {r['text'][:400].replace(chr(10),' ')}\n\n"

    return text, results


def main():
    while True:
        q = input("\nSoru (çıkmak için boş): ").strip()
        if not q:
            break
        ans, sources = answer(q)
        print("\n" + ans)


if __name__ == "__main__":
    main()
