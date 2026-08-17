import pandas as pd

REVIEW_FILE = "manual_review_10_articles.csv"
TEXT_FILE = "real_trdizin_texts.csv"
OUTPUT_FILE = "manual_review_10_with_abstracts.csv"

review = pd.read_csv(REVIEW_FILE, encoding="utf-8-sig")
texts = pd.read_csv(TEXT_FILE, encoding="utf-8-sig")

print("real_trdizin_texts.csv kolonları:")
print(texts.columns.tolist())

# Sadece inceleyeceğimiz 10 makale
article_ids = review["article_id"].unique()

selected = texts[
    texts["article_id"].isin(article_ids)
].copy()

print("\n" + "=" * 110)
print("10 MAKALE - METİNLER")
print("=" * 110)

for article_id in article_ids:

    rows = selected[
        selected["article_id"] == article_id
    ]

    print("\n" + "-" * 110)
    print("ARTICLE ID:", article_id)

    if len(rows) == 0:
        print("Metin bulunamadı.")
        continue

    for _, row in rows.iterrows():

        print("\nDil:", row.get("language", ""))

        print("\nEmbedding text:")
        print(row.get("embedding_text", ""))


selected.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 110)
print("Dosya oluşturuldu:", OUTPUT_FILE)