import psycopg2
import pandas as pd
import json


# --------------------------------------------------
# POSTGRESQL BAĞLANTISI
# --------------------------------------------------

conn = psycopg2.connect(
    host="localhost",
    port=5433,
    database="trdizin_ai_db",
    user="postgres",
    password="postgres"
)


# --------------------------------------------------
# MAKALE + METİNLERİ ÇEK
# --------------------------------------------------

query = """
SELECT
    a.id AS article_id,
    a.external_id,
    at.language,
    at.title,
    at.abstract,
    at.keywords
FROM article a
JOIN article_text at
    ON a.id = at.article_id
ORDER BY a.id;
"""

df = pd.read_sql_query(query, conn)

conn.close()


print("Veritabanından gelen metin:", len(df))


# --------------------------------------------------
# KEYWORDS TEMİZLE
# --------------------------------------------------

def clean_keywords(value):

    if value is None:
        return ""

    try:
        keywords = json.loads(value)

        if isinstance(keywords, list):
            return ", ".join(str(x) for x in keywords)

    except (json.JSONDecodeError, TypeError):
        pass

    return str(value)


df["keywords_clean"] = df["keywords"].apply(clean_keywords)


# --------------------------------------------------
# BOŞ DEĞERLERİ TEMİZLE
# --------------------------------------------------

df["title"] = df["title"].fillna("")
df["abstract"] = df["abstract"].fillna("")
df["keywords_clean"] = df["keywords_clean"].fillna("")


# --------------------------------------------------
# EMBEDDING METNİ
# --------------------------------------------------

df["embedding_text"] = (
    df["title"].str.strip()
    + ". "
    + df["abstract"].str.strip()
    + ". Keywords: "
    + df["keywords_clean"].str.strip()
)


# --------------------------------------------------
# ÇOK KISA / BOŞ KAYITLARI ELE
# --------------------------------------------------

df = df[
    (df["title"].str.strip() != "")
    &
    (df["embedding_text"].str.len() > 50)
].copy()


print("Embedding için kullanılabilir metin:", len(df))


# --------------------------------------------------
# ÖRNEK GÖSTER
# --------------------------------------------------

print("\n" + "=" * 70)
print("ÖRNEK EMBEDDING METNİ")
print("=" * 70)

if len(df) > 0:

    row = df.iloc[0]

    print("Article ID:", row["article_id"])
    print("Dil:", row["language"])

    print("\nMETİN:")
    print(row["embedding_text"][:1000])


# --------------------------------------------------
# CSV OLARAK KAYDET
# --------------------------------------------------

df.to_csv(
    "real_trdizin_texts.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nDosya oluşturuldu: real_trdizin_texts.csv")