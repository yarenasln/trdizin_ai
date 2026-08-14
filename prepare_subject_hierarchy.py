import psycopg2
import pandas as pd


# --------------------------------------------------
# 1. POSTGRESQL BAĞLANTISI
# --------------------------------------------------

conn = psycopg2.connect(
    host="localhost",
    port=5433,
    database="trdizin_ai_db",
    user="postgres",
    password="postgres"
)


# --------------------------------------------------
# 2. ARTICLE_SUBJECT VERİLERİNİ ÇEK
# --------------------------------------------------

query = """
SELECT
    article_id,
    subject_id,
    subject_name,
    subject_fullname,
    root_name,
    auto_label
FROM article_subject
ORDER BY article_id, subject_fullname;
"""

df = pd.read_sql_query(
    query,
    conn
)

conn.close()

print("Toplam makale-konu ilişkisi:", len(df))


# --------------------------------------------------
# 3. HİYERARŞİYİ PARÇALA
# --------------------------------------------------

def split_subject_hierarchy(fullname):

    if fullname is None:
        return "", "", ""

    parts = [
        part.strip()
        for part in str(fullname).split(">")
        if part.strip()
    ]

    # Örn:
    # Fen > Tıp > Onkoloji
    if len(parts) >= 3:

        main_field = parts[0]
        sub_field = parts[1]

        # İleride 3'ten fazla seviye gelirse
        # kalan kısmı konu olarak kaybetmeyelim.
        subject = " > ".join(parts[2:])

    # Örn:
    # Fen > Tıp
    elif len(parts) == 2:

        main_field = parts[0]
        sub_field = parts[1]
        subject = ""

    # Örn:
    # Fen
    elif len(parts) == 1:

        main_field = parts[0]
        sub_field = ""
        subject = ""

    else:

        main_field = ""
        sub_field = ""
        subject = ""

    return main_field, sub_field, subject


hierarchy = df["subject_fullname"].apply(
    split_subject_hierarchy
)

df[
    [
        "main_field",
        "sub_field",
        "leaf_subject"
    ]
] = pd.DataFrame(
    hierarchy.tolist(),
    index=df.index
)


# --------------------------------------------------
# 4. KONTROLLER
# --------------------------------------------------

print("\nANA ALANLAR")
print("=" * 60)

print(
    df["main_field"]
    .value_counts()
)


print("\nALT ALANLAR")
print("=" * 60)

sub_fields = (
    df[
        [
            "main_field",
            "sub_field"
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "main_field",
            "sub_field"
        ]
    )
)

print(
    sub_fields.to_string(
        index=False
    )
)


print("\nFARKLI EN ALT KONU SAYISI")
print("=" * 60)

leaf_count = (
    df.loc[
        df["leaf_subject"] != "",
        "subject_id"
    ]
    .nunique()
)

print(leaf_count)


# --------------------------------------------------
# 5. BİR MAKALEDE KAÇ KONU VAR?
# --------------------------------------------------

article_subject_counts = (
    df.groupby("article_id")["subject_id"]
    .nunique()
    .sort_values(
        ascending=False
    )
)

print("\nBİR MAKALEDEKİ EN FAZLA KONU SAYISI")
print("=" * 60)

print(
    article_subject_counts.head(10)
)


# --------------------------------------------------
# 6. BİRDEN FAZLA ANA ALANDA BULUNAN MAKALELER
# --------------------------------------------------

article_main_field_counts = (
    df.groupby("article_id")["main_field"]
    .nunique()
)

multi_main_articles = (
    article_main_field_counts[
        article_main_field_counts > 1
    ]
)

print("\nBİRDEN FAZLA ANA ALANDA BULUNAN MAKALE")
print("=" * 60)

print(
    "Makale sayısı:",
    len(multi_main_articles)
)


# --------------------------------------------------
# 7. ÖRNEK BİR MAKALEYİ GÖSTER
# --------------------------------------------------

if len(multi_main_articles) > 0:

    example_article_id = (
        multi_main_articles.index[0]
    )

    print("\nÖRNEK MULTI-LABEL MAKALE")
    print("=" * 60)

    example = df[
        df["article_id"] == example_article_id
    ][
        [
            "article_id",
            "main_field",
            "sub_field",
            "leaf_subject",
            "subject_fullname"
        ]
    ]

    print(
        example.to_string(
            index=False
        )
    )


# --------------------------------------------------
# 8. CSV OLARAK KAYDET
# --------------------------------------------------

df.to_csv(
    "trdizin_subject_hierarchy.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nDosya oluşturuldu:"
    " trdizin_subject_hierarchy.csv"
)