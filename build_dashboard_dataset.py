import os
import json
import psycopg2
import pandas as pd


# ============================================================
# AYARLAR
# ============================================================

CLUSTER_FILE = "results/final_article_clusters.csv"
PROFILE_FILE = "results/final_coherent_cluster_profiles.csv"
SUBJECT_FILE = "trdizin_subject_hierarchy.csv"

OUTPUT_DIR = "results"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "dashboard_articles.csv"
)

MAIN_THRESHOLD = 0.75
SUB_THRESHOLD = 0.75
LEAF_THRESHOLD = 0.30


# ============================================================
# 1. POSTGRESQL'DEN MAKALE BİLGİLERİNİ AL
# ============================================================

print("=" * 110)
print("DASHBOARD VERİ SETİ HAZIRLANIYOR")
print("=" * 110)

conn = psycopg2.connect(
    host="localhost",
    port=5433,
    database="trdizin_ai_db",
    user="postgres",
    password="postgres"
)


article_query = """
SELECT
    a.id AS article_id,
    a.external_id,
    a.doi,
    a.publication_year,
    a.publication_type,
    at.language,
    at.title,
    at.abstract,
    at.keywords
FROM article a
JOIN article_text at
    ON a.id = at.article_id
ORDER BY a.id, at.language;
"""

texts = pd.read_sql_query(
    article_query,
    conn
)

conn.close()

print("Veritabanından gelen metin satırı:", len(texts))


# ============================================================
# 2. AYNI MAKALEDE TÜRKÇE / İNGİLİZCE METNİ TEK SATIRA TOPLA
# ============================================================

def clean_value(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def clean_keywords(value):

    if pd.isna(value):
        return ""

    try:
        parsed = json.loads(value)

        if isinstance(parsed, list):
            return ", ".join(
                str(x) for x in parsed
            )

    except (json.JSONDecodeError, TypeError):
        pass

    return str(value)


texts["title"] = texts["title"].apply(clean_value)
texts["abstract"] = texts["abstract"].apply(clean_value)
texts["keywords_clean"] = texts["keywords"].apply(clean_keywords)


article_rows = []


for article_id, group in texts.groupby("article_id"):

    first = group.iloc[0]

    tur = group[
        group["language"]
        .astype(str)
        .str.upper()
        ==
        "TUR"
    ]

    eng = group[
        group["language"]
        .astype(str)
        .str.upper()
        ==
        "ENG"
    ]


    # --------------------------------------------------------
    # Gösterilecek temel metni seç
    # Önce Türkçe, yoksa İngilizce, o da yoksa ilk kayıt
    # --------------------------------------------------------

    if not tur.empty:
        display_row = tur.iloc[0]

    elif not eng.empty:
        display_row = eng.iloc[0]

    else:
        display_row = first


    title_tr = (
        tur.iloc[0]["title"]
        if not tur.empty
        else ""
    )

    abstract_tr = (
        tur.iloc[0]["abstract"]
        if not tur.empty
        else ""
    )

    title_en = (
        eng.iloc[0]["title"]
        if not eng.empty
        else ""
    )

    abstract_en = (
        eng.iloc[0]["abstract"]
        if not eng.empty
        else ""
    )


    languages = sorted(
        set(
            group["language"]
            .dropna()
            .astype(str)
        )
    )


    article_rows.append(
        {
            "article_id":
                article_id,

            "external_id":
                first["external_id"],

            "doi":
                clean_value(first["doi"]),

            "publication_year":
                first["publication_year"],

            "publication_type":
                clean_value(
                    first["publication_type"]
                ),

            "languages":
                " || ".join(languages),

            "display_language":
                clean_value(
                    display_row["language"]
                ),

            "title":
                clean_value(
                    display_row["title"]
                ),

            "abstract":
                clean_value(
                    display_row["abstract"]
                ),

            "keywords":
                clean_value(
                    display_row[
                        "keywords_clean"
                    ]
                ),

            "title_tr":
                title_tr,

            "abstract_tr":
                abstract_tr,

            "title_en":
                title_en,

            "abstract_en":
                abstract_en
        }
    )


articles = pd.DataFrame(
    article_rows
)

print(
    "Benzersiz DB makalesi:",
    len(articles)
)


# ============================================================
# 3. FINAL CLUSTER SONUÇLARINI OKU
# ============================================================

clusters = pd.read_csv(
    CLUSTER_FILE,
    encoding="utf-8-sig"
)

profiles = pd.read_csv(
    PROFILE_FILE,
    encoding="utf-8-sig"
)

subjects = pd.read_csv(
    SUBJECT_FILE,
    encoding="utf-8-sig"
)


print(
    "Cluster sonucu bulunan makale:",
    clusters["article_id"].nunique()
)


# ============================================================
# 4. TR DİZİN ETİKETLERİNİ MAKALE BAZINDA TOPLA
# ============================================================

subjects["main_field"] = (
    subjects["main_field"]
    .fillna("")
    .astype(str)
    .str.strip()
)

subjects["sub_field"] = (
    subjects["sub_field"]
    .fillna("")
    .astype(str)
    .str.strip()
)

subjects["leaf_subject"] = (
    subjects["leaf_subject"]
    .fillna("")
    .astype(str)
    .str.strip()
)

subjects["subject_fullname"] = (
    subjects["subject_fullname"]
    .fillna("")
    .astype(str)
    .str.strip()
)


def join_unique(values):

    cleaned = []

    for value in values:

        value = str(value).strip()

        if (
            value
            and
            value not in cleaned
        ):
            cleaned.append(value)

    return " || ".join(cleaned)


subject_summary = (
    subjects
    .groupby("article_id")
    .agg(
        trdizin_main_fields=(
            "main_field",
            join_unique
        ),

        trdizin_sub_fields=(
            "sub_field",
            join_unique
        ),

        trdizin_leaf_subjects=(
            "leaf_subject",
            join_unique
        ),

        trdizin_subject_paths=(
            "subject_fullname",
            join_unique
        )
    )
    .reset_index()
)


# ============================================================
# 5. CLUSTER + PROFİL
# ============================================================

prediction = clusters.merge(
    profiles[
        [
            "cluster_id",
            "cluster_size",
            "dominant_main_field",
            "main_confidence",
            "dominant_sub_field",
            "sub_confidence_within_main",
            "top_subject",
            "leaf_confidence_within_sub"
        ]
    ],
    on="cluster_id",
    how="left"
)


# ============================================================
# 6. BULUNDU / BULUNAMADI
# ============================================================

prediction["main_found"] = (
    prediction["main_confidence"]
    >=
    MAIN_THRESHOLD
)

prediction["sub_found"] = (
    prediction[
        "sub_confidence_within_main"
    ]
    >=
    SUB_THRESHOLD
)

prediction["leaf_found"] = (
    prediction[
        "leaf_confidence_within_sub"
    ]
    >=
    LEAF_THRESHOLD
)


# Güven yeterli değilse kullanıcıya
# tahmini kesin sonuç gibi göstermiyoruz.

prediction["predicted_main_field"] = (
    prediction[
        "dominant_main_field"
    ].where(
        prediction["main_found"],
        ""
    )
)

prediction["predicted_sub_field"] = (
    prediction[
        "dominant_sub_field"
    ].where(
        prediction["sub_found"],
        ""
    )
)

prediction["predicted_subject"] = (
    prediction[
        "top_subject"
    ].where(
        prediction["leaf_found"],
        ""
    )
)


# ============================================================
# 7. TÜM VERİLERİ BİRLEŞTİR
# ============================================================

dashboard = (
    articles
    .merge(
        prediction,
        on="article_id",
        how="inner"
    )
    .merge(
        subject_summary,
        on="article_id",
        how="left"
    )
)


# ============================================================
# 8. TR DİZİN İLE UYUM KONTROLLERİ
# ============================================================

def split_labels(value):

    if pd.isna(value):
        return set()

    return {
        x.strip()
        for x in str(value).split("||")
        if x.strip()
    }


def exact_match(
    predicted,
    actual_string
):

    if pd.isna(predicted):
        return False

    predicted = str(
        predicted
    ).strip()

    if not predicted:
        return False

    actual = split_labels(
        actual_string
    )

    return predicted in actual


dashboard["main_match"] = dashboard.apply(
    lambda row:
        exact_match(
            row["predicted_main_field"],
            row["trdizin_main_fields"]
        ),
    axis=1
)


dashboard["sub_match"] = dashboard.apply(
    lambda row:
        exact_match(
            row["predicted_sub_field"],
            row["trdizin_sub_fields"]
        ),
    axis=1
)


dashboard["leaf_match"] = dashboard.apply(
    lambda row:
        exact_match(
            row["predicted_subject"],
            row["trdizin_subject_paths"]
        ),
    axis=1
)


# ============================================================
# 9. GENEL DURUM
# ============================================================

dashboard["all_levels_found"] = (
    dashboard["main_found"]
    &
    dashboard["sub_found"]
    &
    dashboard["leaf_found"]
)


dashboard["all_levels_match"] = (
    dashboard["main_match"]
    &
    dashboard["sub_match"]
    &
    dashboard["leaf_match"]
)


# DOI var mı?

dashboard["has_doi"] = (
    dashboard["doi"]
    .fillna("")
    .astype(str)
    .str.strip()
    != ""
)


# Abstract var mı?

dashboard["has_abstract"] = (
    dashboard["abstract"]
    .fillna("")
    .astype(str)
    .str.strip()
    != ""
)


# ============================================================
# 10. ARAYÜZ İÇİN KOLON SIRASI
# ============================================================

columns = [

    "article_id",
    "external_id",
    "doi",
    "has_doi",

    "publication_year",
    "publication_type",

    "languages",
    "display_language",

    "title",
    "abstract",
    "has_abstract",
    "keywords",

    "title_tr",
    "abstract_tr",

    "title_en",
    "abstract_en",

    "cluster_id",
    "cluster_size",
    "cluster_similarity",

    "predicted_main_field",
    "main_confidence",
    "main_found",

    "predicted_sub_field",
    "sub_confidence_within_main",
    "sub_found",

    "predicted_subject",
    "leaf_confidence_within_sub",
    "leaf_found",

    "trdizin_main_fields",
    "trdizin_sub_fields",
    "trdizin_leaf_subjects",
    "trdizin_subject_paths",

    "main_match",
    "sub_match",
    "leaf_match",

    "all_levels_found",
    "all_levels_match"
]


dashboard = dashboard[
    columns
]


# ============================================================
# 11. GENEL İSTATİSTİKLER
# ============================================================

print("\n" + "=" * 110)
print("DASHBOARD İSTATİSTİKLERİ")
print("=" * 110)

print(
    "Toplam makale:",
    len(dashboard)
)

print(
    "DOI bulunan:",
    int(
        dashboard["has_doi"].sum()
    )
)

print(
    "DOI bulunmayan:",
    int(
        (~dashboard["has_doi"]).sum()
    )
)

print(
    "Abstract bulunan:",
    int(
        dashboard["has_abstract"].sum()
    )
)

print(
    "Abstract bulunmayan:",
    int(
        (~dashboard["has_abstract"]).sum()
    )
)


print("\nTAHMİN DURUMU")
print("-" * 60)

print(
    "Ana alan bulunan:",
    int(
        dashboard["main_found"].sum()
    )
)

print(
    "Ana alan bulunamayan:",
    int(
        (~dashboard["main_found"]).sum()
    )
)

print(
    "Alt alan bulunan:",
    int(
        dashboard["sub_found"].sum()
    )
)

print(
    "Alt alan bulunamayan:",
    int(
        (~dashboard["sub_found"]).sum()
    )
)

print(
    "Leaf konu bulunan:",
    int(
        dashboard["leaf_found"].sum()
    )
)

print(
    "Leaf konu bulunamayan:",
    int(
        (~dashboard["leaf_found"]).sum()
    )
)

print(
    "Üç seviye de bulunan:",
    int(
        dashboard[
            "all_levels_found"
        ].sum()
    )
)


print("\nTR DİZİN EXACT UYUM")
print("-" * 60)

print(
    "Ana alan uyumu:",
    f"{dashboard['main_match'].mean():.2%}"
)

print(
    "Alt alan uyumu:",
    f"{dashboard['sub_match'].mean():.2%}"
)

print(
    "Leaf exact uyumu:",
    f"{dashboard['leaf_match'].mean():.2%}"
)


# ============================================================
# 12. KAYDET
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

dashboard.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


print("\n" + "=" * 110)
print("TAMAMLANDI")
print("=" * 110)

print(
    "Dosya oluşturuldu:",
    OUTPUT_FILE
)