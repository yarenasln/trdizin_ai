import os
import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer


# ============================================================
# DOSYALAR
# ============================================================

TEXT_FILE = "real_trdizin_texts.csv"

CLUSTER_FILE = (
    "results/final_article_clusters.csv"
)

PROFILE_FILE = (
    "results/final_coherent_cluster_profiles.csv"
)

OUTPUT_DIR = "results"

KEYWORD_FILE = os.path.join(
    OUTPUT_DIR,
    "final_cluster_ctfidf_keywords.csv"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "final_cluster_ctfidf_summary.csv"
)


# ============================================================
# AYARLAR
# ============================================================

TOP_N = 15

# Tek kelime + iki kelimelik ifadeler
NGRAM_RANGE = (1, 2)

# Çok nadir terimleri azaltmak için
MIN_DF = 2


# ============================================================
# BASİT STOPWORD LİSTESİ
# Türkçe + İngilizce
# ============================================================

CUSTOM_STOPWORDS = {

    # Türkçe
    "ve",
    "veya",
    "ile",
    "için",
    "bu",
    "bir",
    "olarak",
    "olan",
    "olduğu",
    "olmuştur",
    "çalışma",
    "çalışmada",
    "çalışmanın",
    "sonuç",
    "sonuçlar",
    "sonuçları",
    "amaç",
    "amacı",
    "yöntem",
    "yöntemi",
    "göre",
    "daha",
    "ise",
    "da",
    "de",
    "ile",
    "üzerine",
    "arasında",
    "tarafından",
    "bulunmuştur",
    "belirlenmiştir",
    "incelenmiştir",
    "elde",
    "edilmiştir",

    # İngilizce
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "for",
    "with",
    "on",
    "by",
    "from",
    "as",
    "at",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "this",
    "that",
    "these",
    "those",
    "a",
    "an",
    "study",
    "results",
    "result",
    "method",
    "methods",
    "aim",
    "purpose",
    "using",
    "used",
    "use",
    "based",
    "between",
    "among",
    "showed",
    "found",
    "analysis"
}


# ============================================================
# METİN TEMİZLEME
# ============================================================

def clean_text(value):

    if pd.isna(value):
        return ""

    text = str(value)

    # URL kaldır
    text = re.sub(
        r"http\S+|www\.\S+",
        " ",
        text
    )

    # Fazla boşlukları temizle
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# VERİLERİ OKU
# ============================================================

print("=" * 110)
print("K-MEANS + c-TF-IDF CLUSTER KONU TEMSİLİ")
print("=" * 110)


texts = pd.read_csv(
    TEXT_FILE,
    encoding="utf-8-sig"
)

clusters = pd.read_csv(
    CLUSTER_FILE,
    encoding="utf-8-sig"
)

profiles = pd.read_csv(
    PROFILE_FILE,
    encoding="utf-8-sig"
)


print(
    "Metin satırı:",
    len(texts)
)

print(
    "Cluster makalesi:",
    clusters["article_id"].nunique()
)

print(
    "Cluster sayısı:",
    clusters["cluster_id"].nunique()
)


# ============================================================
# HER MAKALEDEN TEK METİN SEÇ
# ============================================================
#
# Aynı makalede TUR ve ENG varsa:
#
# 1. TUR tercih edilir
# 2. TUR yoksa ENG
# 3. İkisi de yoksa ilk kayıt
#
# Böylece aynı makalenin iki farklı dildeki versiyonu
# cluster metninde iki kat ağırlık oluşturmaz.
# ============================================================

article_rows = []


for article_id, group in texts.groupby(
    "article_id"
):

    group = group.copy()

    group["language"] = (
        group["language"]
        .fillna("")
        .astype(str)
        .str.upper()
    )


    tur = group[
        group["language"] == "TUR"
    ]

    eng = group[
        group["language"] == "ENG"
    ]


    if not tur.empty:

        selected = tur.iloc[0]

    elif not eng.empty:

        selected = eng.iloc[0]

    else:

        selected = group.iloc[0]


    # prepare_real_data.py içinde zaten
    # title + abstract + keyword birleşmişti.
    text = clean_text(
        selected["embedding_text"]
    )


    article_rows.append(
        {
            "article_id":
                article_id,

            "selected_language":
                selected["language"],

            "text":
                text
        }
    )


article_texts = pd.DataFrame(
    article_rows
)


print(
    "Tekilleştirilmiş makale metni:",
    len(article_texts)
)


# ============================================================
# CLUSTER BİLGİSİYLE BİRLEŞTİR
# ============================================================

merged = clusters[
    [
        "article_id",
        "cluster_id"
    ]
].merge(
    article_texts,
    on="article_id",
    how="left"
)


merged["text"] = (
    merged["text"]
    .fillna("")
    .astype(str)
)


# Boş metinleri çıkar
merged = merged[
    merged["text"].str.strip() != ""
].copy()


print(
    "c-TF-IDF için kullanılan makale:",
    merged["article_id"].nunique()
)


# ============================================================
# HER CLUSTER'I TEK BÜYÜK DOKÜMAN GİBİ ELE AL
# ============================================================
#
# BERTopic'teki temel fikir:
#
# Aynı sınıf / cluster içindeki dokümanlar birleştirilir.
#
# Cluster 3:
#
# Makale A
# Makale B
# Makale C
#
#          ↓
#
# Tek büyük Cluster 3 dokümanı
# ============================================================

cluster_documents = (
    merged
    .groupby("cluster_id")["text"]
    .apply(
        lambda values:
            " ".join(
                values.astype(str)
            )
    )
    .reset_index()
)


print(
    "Cluster dokümanı:",
    len(cluster_documents)
)


# ============================================================
# COUNT VECTORIZER
# ============================================================

vectorizer = CountVectorizer(

    lowercase=True,

    stop_words=list(
        CUSTOM_STOPWORDS
    ),

    ngram_range=NGRAM_RANGE,

    min_df=MIN_DF,

    token_pattern=(
        r"(?u)\b"
        r"[A-Za-zÇĞİÖŞÜ"
        r"çğıöşü]"
        r"[A-Za-zÇĞİÖŞÜ"
        r"çğıöşü0-9\-]{2,}"
        r"\b"
    )
)


count_matrix = vectorizer.fit_transform(
    cluster_documents["text"]
)


terms = np.array(
    vectorizer.get_feature_names_out()
)


print(
    "Terim sayısı:",
    len(terms)
)

print(
    "Count matrix:",
    count_matrix.shape
)


# ============================================================
# c-TF-IDF
# ============================================================
#
# BERTopic yaklaşımına benzer şekilde:
#
# 1. Cluster içindeki term frequency normalize edilir.
# 2. Tüm cluster'larda terimin genel sıklığı hesaplanır.
# 3. Sık görülen genel kelimelerin ağırlığı azaltılır.
# 4. Bir cluster'a özgü terimlerin ağırlığı artırılır.
# ============================================================

counts = count_matrix.toarray().astype(
    np.float64
)


# ------------------------------------------------------------
# CLASS TERM FREQUENCY
# ------------------------------------------------------------

row_sums = counts.sum(
    axis=1,
    keepdims=True
)


row_sums[
    row_sums == 0
] = 1


class_tf = (
    counts
    /
    row_sums
)


# ------------------------------------------------------------
# IDF
# ------------------------------------------------------------
#
# Her terimin tüm cluster'lardaki toplam frekansı
# ------------------------------------------------------------

term_frequency = counts.sum(
    axis=0
)


term_frequency[
    term_frequency == 0
] = 1


# Ortalama cluster doküman uzunluğu
average_class_length = (
    row_sums.mean()
)


idf = np.log(
    1
    +
    (
        average_class_length
        /
        term_frequency
    )
)


# ------------------------------------------------------------
# FINAL c-TF-IDF
# ------------------------------------------------------------

ctfidf = (
    class_tf
    *
    idf
)


print(
    "c-TF-IDF matrix:",
    ctfidf.shape
)


# ============================================================
# HER CLUSTER İÇİN TOP TERİMLER
# ============================================================

keyword_rows = []
summary_rows = []


for row_index, cluster_row in (
    cluster_documents.iterrows()
):

    cluster_id = int(
        cluster_row["cluster_id"]
    )


    scores = ctfidf[
        row_index
    ]


    # Büyükten küçüğe sırala
    top_indices = np.argsort(
        scores
    )[::-1]


    top_terms = []


    for term_index in top_indices:

        score = scores[
            term_index
        ]

        if score <= 0:
            continue


        term = terms[
            term_index
        ]


        top_terms.append(
            (
                term,
                score
            )
        )


        if len(top_terms) >= TOP_N:
            break


    # --------------------------------------------------------
    # DETAY SATIRLARI
    # --------------------------------------------------------

    for rank, (
        term,
        score
    ) in enumerate(
        top_terms,
        start=1
    ):

        keyword_rows.append(
            {
                "cluster_id":
                    cluster_id,

                "rank":
                    rank,

                "term":
                    term,

                "ctfidf_score":
                    round(
                        float(score),
                        8
                    )
            }
        )


    # --------------------------------------------------------
    # TEK SATIR ÖZET
    # --------------------------------------------------------

    summary_rows.append(
        {
            "cluster_id":
                cluster_id,

            "ctfidf_keywords":
                " || ".join(
                    [
                        term
                        for term, _
                        in top_terms
                    ]
                )
        }
    )


keywords_df = pd.DataFrame(
    keyword_rows
)

summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# CLUSTER PROFİLİYLE BİRLEŞTİR
# ============================================================

summary_df = profiles.merge(

    summary_df,

    on="cluster_id",

    how="left"
)


# ============================================================
# KAYDET
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


keywords_df.to_csv(
    KEYWORD_FILE,
    index=False,
    encoding="utf-8-sig"
)


summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# ÖRNEK CLUSTER'LAR
# ============================================================

print("\n" + "=" * 130)
print("ÖRNEK c-TF-IDF CLUSTER TEMSİLLERİ")
print("=" * 130)


example_clusters = (
    summary_df
    .sort_values(
        "cluster_id"
    )
    .head(15)
)


for _, row in (
    example_clusters.iterrows()
):

    print("\n" + "-" * 110)

    print(
        "Cluster:",
        int(
            row["cluster_id"]
        )
    )

    print(
        "Cluster boyutu:",
        int(
            row["cluster_size"]
        )
    )

    print(
        "Ana alan:",
        row[
            "dominant_main_field"
        ]
    )

    print(
        "Alt alan:",
        row[
            "dominant_sub_field"
        ]
    )

    print(
        "Leaf konu:",
        row[
            "top_subject"
        ]
    )

    print(
        "c-TF-IDF:"
    )


    cluster_keywords = (
        keywords_df[
            keywords_df[
                "cluster_id"
            ]
            ==
            row["cluster_id"]
        ]
        .sort_values(
            "rank"
        )
    )


    for _, keyword in (
        cluster_keywords.iterrows()
    ):

        print(
            f"  {int(keyword['rank'])}. "
            f"{keyword['term']} "
            f"| score="
            f"{keyword['ctfidf_score']:.6f}"
        )


# ============================================================
# GENEL SONUÇ
# ============================================================

print("\n" + "=" * 110)
print("TAMAMLANDI")
print("=" * 110)

print(
    "Cluster sayısı:",
    summary_df[
        "cluster_id"
    ].nunique()
)

print(
    "Toplam keyword satırı:",
    len(
        keywords_df
    )
)

print(
    "\nDosya oluşturuldu:",
    KEYWORD_FILE
)

print(
    "Dosya oluşturuldu:",
    SUMMARY_FILE
)