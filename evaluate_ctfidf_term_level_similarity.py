import os
import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer


# ============================================================
# 1. AYARLAR
# ============================================================

INPUT_FILE = "results/final_cluster_ctfidf_summary.csv"

OUTPUT_FILE = (
    "results/ctfidf_term_level_similarity.csv"
)

DETAIL_OUTPUT_FILE = (
    "results/ctfidf_term_level_details.csv"
)

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

# Her cluster için ilk kaç c-TF-IDF terimini inceleyelim?
TOP_N_TERMS = 10

# En iyi kaç terimin ortalamasını alalım?
TOP_K = 5


# ============================================================
# 2. VERİYİ OKU
# ============================================================

df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8-sig"
)

print("=" * 110)
print("c-TF-IDF TERM-LEVEL SEMANTIC VALIDATION")
print("=" * 110)

print(
    "Toplam cluster:",
    df["cluster_id"].nunique()
)


# ============================================================
# 3. BOŞ DEĞERLERİ TEMİZLE
# ============================================================

df["top_subject"] = (
    df["top_subject"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["ctfidf_keywords"] = (
    df["ctfidf_keywords"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# ============================================================
# 4. LEAF SUBJECT ÇIKAR
# ============================================================

def extract_leaf_subject(full_path):

    parts = [
        part.strip()
        for part in str(full_path).split(">")
        if part.strip()
    ]

    if not parts:
        return ""

    return parts[-1]


df["leaf_subject"] = (
    df["top_subject"]
    .apply(extract_leaf_subject)
)


# ============================================================
# 5. c-TF-IDF TERİMLERİNİ AYIR
# ============================================================

def extract_terms(value):

    terms = [
        term.strip()
        for term in str(value).split("||")
        if term.strip()
    ]

    return terms[:TOP_N_TERMS]


df["terms"] = (
    df["ctfidf_keywords"]
    .apply(extract_terms)
)


# Boş kayıtları çıkar
df = df[
    (df["leaf_subject"] != "")
    &
    (df["terms"].apply(len) > 0)
].copy()


print(
    "Değerlendirilecek cluster:",
    len(df)
)


# ============================================================
# 6. DEVICE
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Kullanılan cihaz:",
    device
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 7. QWEN3
# ============================================================

print("\nQwen3 yükleniyor...")

model = SentenceTransformer(
    MODEL_NAME,
    device=device
)


# ============================================================
# 8. TÜM LEAF SUBJECTLERİ EMBED ET
# ============================================================

print(
    "\nLeaf subject embeddingleri üretiliyor..."
)

leaf_embeddings = model.encode(
    df["leaf_subject"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True
)


# ============================================================
# 9. TÜM TERİMLERİ TEK SEFERDE EMBED ET
# ============================================================
#
# Aynı kelime birden fazla cluster'da bulunabilir.
# Aynı terimi tekrar tekrar modele vermemek için
# önce benzersiz terimleri topluyoruz.
# ============================================================

all_terms = []

for terms in df["terms"]:

    all_terms.extend(
        terms
    )


unique_terms = sorted(
    set(all_terms)
)


print(
    "Benzersiz c-TF-IDF terimi:",
    len(unique_terms)
)


print(
    "Terim embeddingleri üretiliyor..."
)


term_embeddings = model.encode(
    unique_terms,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True
)


# Terim -> embedding
term_embedding_map = {

    term: embedding

    for term, embedding
    in zip(
        unique_terms,
        term_embeddings
    )
}


# ============================================================
# 10. CLUSTER BAZINDA KARŞILAŞTIR
# ============================================================

summary_rows = []
detail_rows = []


for row_position, (_, row) in enumerate(
    df.iterrows()
):

    cluster_id = int(
        row["cluster_id"]
    )

    leaf_subject = row[
        "leaf_subject"
    ]

    terms = row[
        "terms"
    ]

    leaf_vector = (
        leaf_embeddings[
            row_position
        ]
    )


    similarities = []


    # --------------------------------------------------------
    # HER TERİMİ LEAF İLE AYRI AYRI KARŞILAŞTIR
    # --------------------------------------------------------

    for term in terms:

        term_vector = (
            term_embedding_map[
                term
            ]
        )

        # Normalize embedding kullandığımız için
        # dot product = cosine similarity
        similarity = float(
            np.dot(
                leaf_vector,
                term_vector
            )
        )

        similarities.append(
            (
                term,
                similarity
            )
        )


    # --------------------------------------------------------
    # BENZERLİĞE GÖRE SIRALA
    # --------------------------------------------------------

    similarities.sort(
        key=lambda x: x[1],
        reverse=True
    )


    scores = [
        score
        for _, score
        in similarities
    ]


    # --------------------------------------------------------
    # METRİKLER
    # --------------------------------------------------------

    max_similarity = (
        max(scores)
        if scores
        else 0
    )

    mean_similarity = (
        np.mean(scores)
        if scores
        else 0
    )


    top_k_values = (
        scores[:TOP_K]
    )

    top_k_mean = (
        np.mean(top_k_values)
        if top_k_values
        else 0
    )


    median_similarity = (
        np.median(scores)
        if scores
        else 0
    )


    # --------------------------------------------------------
    # EN İYİ TERİMLER
    # --------------------------------------------------------

    best_terms = (
        similarities[:TOP_K]
    )


    best_terms_text = " || ".join(
        [
            f"{term} ({score:.4f})"
            for term, score
            in best_terms
        ]
    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary_rows.append(
        {
            "cluster_id":
                cluster_id,

            "leaf_subject":
                leaf_subject,

            "term_count":
                len(terms),

            "max_term_similarity":
                max_similarity,

            "mean_term_similarity":
                mean_similarity,

            "median_term_similarity":
                median_similarity,

            "top5_mean_similarity":
                top_k_mean,

            "best_matching_terms":
                best_terms_text
        }
    )


    # --------------------------------------------------------
    # DETAIL
    # --------------------------------------------------------

    for rank, (
        term,
        similarity
    ) in enumerate(
        similarities,
        start=1
    ):

        detail_rows.append(
            {
                "cluster_id":
                    cluster_id,

                "leaf_subject":
                    leaf_subject,

                "rank":
                    rank,

                "term":
                    term,

                "similarity":
                    similarity
            }
        )


# ============================================================
# 11. DATAFRAME
# ============================================================

results = pd.DataFrame(
    summary_rows
)

details = pd.DataFrame(
    detail_rows
)


# ============================================================
# 12. GENEL İSTATİSTİKLER
# ============================================================

print("\n" + "=" * 110)
print("GENEL TERM-LEVEL SONUÇ")
print("=" * 110)


metrics = [

    (
        "MAX TERM",
        "max_term_similarity"
    ),

    (
        "TOP-5 ORTALAMA",
        "top5_mean_similarity"
    ),

    (
        "TÜM TERİMLER ORTALAMA",
        "mean_term_similarity"
    ),

    (
        "MEDYAN",
        "median_term_similarity"
    )
]


for name, column in metrics:

    print(
        f"\n{name}"
    )

    print(
        "Ortalama:",
        round(
            results[column].mean(),
            4
        )
    )

    print(
        "Medyan:",
        round(
            results[column].median(),
            4
        )
    )

    print(
        "Minimum:",
        round(
            results[column].min(),
            4
        )
    )

    print(
        "Maksimum:",
        round(
            results[column].max(),
            4
        )
    )


# ============================================================
# 13. TOP-5 PERCENTİLLER
# ============================================================

print("\n" + "=" * 110)
print("TOP-5 MEAN PERCENTİLLERİ")
print("=" * 110)


for percentile in [
    5,
    10,
    25,
    50,
    75,
    90,
    95
]:

    value = (
        results[
            "top5_mean_similarity"
        ]
        .quantile(
            percentile / 100
        )
    )

    print(
        f"P{percentile}:",
        round(
            value,
            4
        )
    )


# ============================================================
# 14. EN İYİ 15 CLUSTER
# ============================================================

print("\n" + "=" * 140)
print("TERM-LEVEL UYUMU EN YÜKSEK 15 CLUSTER")
print("=" * 140)


print(
    results
    .sort_values(
        "top5_mean_similarity",
        ascending=False
    )
    .head(15)
    [
        [
            "cluster_id",
            "leaf_subject",
            "max_term_similarity",
            "top5_mean_similarity",
            "best_matching_terms"
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 15. EN DÜŞÜK 15 CLUSTER
# ============================================================

print("\n" + "=" * 140)
print("TERM-LEVEL UYUMU EN DÜŞÜK 15 CLUSTER")
print("=" * 140)


print(
    results
    .sort_values(
        "top5_mean_similarity",
        ascending=True
    )
    .head(15)
    [
        [
            "cluster_id",
            "leaf_subject",
            "max_term_similarity",
            "top5_mean_similarity",
            "best_matching_terms"
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 16. EN İYİ EŞLEŞEN TERİMLERDEN ÖRNEKLER
# ============================================================

print("\n" + "=" * 140)
print("ÖRNEK CLUSTER - TERİM EŞLEŞMELERİ")
print("=" * 140)


for _, row in (
    results
    .sort_values(
        "cluster_id"
    )
    .head(10)
    .iterrows()
):

    print(
        f"\nCluster {int(row['cluster_id'])}"
    )

    print(
        "Leaf:",
        row["leaf_subject"]
    )

    print(
        "En iyi terimler:",
        row["best_matching_terms"]
    )

    print(
        "Top-5 ortalama:",
        round(
            row[
                "top5_mean_similarity"
            ],
            4
        )
    )


# ============================================================
# 17. DOSYALARI KAYDET
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)


results.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


details.to_csv(
    DETAIL_OUTPUT_FILE,
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

print(
    "Dosya oluşturuldu:",
    DETAIL_OUTPUT_FILE
)