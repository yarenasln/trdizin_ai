import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# DOSYALAR
# ============================================================

MISMATCH_FILE = "final_mismatch_analysis.csv"
SUBJECT_METADATA_FILE = "Qwen3_subject_metadata.csv"
SUBJECT_EMBEDDING_FILE = "embeddings/Qwen3_subject_embeddings.npy"

OUTPUT_FILE = "near_subject_similarity_analysis.csv"


# ============================================================
# VERİLERİ OKU
# ============================================================

mismatch = pd.read_csv(
    MISMATCH_FILE,
    encoding="utf-8-sig"
)

subject_metadata = pd.read_csv(
    SUBJECT_METADATA_FILE,
    encoding="utf-8-sig"
)

subject_embeddings = np.load(
    SUBJECT_EMBEDDING_FILE
).astype(np.float32)


print("=" * 110)
print("YAKIN ALT KONU SEMANTIC SIMILARITY ANALİZİ")
print("=" * 110)

print(
    "Mismatch makale:",
    mismatch["article_id"].nunique()
)

print(
    "Konu embedding:",
    subject_embeddings.shape
)


# ============================================================
# SADECE YAKIN ALT KONU HATALARI
# ============================================================

near_errors = mismatch[
    mismatch["error_type"]
    ==
    "YAKIN_ALT_KONU_HATASI"
].copy()


print(
    "Yakın alt konu hatası:",
    len(near_errors)
)


# ============================================================
# SUBJECT -> INDEX
# ============================================================

subject_index = {}


for _, row in subject_metadata.iterrows():

    subject_index[
        str(
            row["subject_fullname"]
        ).strip()
    ] = int(
        row["subject_embedding_id"]
    )


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def parse_subjects(value):

    if pd.isna(value):
        return []

    return [
        x.strip()
        for x in str(value).split("||")
        if x.strip()
    ]


def similarity_between(subject1, subject2):

    if (
        subject1 not in subject_index
        or
        subject2 not in subject_index
    ):

        return np.nan


    vec1 = subject_embeddings[
        subject_index[
            subject1
        ]
    ]


    vec2 = subject_embeddings[
        subject_index[
            subject2
        ]
    ]


    similarity = float(
        cosine_similarity(
            vec1.reshape(
                1,
                -1
            ),
            vec2.reshape(
                1,
                -1
            )
        )[0][0]
    )


    return similarity


# ============================================================
# ANALİZ
# ============================================================

rows = []


for _, row in near_errors.iterrows():

    predicted_subjects = parse_subjects(
        row["predicted_subjects"]
    )

    real_subjects = parse_subjects(
        row["trdizin_subjects"]
    )


    best_similarity = -1
    best_predicted = ""
    best_real = ""


    all_pairs = []


    for predicted in predicted_subjects:

        for real in real_subjects:

            similarity = similarity_between(
                predicted,
                real
            )


            if pd.isna(
                similarity
            ):
                continue


            all_pairs.append(
                (
                    predicted,
                    real,
                    similarity
                )
            )


            if similarity > best_similarity:

                best_similarity = similarity
                best_predicted = predicted
                best_real = real


    # ========================================================
    # EN KÖTÜ / ORTALAMA BENZERLİK
    # ========================================================

    similarities = [
        x[2]
        for x in all_pairs
    ]


    mean_similarity = (
        np.mean(similarities)
        if similarities
        else np.nan
    )


    max_similarity = (
        np.max(similarities)
        if similarities
        else np.nan
    )


    min_similarity = (
        np.min(similarities)
        if similarities
        else np.nan
    )


    # ========================================================
    # YAKINLIK SINIFI
    # ========================================================
    #
    # Bunlar şimdilik analiz sınıflarıdır.
    # Final karar eşiği değildir.
    # ========================================================

    if pd.isna(
        max_similarity
    ):

        similarity_level = "BILINMIYOR"


    elif max_similarity >= 0.75:

        similarity_level = "COK_YAKIN"


    elif max_similarity >= 0.60:

        similarity_level = "YAKIN"


    elif max_similarity >= 0.45:

        similarity_level = "ORTA_YAKIN"


    else:

        similarity_level = "UZAK"


    rows.append(
        {
            "article_id":
                row["article_id"],

            "title":
                row["title"],

            "trdizin_subjects":
                row["trdizin_subjects"],

            "predicted_subjects":
                row["predicted_subjects"],

            "top_prediction":
                row["top_prediction"],

            "closest_actual_subject":
                row["closest_actual_subject"],

            "best_subject_pair_predicted":
                best_predicted,

            "best_subject_pair_actual":
                best_real,

            "max_subject_similarity":
                round(
                    max_similarity,
                    4
                )
                if pd.notna(
                    max_similarity
                )
                else np.nan,

            "mean_subject_similarity":
                round(
                    mean_similarity,
                    4
                )
                if pd.notna(
                    mean_similarity
                )
                else np.nan,

            "min_subject_similarity":
                round(
                    min_similarity,
                    4
                )
                if pd.notna(
                    min_similarity
                )
                else np.nan,

            "similarity_level":
                similarity_level,

            "top_kmeans_score":
                row["top_kmeans_score"],

            "top_semantic_score":
                row["top_semantic_score"],

            "top_final_score":
                row["top_final_score"],

            "best_similarity":
                row["best_similarity"],

            "selected_cluster_count":
                row["selected_cluster_count"]
        }
    )


# ============================================================
# DATAFRAME
# ============================================================

result = pd.DataFrame(
    rows
)


# ============================================================
# KAYDET
# ============================================================

result.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# GENEL DAĞILIM
# ============================================================

print("\n" + "=" * 110)
print("KONU-KONU BENZERLİK DAĞILIMI")
print("=" * 110)


print(
    result[
        "similarity_level"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# İSTATİSTİKLER
# ============================================================

print("\n" + "=" * 110)
print("SEMANTIC SIMILARITY İSTATİSTİKLERİ")
print("=" * 110)


print(
    "Ortalama max similarity:",
    round(
        result[
            "max_subject_similarity"
        ].mean(),
        4
    )
)


print(
    "Medyan max similarity:",
    round(
        result[
            "max_subject_similarity"
        ].median(),
        4
    )
)


print(
    "Minimum:",
    round(
        result[
            "max_subject_similarity"
        ].min(),
        4
    )
)


print(
    "Maksimum:",
    round(
        result[
            "max_subject_similarity"
        ].max(),
        4
    )
)


# ============================================================
# PERCENTILE
# ============================================================

print("\nPercentiller:")


for p in [
    10,
    25,
    50,
    75,
    90,
    95
]:

    value = np.percentile(
        result[
            "max_subject_similarity"
        ].dropna(),
        p
    )


    print(
        f"P{p}:",
        round(
            float(value),
            4
        )
    )


# ============================================================
# EN YAKIN 15
# ============================================================

print("\n" + "=" * 120)
print("ANLAMSAL OLARAK EN YAKIN 15 UYUŞMAZLIK")
print("=" * 120)


columns = [
    "article_id",
    "best_subject_pair_predicted",
    "best_subject_pair_actual",
    "max_subject_similarity",
    "top_final_score"
]


print(
    result
    .sort_values(
        "max_subject_similarity",
        ascending=False
    )
    [columns]
    .head(15)
    .to_string(
        index=False
    )
)


# ============================================================
# EN UZAK 15
# ============================================================

print("\n" + "=" * 120)
print("ANLAMSAL OLARAK EN UZAK 15 YAKIN-ALT-KONU HATASI")
print("=" * 120)


print(
    result
    .sort_values(
        "max_subject_similarity",
        ascending=True
    )
    [columns]
    .head(15)
    .to_string(
        index=False
    )
)


# ============================================================
# FINAL SCORE İLE SUBJECT SIMILARITY İLİŞKİSİ
# ============================================================

valid = result[
    [
        "top_final_score",
        "max_subject_similarity"
    ]
].dropna()


if len(
    valid
) > 1:

    correlation = valid[
        "top_final_score"
    ].corr(
        valid[
            "max_subject_similarity"
        ]
    )


else:

    correlation = np.nan


print("\n" + "=" * 110)
print("FINAL SCORE - KONU BENZERLİĞİ İLİŞKİSİ")
print("=" * 110)


print(
    "Pearson correlation:",
    round(
        correlation,
        4
    )
    if pd.notna(
        correlation
    )
    else "hesaplanamadı"
)


# ============================================================
# OLASI 'GERÇEK HATA' ADAYLARI
# ============================================================
#
# Sistem yüksek güvenle tahmin yapmış
# ama gerçek etiket ile tahmin edilen konu
# anlamsal olarak uzak.
#
# Bunlar özellikle incelenmeli.
# ============================================================

suspicious = result[
    (
        result[
            "top_final_score"
        ]
        >=
        0.50
    )
    &
    (
        result[
            "max_subject_similarity"
        ]
        <
        0.45
    )
].copy()


print("\n" + "=" * 120)
print("YÜKSEK GÜVEN AMA KONU OLARAK UZAK KAYITLAR")
print("=" * 120)


print(
    "Makale sayısı:",
    len(
        suspicious
    )
)


if not suspicious.empty:

    print(
        suspicious[
            [
                "article_id",
                "top_prediction",
                "closest_actual_subject",
                "max_subject_similarity",
                "top_final_score"
            ]
        ]
        .sort_values(
            "top_final_score",
            ascending=False
        )
        .head(20)
        .to_string(
            index=False
        )
    )


# ============================================================
# DOSYA
# ============================================================

print(
    "\nDosya oluşturuldu:",
    OUTPUT_FILE
)