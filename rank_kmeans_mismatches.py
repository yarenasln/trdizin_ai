import pandas as pd
import numpy as np


# ============================================================
# DOSYA
# ============================================================

INPUT_FILE = "kmeans_final_topic_predictions.csv"


# ============================================================
# VERİ
# ============================================================

df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8-sig"
)


# ============================================================
# SADECE UYUŞMAZLIKLAR
# ============================================================

mismatch = df[
    df["status"] == "UYUSMAZLIK"
].copy()


print("=" * 100)
print("K-MEANS UYUŞMAZLIK ANALİZİ")
print("=" * 100)

print(
    "Uyuşmazlık makale sayısı:",
    mismatch["article_id"].nunique()
)


# ============================================================
# SAYISAL KOLONLARI TEMİZLE
# ============================================================

for column in [
    "prediction_score",
    "support_count",
    "best_similarity",
    "selected_cluster_count"
]:

    mismatch[column] = pd.to_numeric(
        mismatch[column],
        errors="coerce"
    ).fillna(0)


# ============================================================
# HER MAKALE İÇİN EN GÜÇLÜ TAHMİN
# ============================================================
#
# Bir makaleye birden fazla aday konu verilmiş olabilir.
# Burada en güçlü tahmini ana güven sinyali olarak kullanıyoruz.
# ============================================================

best_predictions = (
    mismatch
    .sort_values(
        [
            "article_id",
            "prediction_score"
        ],
        ascending=[
            True,
            False
        ]
    )
    .groupby(
        "article_id",
        as_index=False
    )
    .first()
)


# ============================================================
# NORMALIZATION (NORMALİZASYON)
# ============================================================

def normalize(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:

        return pd.Series(
            np.ones(len(series)),
            index=series.index
        )

    return (
        series - minimum
    ) / (
        maximum - minimum
    )


best_predictions[
    "score_norm"
] = normalize(
    best_predictions[
        "prediction_score"
    ]
)


best_predictions[
    "support_norm"
] = normalize(
    best_predictions[
        "support_count"
    ]
)


best_predictions[
    "similarity_norm"
] = normalize(
    best_predictions[
        "best_similarity"
    ]
)


# ============================================================
# CLUSTER CEZASI
# ============================================================
#
# Az cluster seçilmişse daha güvenli kabul ediyoruz.
#
# 1 cluster → yüksek güven
# çok cluster → daha düşük güven
# ============================================================

best_predictions[
    "cluster_penalty"
] = (

    1
    /
    best_predictions[
        "selected_cluster_count"
    ].clip(
        lower=1
    )
)


best_predictions[
    "cluster_penalty_norm"
] = normalize(
    best_predictions[
        "cluster_penalty"
    ]
)


# ============================================================
# MISMATCH CONFIDENCE
# (UYUŞMAZLIK GÜVEN SKORU)
# ============================================================
#
# Bu standart akademik bir metrik değildir.
# Sadece manuel inceleme önceliği için kullanıyoruz.
#
# %40 prediction score
# %25 support
# %25 centroid similarity
# %10 az cluster seçilmesi
# ============================================================

best_predictions[
    "mismatch_confidence"
] = (

    0.40
    *
    best_predictions[
        "score_norm"
    ]

    +

    0.25
    *
    best_predictions[
        "support_norm"
    ]

    +

    0.25
    *
    best_predictions[
        "similarity_norm"
    ]

    +

    0.10
    *
    best_predictions[
        "cluster_penalty_norm"
    ]
)


# ============================================================
# YÜKSEKTEN DÜŞÜĞE SIRALA
# ============================================================

ranked = (
    best_predictions
    .sort_values(
        "mismatch_confidence",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


ranked[
    "mismatch_rank"
] = (
    ranked.index
    +
    1
)


# ============================================================
# GÜVEN SEVİYESİ
# ============================================================

def confidence_label(score):

    if score >= 0.75:
        return "YUKSEK"

    elif score >= 0.50:
        return "ORTA"

    else:
        return "DUSUK"


ranked[
    "confidence_level"
] = ranked[
    "mismatch_confidence"
].apply(
    confidence_label
)


# ============================================================
# KOLONLARI SEÇ
# ============================================================

output = ranked[
    [
        "mismatch_rank",
        "article_id",
        "title",

        "predicted_subject",
        "prediction_score",
        "support_count",

        "trdizin_subjects",

        "best_cluster",
        "best_similarity",
        "selected_cluster_count",

        "mismatch_confidence",
        "confidence_level"
    ]
].copy()


# ============================================================
# EKRANA İLK 30
# ============================================================

print("\n" + "=" * 120)
print("EN YÜKSEK GÜVENLİ 30 UYUŞMAZLIK")
print("=" * 120)


pd.set_option(
    "display.max_colwidth",
    120
)

pd.set_option(
    "display.width",
    250
)


print(
    output
    .head(30)
    .to_string(
        index=False
    )
)


# ============================================================
# GÜVEN DAĞILIMI
# ============================================================

print("\n" + "=" * 100)
print("GÜVEN SEVİYESİ DAĞILIMI")
print("=" * 100)


print(
    output[
        "confidence_level"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# CSV
# ============================================================

output.to_csv(
    "ranked_kmeans_mismatches.csv",
    index=False,
    encoding="utf-8-sig"
)


print(
    "\nDosya oluşturuldu:"
    " ranked_kmeans_mismatches.csv"
)