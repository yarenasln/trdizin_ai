import pandas as pd
import numpy as np

# ============================================================
# AYARLAR
# ============================================================

INPUT_FILE = "final_hybrid_mismatches.csv"
OUTPUT_ARTICLES = "final_mismatch_analysis.csv"
OUTPUT_SUMMARY = "final_mismatch_summary.csv"

# Aynı ana alanı paylaşan tahminleri "yakın alan" kabul etmek için kullanacağız.
# Örn:
# Fen > Temel Bilimler > Kimya, Organik
# Fen > Temel Bilimler > Kimya, İnorganik ve Nükleer
#
# İlk iki seviye aynıdır: Fen > Temel Bilimler

# ============================================================
# VERİYİ OKU
# ============================================================

df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

print("=" * 110)
print("FINAL HİBRİT SİSTEM - UYUŞMAZLIK ANALİZİ")
print("=" * 110)

print("CSV satırı:", len(df))
print("Benzersiz uyuşmaz makale:", df["article_id"].nunique())


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def parse_subjects(value):
    """TR Dizin konu listesini parçalar."""

    if pd.isna(value):
        return []

    return [
        x.strip()
        for x in str(value).split("||")
        if x.strip()
    ]


def hierarchy_parts(subject):
    """Fen > Tıp > Cerrahi gibi bir konuyu seviyelerine ayırır."""

    if pd.isna(subject):
        return []

    return [
        x.strip()
        for x in str(subject).split(">")
        if x.strip()
    ]


def common_hierarchy_level(predicted, actual):
    """
    Tahmin ile gerçek konu kaç hiyerarşi seviyesi boyunca aynı?
    """

    p = hierarchy_parts(predicted)
    a = hierarchy_parts(actual)

    common = 0

    for x, y in zip(p, a):
        if x == y:
            common += 1
        else:
            break

    return common


# ============================================================
# MAKALE BAZINDA ANALİZ
# ============================================================

article_rows = []

for article_id, group in df.groupby("article_id"):

    group = group.sort_values("prediction_rank")

    first = group.iloc[0]

    actual_subjects = parse_subjects(
        first["trdizin_subjects"]
    )

    predicted_subjects = (
        group["predicted_subject"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    predicted_subjects = [
        x for x in predicted_subjects
        if x
    ]

    # --------------------------------------------------------
    # EN YÜKSEK SKORLU TAHMİN
    # --------------------------------------------------------

    top_prediction = (
        predicted_subjects[0]
        if predicted_subjects
        else ""
    )

    top_final_score = (
        float(group.iloc[0]["final_score"])
        if len(group) > 0
        and pd.notna(group.iloc[0]["final_score"])
        else np.nan
    )

    top_kmeans_score = (
        float(group.iloc[0]["kmeans_score"])
        if len(group) > 0
        and pd.notna(group.iloc[0]["kmeans_score"])
        else np.nan
    )

    top_semantic_score = (
        float(group.iloc[0]["semantic_score"])
        if len(group) > 0
        and pd.notna(group.iloc[0]["semantic_score"])
        else np.nan
    )

    # --------------------------------------------------------
    # HİYERARŞİ UYUMU
    # --------------------------------------------------------

    max_common_level = 0
    closest_actual_subject = ""

    for predicted in predicted_subjects:

        for actual in actual_subjects:

            common = common_hierarchy_level(
                predicted,
                actual
            )

            if common > max_common_level:

                max_common_level = common
                closest_actual_subject = actual

    # --------------------------------------------------------
    # HATA TÜRÜ
    # --------------------------------------------------------

    # Level:
    # 0 = Fen/Sosyal bile farklı
    # 1 = Fen/Sosyal aynı
    # 2 = ana bölüm aynı
    # 3 = tam konu aynı
    #
    # UYUSMAZLIK verisinde normalde level=3 olmamalı.

    if max_common_level >= 2:

        error_type = "YAKIN_ALT_KONU_HATASI"

    elif max_common_level == 1:

        error_type = "AYNI_UST_ALAN_FARKLI_ALT_ALAN"

    else:

        error_type = "TAMAMEN_FARKLI_ALAN"

    # --------------------------------------------------------
    # SKOR DAVRANIŞI
    # --------------------------------------------------------

    if (
        pd.notna(top_kmeans_score)
        and pd.notna(top_semantic_score)
    ):

        score_difference = (
            top_kmeans_score
            -
            top_semantic_score
        )

        if score_difference >= 0.20:

            score_behavior = "KMEANS_BASKIN"

        elif score_difference <= -0.20:

            score_behavior = "SEMANTIC_BASKIN"

        else:

            score_behavior = "DENGELI"

    else:

        score_difference = np.nan
        score_behavior = "BILINMIYOR"

    # --------------------------------------------------------
    # CLUSTER GÜVENİ
    # --------------------------------------------------------

    best_similarity = first.get(
        "best_similarity",
        np.nan
    )

    if pd.isna(best_similarity):

        cluster_confidence = "BILINMIYOR"

    elif best_similarity >= 0.75:

        cluster_confidence = "YUKSEK"

    elif best_similarity >= 0.60:

        cluster_confidence = "ORTA"

    else:

        cluster_confidence = "DUSUK"

    # --------------------------------------------------------
    # SONUÇ
    # --------------------------------------------------------

    article_rows.append(
        {
            "article_id": article_id,

            "title":
                first["title"],

            "trdizin_subjects":
                " || ".join(actual_subjects),

            "predicted_subjects":
                " || ".join(predicted_subjects),

            "top_prediction":
                top_prediction,

            "closest_actual_subject":
                closest_actual_subject,

            "max_common_hierarchy_level":
                max_common_level,

            "error_type":
                error_type,

            "top_kmeans_score":
                round(top_kmeans_score, 4)
                if pd.notna(top_kmeans_score)
                else np.nan,

            "top_semantic_score":
                round(top_semantic_score, 4)
                if pd.notna(top_semantic_score)
                else np.nan,

            "top_final_score":
                round(top_final_score, 4)
                if pd.notna(top_final_score)
                else np.nan,

            "kmeans_semantic_difference":
                round(score_difference, 4)
                if pd.notna(score_difference)
                else np.nan,

            "score_behavior":
                score_behavior,

            "best_cluster":
                first.get(
                    "best_cluster",
                    np.nan
                ),

            "best_similarity":
                best_similarity,

            "cluster_confidence":
                cluster_confidence,

            "selected_cluster_count":
                first.get(
                    "selected_cluster_count",
                    np.nan
                ),

            "prediction_count":
                len(predicted_subjects)
        }
    )


# ============================================================
# DATAFRAME
# ============================================================

analysis = pd.DataFrame(article_rows)


# ============================================================
# KAYDET
# ============================================================

analysis.to_csv(
    OUTPUT_ARTICLES,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# ÖZET
# ============================================================

print("\n")
print("=" * 110)
print("HATA TÜRÜ DAĞILIMI")
print("=" * 110)

error_summary = (
    analysis["error_type"]
    .value_counts()
)

print(
    error_summary.to_string()
)


print("\n")
print("=" * 110)
print("SKOR DAVRANIŞI")
print("=" * 110)

score_summary = (
    analysis["score_behavior"]
    .value_counts()
)

print(
    score_summary.to_string()
)


print("\n")
print("=" * 110)
print("CLUSTER GÜVENİ")
print("=" * 110)

confidence_summary = (
    analysis["cluster_confidence"]
    .value_counts()
)

print(
    confidence_summary.to_string()
)


# ============================================================
# ORTALAMALAR
# ============================================================

print("\n")
print("=" * 110)
print("ORTALAMA SKORLAR")
print("=" * 110)

print(
    "K-Means:",
    round(
        analysis["top_kmeans_score"].mean(),
        4
    )
)

print(
    "Semantic:",
    round(
        analysis["top_semantic_score"].mean(),
        4
    )
)

print(
    "Final:",
    round(
        analysis["top_final_score"].mean(),
        4
    )
)

print(
    "Best centroid similarity:",
    round(
        analysis["best_similarity"].mean(),
        4
    )
)

print(
    "Ortalama tahmin sayısı:",
    round(
        analysis["prediction_count"].mean(),
        2
    )
)


# ============================================================
# ÇAPRAZ ANALİZ
# ============================================================

print("\n")
print("=" * 110)
print("HATA TÜRÜ x SKOR DAVRANIŞI")
print("=" * 110)

cross = pd.crosstab(
    analysis["error_type"],
    analysis["score_behavior"]
)

print(
    cross.to_string()
)


# ============================================================
# EN YÜKSEK GÜVENLE YAPILAN YANLIŞ TAHMİNLER
# ============================================================

print("\n")
print("=" * 110)
print("EN YÜKSEK FINAL SKORLU 15 UYUŞMAZLIK")
print("=" * 110)

columns = [
    "article_id",
    "top_prediction",
    "closest_actual_subject",
    "error_type",
    "top_kmeans_score",
    "top_semantic_score",
    "top_final_score",
    "best_similarity"
]

print(
    analysis
    .sort_values(
        "top_final_score",
        ascending=False
    )
    [columns]
    .head(15)
    .to_string(index=False)
)


# ============================================================
# ÖZET CSV
# ============================================================

summary_rows = []

for error_type, count in error_summary.items():

    subset = analysis[
        analysis["error_type"]
        ==
        error_type
    ]

    summary_rows.append(
        {
            "error_type":
                error_type,

            "article_count":
                count,

            "percentage":
                round(
                    count
                    /
                    len(analysis)
                    *
                    100,
                    2
                ),

            "avg_kmeans_score":
                round(
                    subset[
                        "top_kmeans_score"
                    ].mean(),
                    4
                ),

            "avg_semantic_score":
                round(
                    subset[
                        "top_semantic_score"
                    ].mean(),
                    4
                ),

            "avg_final_score":
                round(
                    subset[
                        "top_final_score"
                    ].mean(),
                    4
                ),

            "avg_best_similarity":
                round(
                    subset[
                        "best_similarity"
                    ].mean(),
                    4
                )
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    OUTPUT_SUMMARY,
    index=False,
    encoding="utf-8-sig"
)


print("\n")
print("=" * 110)
print("DOSYALAR OLUŞTURULDU")
print("=" * 110)

print(OUTPUT_ARTICLES)
print(OUTPUT_SUMMARY)