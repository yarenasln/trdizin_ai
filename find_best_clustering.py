import pandas as pd

# ============================================================
# 1. SONUÇLARI OKU
# ============================================================

df = pd.read_csv(
    "cluster_subject_alignment_results.csv",
    encoding="utf-8-sig"
)

print("=" * 110)
print("KÜMELEME SONUÇLARININ GENEL ANALİZİ")
print("=" * 110)


# ============================================================
# 2. HER MODEL + ALGORİTMA İÇİN SONUÇLARI GÖSTER
# ============================================================

for (model, algorithm), group in df.groupby(
    ["Model", "Algorithm"]
):

    print("\n")
    print("=" * 80)
    print(f"{model} + {algorithm}")
    print("=" * 80)

    group = group.sort_values("K")

    print(
        group[
            [
                "K",
                "Silhouette",
                "Weighted_Purity",
                "Singleton_Clusters",
                "Singleton_Ratio",
                "Largest_Cluster"
            ]
        ].to_string(index=False)
    )


# ============================================================
# 3. NORMALİZE EDİLMİŞ DENGE PUANI
# ============================================================
#
# Amaç:
#
# Silhouette yüksek olsun        +
# Purity yüksek olsun            +
# Singleton oranı düşük olsun    -
#
# Bu "resmi" bir clustering metriği değildir.
# Sadece deneyler arasında karar desteği sağlar.
# ============================================================

def normalize(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(
            [1.0] * len(series),
            index=series.index
        )

    return (
        (series - minimum)
        / (maximum - minimum)
    )


df["Silhouette_Normalized"] = normalize(
    df["Silhouette"]
)

df["Purity_Normalized"] = normalize(
    df["Weighted_Purity"]
)

df["Singleton_Normalized"] = normalize(
    df["Singleton_Ratio"]
)


# ============================================================
# DENGE PUANI
#
# Purity bizim için daha önemli olduğu için %50
# Silhouette %30
# Singleton cezası %20
# ============================================================

df["Balance_Score"] = (

    0.50 * df["Purity_Normalized"]

    +

    0.30 * df["Silhouette_Normalized"]

    -

    0.20 * df["Singleton_Normalized"]
)


# ============================================================
# 4. EN İYİ 15 SONUÇ
# ============================================================

best = df.sort_values(
    "Balance_Score",
    ascending=False
).head(15)


print("\n\n")
print("=" * 110)
print("EN İYİ 15 DENGELİ SONUÇ")
print("=" * 110)

print(
    best[
        [
            "Model",
            "Algorithm",
            "K",
            "Silhouette",
            "Weighted_Purity",
            "Singleton_Clusters",
            "Largest_Cluster",
            "Balance_Score"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 5. HER MODEL + ALGORİTMA İÇİN EN İYİ ADAY
# ============================================================

best_per_method = (
    df
    .sort_values(
        "Balance_Score",
        ascending=False
    )
    .groupby(
        ["Model", "Algorithm"],
        as_index=False
    )
    .first()
)


print("\n\n")
print("=" * 110)

print(
    "HER MODEL + ALGORİTMA İÇİN "
    "EN İYİ DENGELİ K"
)

print("=" * 110)

print(
    best_per_method[
        [
            "Model",
            "Algorithm",
            "K",
            "Silhouette",
            "Weighted_Purity",
            "Singleton_Clusters",
            "Largest_Cluster",
            "Balance_Score"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 6. DOSYAYA KAYDET
# ============================================================

df.to_csv(
    "clustering_balanced_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nDosya oluşturuldu: "
    "clustering_balanced_results.csv"
)