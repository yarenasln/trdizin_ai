import pandas as pd

CLUSTER_FILE = "kmeans_article_clusters.csv"
PROFILE_FILE = "kmeans_cluster_topic_profiles.csv"

clusters = pd.read_csv(CLUSTER_FILE, encoding="utf-8-sig")
profiles = pd.read_csv(PROFILE_FILE, encoding="utf-8-sig")


# ============================================================
# HER KÜME İÇİN ÖZET
# ============================================================

rows = []

for cluster_id in sorted(clusters["cluster_id"].unique()):

    cluster_articles = clusters[
        clusters["cluster_id"] == cluster_id
    ]

    cluster_size = len(cluster_articles)

    cluster_profile = profiles[
        profiles["cluster_id"] == cluster_id
    ].sort_values("rank")

    if len(cluster_profile) > 0:

        dominant = cluster_profile.iloc[0]

        dominant_subject = dominant["subject_path"]
        dominant_ratio = dominant["subject_ratio"]
        dominant_count = dominant["article_count"]

        subject_count = len(cluster_profile)

    else:

        dominant_subject = ""
        dominant_ratio = 0
        dominant_count = 0
        subject_count = 0

    rows.append({
        "cluster_id": cluster_id,
        "cluster_size": cluster_size,
        "dominant_subject": dominant_subject,
        "dominant_count": dominant_count,
        "dominant_ratio": dominant_ratio,
        "different_subject_count": subject_count
    })


summary = pd.DataFrame(rows)


# ============================================================
# GENEL İSTATİSTİKLER
# ============================================================

print("\n" + "=" * 90)
print("K-MEANS CLUSTER ANALİZİ")
print("=" * 90)

print("Toplam cluster:", len(summary))

print(
    "En küçük cluster:",
    summary["cluster_size"].min()
)

print(
    "En büyük cluster:",
    summary["cluster_size"].max()
)

print(
    "Ortalama cluster büyüklüğü:",
    round(summary["cluster_size"].mean(), 2)
)

print(
    "Medyan cluster büyüklüğü:",
    summary["cluster_size"].median()
)


# ============================================================
# KÜÇÜK KÜMELER
# ============================================================

print("\n" + "=" * 90)
print("KÜÇÜK CLUSTER SAYILARI")
print("=" * 90)

for limit in [1, 2, 3, 5, 10]:

    count = (
        summary["cluster_size"] <= limit
    ).sum()

    print(
        f"{limit} veya daha az makaleli cluster: {count}"
    )


# ============================================================
# BASKIN KONU GÜCÜ
# ============================================================

print("\n" + "=" * 90)
print("BASKIN KONU ORANLARI")
print("=" * 90)

for threshold in [0.25, 0.50, 0.75, 1.00]:

    count = (
        summary["dominant_ratio"] >= threshold
    ).sum()

    print(
        f"Baskın konu oranı >= {threshold:.2f}: "
        f"{count} cluster"
    )


# ============================================================
# EN TEMİZ 10 KÜME
# ============================================================

print("\n" + "=" * 90)
print("EN YÜKSEK BASKIN KONU ORANINA SAHİP CLUSTERLAR")
print("=" * 90)

best = summary.sort_values(
    ["dominant_ratio", "cluster_size"],
    ascending=[False, False]
).head(10)

print(
    best[
        [
            "cluster_id",
            "cluster_size",
            "dominant_subject",
            "dominant_count",
            "dominant_ratio"
        ]
    ].to_string(index=False)
)


# ============================================================
# EN KARIŞIK 10 KÜME
# ============================================================

print("\n" + "=" * 90)
print("EN KARIŞIK CLUSTERLAR")
print("=" * 90)

mixed = summary[
    summary["cluster_size"] >= 5
].sort_values(
    "dominant_ratio"
).head(10)

print(
    mixed[
        [
            "cluster_id",
            "cluster_size",
            "dominant_subject",
            "dominant_count",
            "dominant_ratio",
            "different_subject_count"
        ]
    ].to_string(index=False)
)


summary.to_csv(
    "kmeans_cluster_analysis.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nDosya oluşturuldu: kmeans_cluster_analysis.csv")