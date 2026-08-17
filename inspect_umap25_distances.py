import numpy as np
from sklearn.metrics import pairwise_distances


# ============================================================
# DOSYALAR
# ============================================================

UMAP_FILE = "umap25_qwen3_embeddings.npy"
CENTROID_FILE = "umap25_kmeans_centroids.npy"


# ============================================================
# VERİLERİ YÜKLE
# ============================================================

X = np.load(UMAP_FILE).astype(np.float32)
centroids = np.load(CENTROID_FILE).astype(np.float32)

print("=" * 105)
print("UMAP25 + K-MEANS EUCLIDEAN CENTROID DISTANCE ANALİZİ")
print("=" * 105)

print("Makale embedding:", X.shape)
print("Centroid:", centroids.shape)


# ============================================================
# TÜM MAKALE - CENTROID UZAKLIKLARI
# ============================================================

distances = pairwise_distances(
    X,
    centroids,
    metric="euclidean"
)

print("Distance matrisi:", distances.shape)


# ============================================================
# EN YAKIN CENTROIDLER
# ============================================================

sorted_distances = np.sort(
    distances,
    axis=1
)

best = sorted_distances[:, 0]
second = sorted_distances[:, 1]
third = sorted_distances[:, 2]


# ============================================================
# ABSOLUTE MARGIN
#
# ikinci en yakın ile en yakın centroid arasındaki uzaklık farkı
# Büyük olması -> birinci cluster daha belirgin
# ============================================================

margin = second - best


# ============================================================
# RELATIVE DISTANCE
#
# second / best
#
# 1'e yakın -> iki cluster birbirine yakın
# büyük -> birinci cluster daha belirgin
# ============================================================

eps = 1e-8

relative_second = (
    second / (best + eps)
)

relative_third = (
    third / (best + eps)
)


# ============================================================
# GENEL İSTATİSTİKLER
# ============================================================

print("\n" + "=" * 105)
print("EN YAKIN CENTROID UZAKLIĞI")
print("=" * 105)

print(
    "Ortalama:",
    round(float(np.mean(best)), 6)
)

print(
    "Medyan:",
    round(float(np.median(best)), 6)
)

print(
    "Minimum:",
    round(float(np.min(best)), 6)
)

print(
    "Maksimum:",
    round(float(np.max(best)), 6)
)


print("\n" + "=" * 105)
print("İKİNCİ EN YAKIN CENTROID UZAKLIĞI")
print("=" * 105)

print(
    "Ortalama:",
    round(float(np.mean(second)), 6)
)

print(
    "Medyan:",
    round(float(np.median(second)), 6)
)


print("\n" + "=" * 105)
print("BEST -> SECOND DISTANCE MARGIN")
print("=" * 105)

print(
    "Ortalama:",
    round(float(np.mean(margin)), 6)
)

print(
    "Medyan:",
    round(float(np.median(margin)), 6)
)


# ============================================================
# PERCENTILE
# ============================================================

print("\nMargin percentilleri:")

for p in [
    10,
    25,
    50,
    75,
    90,
    95,
    99
]:

    value = np.percentile(
        margin,
        p
    )

    print(
        f"P{p}:",
        round(float(value), 6)
    )


# ============================================================
# RELATIVE SECOND DISTANCE
# ============================================================

print("\n" + "=" * 105)
print("SECOND / BEST ORANI")
print("=" * 105)

print(
    "Ortalama:",
    round(
        float(
            np.mean(relative_second)
        ),
        4
    )
)

print(
    "Medyan:",
    round(
        float(
            np.median(relative_second)
        ),
        4
    )
)


print("\nSecond/Best percentilleri:")

for p in [
    10,
    25,
    50,
    75,
    90,
    95,
    99
]:

    value = np.percentile(
        relative_second,
        p
    )

    print(
        f"P{p}:",
        round(float(value), 4)
    )


# ============================================================
# BELİRLİ ORANLARDA KAÇ CLUSTER SEÇİLİYOR?
#
# Örneğin ratio=1.10:
# distance <= best_distance * 1.10
# olan clusterlar seçilir.
# ============================================================

print("\n" + "=" * 105)
print("RELATIVE DISTANCE İLE SEÇİLEN ORTALAMA CLUSTER SAYISI")
print("=" * 105)


ratios = [
    1.02,
    1.05,
    1.10,
    1.15,
    1.20,
    1.25,
    1.30,
    1.40,
    1.50
]


for ratio in ratios:

    selected_counts = []

    for i in range(len(X)):

        best_distance = best[i]

        limit = (
            best_distance
            *
            ratio
        )

        count = np.sum(
            distances[i] <= limit
        )

        selected_counts.append(
            count
        )


    selected_counts = np.array(
        selected_counts
    )


    print(
        f"Ratio <= {ratio:.2f} | "
        f"Mean={np.mean(selected_counts):.2f} | "
        f"Median={np.median(selected_counts):.0f} | "
        f"P95={np.percentile(selected_counts, 95):.0f} | "
        f"Max={np.max(selected_counts)}"
    )


# ============================================================
# ABSOLUTE DISTANCE MARGIN İLE DE BAK
#
# distance <= best_distance + margin
# ============================================================

print("\n" + "=" * 105)
print("ABSOLUTE DISTANCE MARGIN İLE SEÇİLEN CLUSTER SAYISI")
print("=" * 105)


absolute_margins = [
    0.05,
    0.10,
    0.20,
    0.30,
    0.50,
    0.75,
    1.00
]


for allowed_margin in absolute_margins:

    selected_counts = []

    for i in range(len(X)):

        limit = (
            best[i]
            +
            allowed_margin
        )

        count = np.sum(
            distances[i] <= limit
        )

        selected_counts.append(
            count
        )


    selected_counts = np.array(
        selected_counts
    )


    print(
        f"Margin <= +{allowed_margin:.2f} | "
        f"Mean={np.mean(selected_counts):.2f} | "
        f"Median={np.median(selected_counts):.0f} | "
        f"P95={np.percentile(selected_counts, 95):.0f} | "
        f"Max={np.max(selected_counts)}"
    )


# ============================================================
# EN BELİRGİN / EN BELİRSİZ ÖRNEKLER
# ============================================================

print("\n" + "=" * 105)
print("EN BELİRGİN 10 MAKALE")
print("=" * 105)


most_clear = np.argsort(
    relative_second
)[::-1][:10]


for idx in most_clear:

    print(
        f"Index={idx} | "
        f"Best={best[idx]:.4f} | "
        f"Second={second[idx]:.4f} | "
        f"Margin={margin[idx]:.4f} | "
        f"Second/Best={relative_second[idx]:.4f}"
    )


print("\n" + "=" * 105)
print("İKİ CLUSTER ARASINDA EN ÇOK KALAN 10 MAKALE")
print("=" * 105)


most_ambiguous = np.argsort(
    relative_second
)[:10]


for idx in most_ambiguous:

    print(
        f"Index={idx} | "
        f"Best={best[idx]:.4f} | "
        f"Second={second[idx]:.4f} | "
        f"Margin={margin[idx]:.4f} | "
        f"Second/Best={relative_second[idx]:.4f}"
    )