import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


TEXT_FILE = "real_trdizin_texts.csv"
UMAP_FILE = "umap25_qwen3_embeddings.npy"
CENTROID_FILE = "umap25_kmeans_centroids.npy"


texts = pd.read_csv(
    TEXT_FILE,
    encoding="utf-8-sig"
)

X = np.load(
    UMAP_FILE
).astype(np.float32)

centroids = np.load(
    CENTROID_FILE
).astype(np.float32)


# ============================================================
# NORMALIZE
# ============================================================

X_norm = np.linalg.norm(
    X,
    axis=1,
    keepdims=True
)

X_norm[X_norm == 0] = 1

X = X / X_norm


centroid_norm = np.linalg.norm(
    centroids,
    axis=1,
    keepdims=True
)

centroid_norm[
    centroid_norm == 0
] = 1

centroids = (
    centroids / centroid_norm
)


# ============================================================
# HER MAKALE İÇİN EN İYİ / İKİNCİ / DAĞILIM
# ============================================================

best_values = []
second_values = []
margins = []

count_090 = []
count_095 = []
count_097 = []
count_098 = []
count_099 = []


for vector in X:

    similarities = cosine_similarity(
        vector.reshape(1, -1),
        centroids
    )[0]

    ordered = np.sort(
        similarities
    )[::-1]

    best = ordered[0]
    second = ordered[1]

    best_values.append(
        best
    )

    second_values.append(
        second
    )

    margins.append(
        best - second
    )

    count_090.append(
        np.sum(similarities >= 0.90)
    )

    count_095.append(
        np.sum(similarities >= 0.95)
    )

    count_097.append(
        np.sum(similarities >= 0.97)
    )

    count_098.append(
        np.sum(similarities >= 0.98)
    )

    count_099.append(
        np.sum(similarities >= 0.99)
    )


# ============================================================
# SONUÇ
# ============================================================

print("=" * 100)
print("UMAP25 CENTROID SIMILARITY DAĞILIMI")
print("=" * 100)

print(
    "Ortalama best similarity:",
    round(
        np.mean(best_values),
        4
    )
)

print(
    "Medyan best similarity:",
    round(
        np.median(best_values),
        4
    )
)

print(
    "Ortalama second similarity:",
    round(
        np.mean(second_values),
        4
    )
)

print(
    "Ortalama best-second margin:",
    round(
        np.mean(margins),
        6
    )
)

print(
    "Medyan best-second margin:",
    round(
        np.median(margins),
        6
    )
)


print("\nBir makalenin ortalama kaç centroid'i geçiyor?")

print(
    "Similarity >= 0.90:",
    round(
        np.mean(count_090),
        2
    )
)

print(
    "Similarity >= 0.95:",
    round(
        np.mean(count_095),
        2
    )
)

print(
    "Similarity >= 0.97:",
    round(
        np.mean(count_097),
        2
    )
)

print(
    "Similarity >= 0.98:",
    round(
        np.mean(count_098),
        2
    )
)

print(
    "Similarity >= 0.99:",
    round(
        np.mean(count_099),
        2
    )
)


print("\nMargin percentilleri:")

for percentile in [
    10,
    25,
    50,
    75,
    90,
    95,
    99
]:

    print(
        f"P{percentile}:",
        round(
            np.percentile(
                margins,
                percentile
            ),
            6
        )
    )