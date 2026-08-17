import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


TEXT_FILE = "real_trdizin_texts.csv"
EMBEDDING_FILE = "embeddings/Qwen3_embeddings.npy"
CENTROID_FILE = "kmeans_centroids.npy"

CLUSTER_MARGIN = 0.15
MIN_CLUSTER_SIMILARITY = 0.45

RELATIVE_VALUES = [
    0.75,
    0.80,
    0.85,
    0.90,
    0.95
]


texts = pd.read_csv(
    TEXT_FILE,
    encoding="utf-8-sig"
)

embeddings = np.load(
    EMBEDDING_FILE
).astype(np.float32)

centroids = np.load(
    CENTROID_FILE
).astype(np.float32)


# ============================================================
# MAKALE EMBEDDINGLERİ
# ============================================================

article_vectors = {}

for article_id, group in texts.groupby("article_id"):

    indices = group.index.to_numpy()

    vector = embeddings[
        indices
    ].mean(axis=0)

    norm = np.linalg.norm(vector)

    if norm > 0:
        vector = vector / norm

    article_vectors[
        article_id
    ] = vector


# ============================================================
# CENTROID NORMALIZE
# ============================================================

norms = np.linalg.norm(
    centroids,
    axis=1,
    keepdims=True
)

norms[norms == 0] = 1

centroids = centroids / norms


# ============================================================
# DAĞILIM ANALİZİ
# ============================================================

results = []


for relative_similarity in RELATIVE_VALUES:

    counts = []

    for article_id, vector in article_vectors.items():

        similarities = cosine_similarity(
            vector.reshape(1, -1),
            centroids
        )[0]

        best_similarity = float(
            similarities.max()
        )

        relative_limit = (
            best_similarity
            *
            relative_similarity
        )

        selected = np.where(

            (
                similarities
                >=
                best_similarity - CLUSTER_MARGIN
            )

            &

            (
                similarities
                >=
                MIN_CLUSTER_SIMILARITY
            )

            &

            (
                similarities
                >=
                relative_limit
            )

        )[0]

        counts.append(
            len(selected)
        )


    counts = np.array(counts)


    results.append({

        "Relative_Similarity":
            relative_similarity,

        "Mean":
            round(
                counts.mean(),
                2
            ),

        "Median":
            round(
                np.median(counts),
                2
            ),

        "P90":
            round(
                np.percentile(counts, 90),
                2
            ),

        "P95":
            round(
                np.percentile(counts, 95),
                2
            ),

        "P99":
            round(
                np.percentile(counts, 99),
                2
            ),

        "Max":
            int(
                counts.max()
            ),

        "More_Than_5":
            int(
                np.sum(counts > 5)
            ),

        "More_Than_10":
            int(
                np.sum(counts > 10)
            )
    })


result = pd.DataFrame(results)


print("\n" + "=" * 110)
print("SEÇİLEN CLUSTER SAYISI DAĞILIMI")
print("=" * 110)

print(
    result.to_string(
        index=False
    )
)


result.to_csv(
    "selected_cluster_distribution.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nDosya oluşturuldu:"
    " selected_cluster_distribution.csv"
)