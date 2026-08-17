import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# DOSYALAR
# ============================================================

TEXT_FILE = "real_trdizin_texts.csv"
EMBEDDING_FILE = "embeddings/Qwen3_embeddings.npy"

CLUSTER_FILE = "kmeans_article_clusters.csv"
CENTROID_FILE = "kmeans_centroids.npy"


# ============================================================
# VERİLER
# ============================================================

texts = pd.read_csv(
    TEXT_FILE,
    encoding="utf-8-sig"
)

clusters = pd.read_csv(
    CLUSTER_FILE,
    encoding="utf-8-sig"
)

embeddings = np.load(
    EMBEDDING_FILE
).astype(np.float32)

centroids = np.load(
    CENTROID_FILE
).astype(np.float32)


print("Metin satırı:", len(texts))
print("Embedding:", embeddings.shape)
print("Centroid:", centroids.shape)


# ============================================================
# ROW-LEVEL -> ARTICLE-LEVEL EMBEDDING
# ============================================================

article_vectors = {}

for article_id, group in texts.groupby("article_id"):

    indices = group.index.to_numpy()

    vectors = embeddings[indices]

    mean_vector = vectors.mean(axis=0)

    norm = np.linalg.norm(mean_vector)

    if norm > 0:
        mean_vector = mean_vector / norm

    article_vectors[article_id] = mean_vector


print(
    "Makale embedding sayısı:",
    len(article_vectors)
)


# ============================================================
# CENTROIDLERİ NORMALIZE ET
# ============================================================

centroid_norms = np.linalg.norm(
    centroids,
    axis=1,
    keepdims=True
)

centroid_norms[
    centroid_norms == 0
] = 1

normalized_centroids = (
    centroids / centroid_norms
)


# ============================================================
# HER MAKALE İÇİN CENTROID YAKINLIĞI
# ============================================================

rows = []

for _, row in clusters.iterrows():

    article_id = row["article_id"]
    assigned_cluster = int(
        row["cluster_id"]
    )

    vector = article_vectors.get(
        article_id
    )

    if vector is None:
        continue

    # Tüm centroidlerle cosine similarity
    similarities = cosine_similarity(
        vector.reshape(1, -1),
        normalized_centroids
    )[0]

    # Büyükten küçüğe sırala
    order = np.argsort(
        similarities
    )[::-1]

    best_cluster = int(
        order[0]
    )

    second_cluster = int(
        order[1]
    )

    best_similarity = float(
        similarities[best_cluster]
    )

    second_similarity = float(
        similarities[second_cluster]
    )

    assigned_similarity = float(
        similarities[assigned_cluster]
    )

    # --------------------------------------------
    # MARGIN
    #
    # En iyi centroid ile ikinci centroid
    # arasındaki fark.
    #
    # Büyükse:
    # makalenin kümesi daha belirgin.
    #
    # Küçükse:
    # makale iki küme arasında kalmış olabilir.
    # --------------------------------------------

    margin = (
        best_similarity
        - second_similarity
    )

    rows.append({

        "article_id":
            article_id,

        "assigned_cluster":
            assigned_cluster,

        "cosine_best_cluster":
            best_cluster,

        "assigned_similarity":
            round(
                assigned_similarity,
                6
            ),

        "best_similarity":
            round(
                best_similarity,
                6
            ),

        "second_best_cluster":
            second_cluster,

        "second_similarity":
            round(
                second_similarity,
                6
            ),

        "centroid_margin":
            round(
                margin,
                6
            ),

        "assigned_is_cosine_best":
            assigned_cluster
            == best_cluster
    })


result = pd.DataFrame(
    rows
)


# ============================================================
# GENEL SONUÇ
# ============================================================

print("\n" + "=" * 100)
print("CENTROID YAKINLIK ANALİZİ")
print("=" * 100)

print(
    "Analiz edilen makale:",
    len(result)
)

print(
    "Ortalama assigned similarity:",
    round(
        result[
            "assigned_similarity"
        ].mean(),
        4
    )
)

print(
    "Medyan assigned similarity:",
    round(
        result[
            "assigned_similarity"
        ].median(),
        4
    )
)

print(
    "Ortalama centroid margin:",
    round(
        result[
            "centroid_margin"
        ].mean(),
        4
    )
)

print(
    "Medyan centroid margin:",
    round(
        result[
            "centroid_margin"
        ].median(),
        4
    )
)

same_ratio = (
    result[
        "assigned_is_cosine_best"
    ].mean()
)

print(
    "K-Means cluster = cosine'a göre "
    "en yakın centroid oranı:",
    round(
        same_ratio,
        4
    )
)


# ============================================================
# MARGIN DAĞILIMI
# ============================================================

print("\n" + "=" * 100)
print("CENTROID MARGIN DAĞILIMI")
print("=" * 100)

for threshold in [
    0.00,
    0.01,
    0.02,
    0.05,
    0.10
]:

    count = (
        result[
            "centroid_margin"
        ] >= threshold
    ).sum()

    ratio = (
        count / len(result)
        if len(result)
        else 0
    )

    print(
        f"Margin >= {threshold:.2f}: "
        f"{count} makale "
        f"({ratio:.2%})"
    )


# ============================================================
# EN BELİRGİN 10 MAKALE
# ============================================================

print("\n" + "=" * 100)
print("KÜMESİ EN BELİRGİN 10 MAKALE")
print("=" * 100)

print(
    result.sort_values(
        "centroid_margin",
        ascending=False
    ).head(10).to_string(
        index=False
    )
)


# ============================================================
# EN BELİRSİZ 10 MAKALE
# ============================================================

print("\n" + "=" * 100)
print("İKİ KÜME ARASINDA EN ÇOK KALAN 10 MAKALE")
print("=" * 100)

print(
    result.sort_values(
        "centroid_margin",
        ascending=True
    ).head(10).to_string(
        index=False
    )
)


# ============================================================
# KAYDET
# ============================================================

result.to_csv(
    "kmeans_centroid_confidence.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nDosya oluşturuldu:"
    " kmeans_centroid_confidence.csv"
)