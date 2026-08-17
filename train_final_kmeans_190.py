import os
import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans


# ============================================================
# AYARLAR
# ============================================================

TEXT_FILE = "real_trdizin_texts.csv"
EMBEDDING_FILE = "embeddings/Qwen3_embeddings.npy"

MODEL_DIR = "models"
RESULT_DIR = "results"

K = 190
RANDOM_STATE = 42


# ============================================================
# KLASÖRLER
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# VERİLERİ OKU
# ============================================================

texts = pd.read_csv(
    TEXT_FILE,
    encoding="utf-8-sig"
)

embeddings = np.load(
    EMBEDDING_FILE
).astype(np.float32)


print("=" * 100)
print("FINAL K-MEANS MODELİ")
print("=" * 100)

print("Metin satırı:", len(texts))
print("Embedding shape:", embeddings.shape)


# ============================================================
# KONTROL
# ============================================================

if len(texts) != len(embeddings):
    raise ValueError(
        f"Metin/embedding uyuşmuyor! "
        f"{len(texts)} != {len(embeddings)}"
    )


# ============================================================
# EMBEDDING NORMALIZATION
# ============================================================

norms = np.linalg.norm(
    embeddings,
    axis=1,
    keepdims=True
)

norms[norms == 0] = 1

embeddings = embeddings / norms


# ============================================================
# ARTICLE LEVEL EMBEDDING
# ============================================================
#
# Aynı makalenin TUR + ENG kayıtlarının
# embedding ortalaması alınıyor.
# ============================================================

article_ids = []
article_vectors = []


for article_id, group in texts.groupby("article_id"):

    indices = group.index.to_numpy()

    vectors = embeddings[indices]

    vector = vectors.mean(axis=0)

    # Ortalama sonrası tekrar normalize
    norm = np.linalg.norm(vector)

    if norm > 0:
        vector = vector / norm

    article_ids.append(article_id)
    article_vectors.append(vector)


X = np.vstack(
    article_vectors
).astype(np.float32)


print("Benzersiz makale:", len(article_ids))
print("Article embedding shape:", X.shape)


# ============================================================
# FINAL K-MEANS
# ============================================================

print("\nK-Means eğitiliyor...")
print("K =", K)


model = KMeans(
    n_clusters=K,
    random_state=RANDOM_STATE,
    n_init=10
)


cluster_labels = model.fit_predict(X)


print("K-Means tamamlandı.")


# ============================================================
# CLUSTER MERKEZLERİNİ NORMALIZE ET
# ============================================================

centroids = model.cluster_centers_.astype(
    np.float32
)

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
# HER MAKALENİN KENDİ CENTROID'İNE BENZERLİĞİ
# ============================================================

assigned_centroids = normalized_centroids[
    cluster_labels
]

cluster_similarity = np.sum(
    X * assigned_centroids,
    axis=1
)


# ============================================================
# ARTICLE -> CLUSTER SONUCU
# ============================================================

article_cluster_df = pd.DataFrame({
    "article_id": article_ids,
    "cluster_id": cluster_labels,
    "cluster_similarity": cluster_similarity
})


# ============================================================
# CLUSTER BOYUTLARI
# ============================================================

cluster_sizes = (
    article_cluster_df["cluster_id"]
    .value_counts()
    .sort_index()
    .rename("cluster_size")
    .reset_index()
)


print("\n" + "=" * 100)
print("CLUSTER İSTATİSTİKLERİ")
print("=" * 100)

print("Cluster sayısı:", len(cluster_sizes))
print("En küçük cluster:", cluster_sizes["cluster_size"].min())
print("En büyük cluster:", cluster_sizes["cluster_size"].max())

print(
    "Ortalama cluster büyüklüğü:",
    round(
        cluster_sizes["cluster_size"].mean(),
        2
    )
)

print(
    "Medyan cluster büyüklüğü:",
    round(
        cluster_sizes["cluster_size"].median(),
        2
    )
)

print(
    "Singleton cluster:",
    int(
        (
            cluster_sizes["cluster_size"] == 1
        ).sum()
    )
)

print(
    "5 veya daha az makaleli:",
    int(
        (
            cluster_sizes["cluster_size"] <= 5
        ).sum()
    )
)

print(
    "10 veya daha az makaleli:",
    int(
        (
            cluster_sizes["cluster_size"] <= 10
        ).sum()
    )
)


# ============================================================
# DOSYALARI KAYDET
# ============================================================

joblib.dump(
    model,
    os.path.join(
        MODEL_DIR,
        "final_kmeans_k190.joblib"
    )
)


np.save(
    os.path.join(
        MODEL_DIR,
        "final_kmeans_k190_centroids.npy"
    ),
    normalized_centroids
)


np.save(
    os.path.join(
        MODEL_DIR,
        "final_article_embeddings.npy"
    ),
    X
)


article_cluster_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "final_article_clusters.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


cluster_sizes.to_csv(
    os.path.join(
        RESULT_DIR,
        "final_cluster_sizes.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


pd.DataFrame({
    "article_id": article_ids
}).to_csv(
    os.path.join(
        RESULT_DIR,
        "final_article_embedding_index.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


print("\n" + "=" * 100)
print("DOSYALAR OLUŞTURULDU")
print("=" * 100)

print("models/final_kmeans_k190.joblib")
print("models/final_kmeans_k190_centroids.npy")
print("models/final_article_embeddings.npy")
print("results/final_article_clusters.csv")
print("results/final_cluster_sizes.csv")
print("results/final_article_embedding_index.csv")