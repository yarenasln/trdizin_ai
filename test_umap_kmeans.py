import numpy as np
import pandas as pd
import umap.umap_ as umap

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# DOSYALAR
# ============================================================

TEXT_FILE = "real_trdizin_texts.csv"
EMBEDDING_FILE = "embeddings/Qwen3_embeddings.npy"
SUBJECT_FILE = "trdizin_subject_hierarchy.csv"


# ============================================================
# AYARLAR
# ============================================================

K = 195

UMAP_COMPONENTS = 50
UMAP_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1

RANDOM_STATE = 42


# ============================================================
# VERİLERİ OKU
# ============================================================

texts = pd.read_csv(
    TEXT_FILE,
    encoding="utf-8-sig"
)

subjects = pd.read_csv(
    SUBJECT_FILE,
    encoding="utf-8-sig"
)

embeddings = np.load(
    EMBEDDING_FILE
).astype(np.float32)


print("=" * 100)
print("QWEN3 + UMAP + K-MEANS TESTİ")
print("=" * 100)

print("Metin satırı:", len(texts))
print("Orijinal embedding:", embeddings.shape)


# ============================================================
# SADECE EN-ALT KONULAR
# ============================================================

subjects["leaf_subject"] = (
    subjects["leaf_subject"]
    .fillna("")
    .astype(str)
    .str.strip()
)

subjects["subject_fullname"] = (
    subjects["subject_fullname"]
    .fillna("")
    .astype(str)
    .str.strip()
)

subjects = subjects[
    (subjects["leaf_subject"] != "")
    &
    (subjects["subject_fullname"] != "")
].copy()


article_subjects = (
    subjects
    .groupby("article_id")["subject_fullname"]
    .apply(set)
    .to_dict()
)


# ============================================================
# ROW LEVEL -> ARTICLE LEVEL
# ============================================================
#
# Aynı makalenin birden fazla satırı varsa
# embedding ortalamasını alıyoruz.
# ============================================================

article_ids = []
article_vectors = []


for article_id, group in texts.groupby("article_id"):

    indices = group.index.to_numpy()

    vectors = embeddings[
        indices
    ]

    mean_vector = vectors.mean(
        axis=0
    )

    norm = np.linalg.norm(
        mean_vector
    )

    if norm > 0:
        mean_vector = (
            mean_vector / norm
        )

    article_ids.append(
        article_id
    )

    article_vectors.append(
        mean_vector
    )


X = np.vstack(
    article_vectors
).astype(np.float32)

article_ids = np.array(
    article_ids
)


print(
    "Makale sayısı:",
    len(article_ids)
)

print(
    "Makale embedding shape:",
    X.shape
)


# ============================================================
# UMAP
# ============================================================

print("\nUMAP çalışıyor...")


reducer = umap.UMAP(

    n_components=UMAP_COMPONENTS,

    n_neighbors=UMAP_NEIGHBORS,

    min_dist=UMAP_MIN_DIST,

    metric="cosine",

    random_state=RANDOM_STATE
)


X_umap = reducer.fit_transform(
    X
)


print(
    "UMAP sonrası shape:",
    X_umap.shape
)


# ============================================================
# K-MEANS
# ============================================================

print("\nK-Means çalışıyor...")


kmeans = KMeans(

    n_clusters=K,

    init="k-means++",

    n_init=10,

    max_iter=300,

    random_state=RANDOM_STATE
)


cluster_labels = kmeans.fit_predict(
    X_umap
)


# ============================================================
# 1. UMAP UZAYINDA SILHOUETTE
# ============================================================
#
# Bu skor sadece UMAP sonrası 50 boyutlu uzayda
# kümelerin ne kadar ayrıştığını gösterir.
# ============================================================

umap_silhouette = silhouette_score(

    X_umap,

    cluster_labels,

    metric="euclidean"
)


# ============================================================
# 2. ORİJİNAL 1024 BOYUTTA SILHOUETTE
# ============================================================
#
# Asıl adil karşılaştırma burada.
#
# UMAP ile oluşturulan cluster etiketlerini alıyoruz
# ama Silhouette'i tekrar orijinal Qwen3
# 1024 boyutlu uzayında cosine ile ölçüyoruz.
# ============================================================

original_space_silhouette = silhouette_score(

    X,

    cluster_labels,

    metric="cosine"
)


# ============================================================
# MULTI-LABEL WEIGHTED PURITY
# ============================================================

total_correct = 0
singleton_clusters = 0


for cluster_id in np.unique(
    cluster_labels
):

    indices = np.where(
        cluster_labels == cluster_id
    )[0]

    cluster_article_ids = [
        article_ids[i]
        for i in indices
    ]


    if len(
        cluster_article_ids
    ) == 1:

        singleton_clusters += 1


    subject_counts = {}


    for article_id in cluster_article_ids:

        labels = article_subjects.get(
            article_id,
            set()
        )


        for label in labels:

            subject_counts[
                label
            ] = (
                subject_counts.get(
                    label,
                    0
                )
                + 1
            )


    if not subject_counts:
        continue


    dominant_count = max(
        subject_counts.values()
    )

    total_correct += (
        dominant_count
    )


weighted_purity = (
    total_correct
    /
    len(article_ids)
)


# ============================================================
# KÜME BOYUTLARI
# ============================================================

_, counts = np.unique(

    cluster_labels,

    return_counts=True
)


smallest_cluster = int(
    counts.min()
)

largest_cluster = int(
    counts.max()
)


# ============================================================
# GENEL KÜME İSTATİSTİKLERİ
# ============================================================

average_cluster_size = float(
    counts.mean()
)

median_cluster_size = float(
    np.median(counts)
)

clusters_le_5 = int(
    np.sum(
        counts <= 5
    )
)

clusters_le_10 = int(
    np.sum(
        counts <= 10
    )
)


# ============================================================
# SONUÇ
# ============================================================

print("\n" + "=" * 100)
print("UMAP + K-MEANS SONUCU")
print("=" * 100)

print(
    "Embedding:",
    "Qwen3"
)

print(
    "Orijinal boyut:",
    1024
)

print(
    "UMAP boyutu:",
    UMAP_COMPONENTS
)

print(
    "K:",
    K
)


print("\n--- SILHOUETTE ---")

print(
    "UMAP uzayında Euclidean Silhouette:",
    round(
        umap_silhouette,
        4
    )
)

print(
    "Orijinal 1024 boyutta Cosine Silhouette:",
    round(
        original_space_silhouette,
        4
    )
)


print("\n--- TR DİZİN KONU UYUMU ---")

print(
    "Weighted Purity:",
    round(
        weighted_purity,
        4
    )
)


print("\n--- CLUSTER DAĞILIMI ---")

print(
    "Singleton:",
    singleton_clusters
)

print(
    "En küçük cluster:",
    smallest_cluster
)

print(
    "En büyük cluster:",
    largest_cluster
)

print(
    "Ortalama cluster büyüklüğü:",
    round(
        average_cluster_size,
        2
    )
)

print(
    "Medyan cluster büyüklüğü:",
    round(
        median_cluster_size,
        2
    )
)

print(
    "5 veya daha az makaleli cluster:",
    clusters_le_5
)

print(
    "10 veya daha az makaleli cluster:",
    clusters_le_10
)


# ============================================================
# MEVCUT BASELINE İLE KARŞILAŞTIRMA
# ============================================================
#
# Bunlar daha önce elde ettiğimiz
# Qwen3 + K-Means, K=195 pilot sonuçları.
# ============================================================

BASELINE_SILHOUETTE = 0.0385
BASELINE_PURITY = 0.5617
BASELINE_SINGLETON = 2
BASELINE_LARGEST_CLUSTER = 26


print("\n" + "=" * 100)
print("MEVCUT 1024 BOYUTLU K-MEANS İLE KARŞILAŞTIRMA")
print("=" * 100)


print(
    f"Cosine Silhouette:"
    f" {BASELINE_SILHOUETTE:.4f}"
    f" -> {original_space_silhouette:.4f}"
)

print(
    f"Weighted Purity:"
    f" {BASELINE_PURITY:.4f}"
    f" -> {weighted_purity:.4f}"
)

print(
    f"Singleton:"
    f" {BASELINE_SINGLETON}"
    f" -> {singleton_clusters}"
)

print(
    f"En büyük cluster:"
    f" {BASELINE_LARGEST_CLUSTER}"
    f" -> {largest_cluster}"
)


# ============================================================
# DOSYALARI KAYDET
# ============================================================

np.save(
    "umap_qwen3_embeddings.npy",
    X_umap
)

np.save(
    "umap_kmeans_centroids.npy",
    kmeans.cluster_centers_
)


pd.DataFrame({

    "article_id":
        article_ids,

    "cluster_id":
        cluster_labels

}).to_csv(

    "umap_kmeans_article_clusters.csv",

    index=False,

    encoding="utf-8-sig"
)


# ============================================================
# SONUÇ ÖZETİNİ CSV'YE KAYDET
# ============================================================

summary = pd.DataFrame([
    {
        "Model":
            "Qwen3",

        "Original_Dimension":
            1024,

        "UMAP_Dimension":
            UMAP_COMPONENTS,

        "K":
            K,

        "UMAP_Euclidean_Silhouette":
            round(
                umap_silhouette,
                4
            ),

        "Original_Cosine_Silhouette":
            round(
                original_space_silhouette,
                4
            ),

        "Weighted_Purity":
            round(
                weighted_purity,
                4
            ),

        "Singleton":
            singleton_clusters,

        "Smallest_Cluster":
            smallest_cluster,

        "Largest_Cluster":
            largest_cluster,

        "Average_Cluster_Size":
            round(
                average_cluster_size,
                2
            ),

        "Median_Cluster_Size":
            round(
                median_cluster_size,
                2
            ),

        "Clusters_LE_5":
            clusters_le_5,

        "Clusters_LE_10":
            clusters_le_10
    }
])


summary.to_csv(
    "umap_kmeans_summary.csv",
    index=False,
    encoding="utf-8-sig"
)


print("\nDosyalar oluşturuldu:")

print(
    "umap_qwen3_embeddings.npy"
)

print(
    "umap_kmeans_centroids.npy"
)

print(
    "umap_kmeans_article_clusters.csv"
)

print(
    "umap_kmeans_summary.csv"
)