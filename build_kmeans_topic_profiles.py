import numpy as np
import pandas as pd

from sklearn.cluster import KMeans


# ============================================================
# 1. AYARLAR
# ============================================================

TEXT_FILE = "real_trdizin_texts.csv"
SUBJECT_FILE = "trdizin_subject_hierarchy.csv"

# Şimdilik clustering tarafında güçlü adayımız Qwen3.
EMBEDDING_FILE = "embeddings/Qwen3_embeddings.npy"

# İlk referans olarak gerçek en-alt konu sayısına yakın K.
# Bunu daha sonra yeniden optimize edebiliriz.
K = 195

RANDOM_STATE = 42


# ============================================================
# 2. VERİLERİ OKU
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


print("Metin satırı:", len(texts))
print("Embedding shape:", embeddings.shape)


# ============================================================
# 3. SADECE GERÇEK EN-ALT KONULAR
# ============================================================

subjects["leaf_subject"] = (
    subjects["leaf_subject"]
    .fillna("")
    .astype(str)
    .str.strip()
)

leaf_subjects = subjects[
    subjects["leaf_subject"] != ""
].copy()


# ============================================================
# 4. MAKALE -> GERÇEK KONU YOLLARI
# ============================================================

article_labels = (
    leaf_subjects
    .groupby("article_id")["subject_fullname"]
    .apply(
        lambda values: set(
            values.dropna().astype(str)
        )
    )
    .to_dict()
)


# ============================================================
# 5. ROW-LEVEL EMBEDDING -> ARTICLE-LEVEL EMBEDDING
# ============================================================
#
# TUR ve ENG aynı makalenin iki metniyse
# ortalamalarını alıyoruz.
#

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
    "Makale seviyesinde kullanılan veri:",
    len(article_ids)
)


# ============================================================
# 6. K-MEANS
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
    X
)


print(
    "Oluşan cluster (küme):",
    len(np.unique(cluster_labels))
)


# ============================================================
# 7. MAKALE + CLUSTER TABLOSU
# ============================================================

article_cluster_df = pd.DataFrame({
    "article_id": article_ids,
    "cluster_id": cluster_labels
})


# ============================================================
# 8. HER CLUSTER İÇİN KONU PROFİLİ
# ============================================================
#
# Örnek:
#
# Cluster 20
#
# Yapay Zeka        -> 12 makale
# Bilgi Sistemleri  -> 5 makale
# Robotik           -> 3 makale
#
# Burada oranı:
#
# konuya sahip makale sayısı / cluster makale sayısı
#
# olarak hesaplıyoruz.
#
# Multi-label olduğu için oranların toplamı
# %100 olmak zorunda değil.
#

profile_rows = []


for cluster_id in sorted(
    article_cluster_df["cluster_id"].unique()
):

    cluster_articles = (
        article_cluster_df[
            article_cluster_df["cluster_id"]
            == cluster_id
        ]["article_id"]
        .tolist()
    )

    cluster_size = len(
        cluster_articles
    )

    subject_counts = {}


    for article_id in cluster_articles:

        labels = article_labels.get(
            article_id,
            set()
        )

        for label in labels:

            subject_counts[label] = (
                subject_counts.get(
                    label,
                    0
                )
                + 1
            )


    # --------------------------------------------
    # Konuları en çok görülenden aza sırala
    # --------------------------------------------

    sorted_subjects = sorted(
        subject_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )


    for rank, (
        subject_path,
        count
    ) in enumerate(
        sorted_subjects,
        start=1
    ):

        ratio = (
            count / cluster_size
            if cluster_size > 0
            else 0
        )

        profile_rows.append({

            "cluster_id":
                cluster_id,

            "cluster_size":
                cluster_size,

            "rank":
                rank,

            "subject_path":
                subject_path,

            "article_count":
                count,

            "subject_ratio":
                round(
                    ratio,
                    4
                )
        })


topic_profiles = pd.DataFrame(
    profile_rows
)


# ============================================================
# 9. HER CLUSTER'IN İLK KONULARINI GÖSTER
# ============================================================

print("\n")
print("=" * 90)
print("ÖRNEK CLUSTER KONU PROFİLLERİ")
print("=" * 90)


for cluster_id in sorted(
    topic_profiles["cluster_id"]
    .unique()
)[:10]:

    print(
        f"\nCLUSTER {cluster_id}"
    )

    sample = (
        topic_profiles[
            topic_profiles["cluster_id"]
            == cluster_id
        ]
        .head(5)
    )

    if sample.empty:

        print(
            "Konu etiketi bulunan makale yok."
        )

    else:

        for _, row in sample.iterrows():

            print(
                f"{row['rank']}. "
                f"{row['subject_path']} "
                f"| {row['article_count']}/"
                f"{row['cluster_size']} "
                f"| oran={row['subject_ratio']:.4f}"
            )


# ============================================================
# 10. DOSYALARI KAYDET
# ============================================================

article_cluster_df.to_csv(
    "kmeans_article_clusters.csv",
    index=False,
    encoding="utf-8-sig"
)

topic_profiles.to_csv(
    "kmeans_cluster_topic_profiles.csv",
    index=False,
    encoding="utf-8-sig"
)


# Centroidleri de sakla.
np.save(
    "kmeans_centroids.npy",
    kmeans.cluster_centers_
)


print("\n")
print("=" * 90)

print(
    "Dosya oluşturuldu:"
    " kmeans_article_clusters.csv"
)

print(
    "Dosya oluşturuldu:"
    " kmeans_cluster_topic_profiles.csv"
)

print(
    "Dosya oluşturuldu:"
    " kmeans_centroids.npy"
)