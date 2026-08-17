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

UMAP_DIMENSIONS = [
    25,
    50,
    100
]

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


print("=" * 110)
print("QWEN3 + UMAP BOYUT OPTİMİZASYONU")
print("=" * 110)

print("Metin satırı:", len(texts))
print("Embedding shape:", embeddings.shape)


# ============================================================
# EN ALT KONULAR
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
# MAKALE SEVİYESİNDE EMBEDDING
# ============================================================

article_ids = []
article_vectors = []


for article_id, group in texts.groupby("article_id"):

    indices = group.index.to_numpy()

    vectors = embeddings[indices]

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


print("Makale sayısı:", len(article_ids))
print("Makale embedding:", X.shape)


# ============================================================
# SONUÇLAR
# ============================================================

results = []


# ============================================================
# HER UMAP BOYUTUNU TEST ET
# ============================================================

for dimension in UMAP_DIMENSIONS:

    print("\n" + "=" * 110)
    print(
        f"UMAP BOYUTU TEST EDİLİYOR: {dimension}"
    )
    print("=" * 110)


    # --------------------------------------------------------
    # UMAP
    # --------------------------------------------------------

    reducer = umap.UMAP(

        n_components=dimension,

        n_neighbors=UMAP_NEIGHBORS,

        min_dist=UMAP_MIN_DIST,

        metric="cosine",

        random_state=RANDOM_STATE
    )


    X_umap = reducer.fit_transform(
        X
    )


    print(
        "UMAP shape:",
        X_umap.shape
    )


    # --------------------------------------------------------
    # K-MEANS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # UMAP UZAYINDA SILHOUETTE
    # --------------------------------------------------------

    umap_silhouette = silhouette_score(

        X_umap,

        cluster_labels,

        metric="euclidean"
    )


    # --------------------------------------------------------
    # ORİJİNAL 1024 BOYUTTA COSINE SILHOUETTE
    # --------------------------------------------------------

    original_cosine_silhouette = silhouette_score(

        X,

        cluster_labels,

        metric="cosine"
    )


    # --------------------------------------------------------
    # WEIGHTED PURITY
    # --------------------------------------------------------

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


        if len(cluster_article_ids) == 1:

            singleton_clusters += 1


        subject_counts = {}


        for article_id in cluster_article_ids:

            labels = article_subjects.get(
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


    # --------------------------------------------------------
    # CLUSTER BOYUTLARI
    # --------------------------------------------------------

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

    average_cluster = float(
        counts.mean()
    )

    median_cluster = float(
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


    # --------------------------------------------------------
    # SONUCU KAYDET
    # --------------------------------------------------------

    results.append({

        "UMAP_Dimension":
            dimension,

        "UMAP_Silhouette":
            umap_silhouette,

        "Original_Cosine_Silhouette":
            original_cosine_silhouette,

        "Weighted_Purity":
            weighted_purity,

        "Singleton":
            singleton_clusters,

        "Smallest_Cluster":
            smallest_cluster,

        "Largest_Cluster":
            largest_cluster,

        "Average_Cluster_Size":
            average_cluster,

        "Median_Cluster_Size":
            median_cluster,

        "Clusters_LE_5":
            clusters_le_5,

        "Clusters_LE_10":
            clusters_le_10
    })


    print(
        "UMAP Silhouette:",
        round(
            umap_silhouette,
            4
        )
    )

    print(
        "1024D Cosine Silhouette:",
        round(
            original_cosine_silhouette,
            4
        )
    )

    print(
        "Weighted Purity:",
        round(
            weighted_purity,
            4
        )
    )

    print(
        "Singleton:",
        singleton_clusters
    )

    print(
        "En büyük cluster:",
        largest_cluster
    )


# ============================================================
# DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# BASELINE EKLE
# ============================================================

baseline = pd.DataFrame([
    {
        "UMAP_Dimension":
            "YOK (1024D)",

        "UMAP_Silhouette":
            np.nan,

        "Original_Cosine_Silhouette":
            0.0385,

        "Weighted_Purity":
            0.5617,

        "Singleton":
            2,

        "Smallest_Cluster":
            1,

        "Largest_Cluster":
            26,

        "Average_Cluster_Size":
            7.69,

        "Median_Cluster_Size":
            7.0,

        "Clusters_LE_5":
            np.nan,

        "Clusters_LE_10":
            np.nan
    }
])


comparison = pd.concat(

    [
        baseline,
        results_df
    ],

    ignore_index=True
)


# ============================================================
# EKRANA YAZ
# ============================================================

print("\n" + "=" * 140)
print("GENEL KARŞILAŞTIRMA")
print("=" * 140)


pd.set_option(
    "display.width",
    250
)

pd.set_option(
    "display.max_columns",
    None
)


print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# EN İYİ SONUÇLAR
# ============================================================

best_silhouette = results_df.loc[
    results_df[
        "Original_Cosine_Silhouette"
    ].idxmax()
]


best_purity = results_df.loc[
    results_df[
        "Weighted_Purity"
    ].idxmax()
]


print("\n" + "=" * 110)
print("EN İYİ ORİJİNAL COSINE SILHOUETTE")
print("=" * 110)

print(
    "UMAP boyutu:",
    int(
        best_silhouette[
            "UMAP_Dimension"
        ]
    )
)

print(
    "Cosine Silhouette:",
    round(
        best_silhouette[
            "Original_Cosine_Silhouette"
        ],
        4
    )
)


print("\n" + "=" * 110)
print("EN İYİ WEIGHTED PURITY")
print("=" * 110)

print(
    "UMAP boyutu:",
    int(
        best_purity[
            "UMAP_Dimension"
        ]
    )
)

print(
    "Weighted Purity:",
    round(
        best_purity[
            "Weighted_Purity"
        ],
        4
    )
)


# ============================================================
# CSV
# ============================================================

comparison.to_csv(

    "umap_dimension_results.csv",

    index=False,

    encoding="utf-8-sig"
)


print(
    "\nDosya oluşturuldu:"
    " umap_dimension_results.csv"
)