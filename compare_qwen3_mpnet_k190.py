import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)


# ============================================================
# AYARLAR
# ============================================================

TEXT_FILE = "real_trdizin_texts.csv"
SUBJECT_FILE = "trdizin_subject_hierarchy.csv"

EMBEDDING_FILES = {
    "Qwen3": "embeddings/Qwen3_embeddings.npy",
    "MPNet-Multilingual": "embeddings/MPNet_multilingual_embeddings.npy"
}

K = 190
RANDOM_STATE = 42

OUTPUT_FILE = "results/qwen3_vs_mpnet_k190.csv"


# ============================================================
# VERİLER
# ============================================================

texts = pd.read_csv(
    TEXT_FILE,
    encoding="utf-8-sig"
)

subjects = pd.read_csv(
    SUBJECT_FILE,
    encoding="utf-8-sig"
)


subjects["leaf_subject"] = (
    subjects["leaf_subject"]
    .fillna("")
    .astype(str)
    .str.strip()
)

# Sadece gerçek leaf etiketler
subjects = subjects[
    subjects["leaf_subject"] != ""
].copy()


# ============================================================
# MAKALE -> GERÇEK TR DİZİN KONU SETİ
# ============================================================

article_labels = (
    subjects
    .groupby("article_id")["subject_fullname"]
    .apply(
        lambda values:
            set(
                values
                .dropna()
                .astype(str)
            )
    )
    .to_dict()
)


# ============================================================
# SATIR EMBEDDING -> MAKALE EMBEDDING
# ============================================================

def build_article_embeddings(row_embeddings):

    article_ids = []
    vectors = []

    for article_id, group in texts.groupby("article_id"):

        # Purity değerlendirmesi için
        # sadece etiketi olanları kullanıyoruz
        if article_id not in article_labels:
            continue

        indices = group.index.to_numpy()

        article_vector = (
            row_embeddings[indices]
            .mean(axis=0)
        )

        norm = np.linalg.norm(
            article_vector
        )

        if norm > 0:
            article_vector = (
                article_vector
                /
                norm
            )

        article_ids.append(
            article_id
        )

        vectors.append(
            article_vector
        )

    X = np.vstack(
        vectors
    ).astype(np.float32)

    return article_ids, X


# ============================================================
# MULTI-LABEL PURITY
# ============================================================

def calculate_multilabel_purity(
    article_ids,
    cluster_labels
):

    total_correct = 0
    cluster_purities = []

    for cluster_id in np.unique(
        cluster_labels
    ):

        indices = np.where(
            cluster_labels
            ==
            cluster_id
        )[0]

        cluster_article_ids = [
            article_ids[i]
            for i in indices
        ]

        subject_counts = {}

        for article_id in cluster_article_ids:

            labels = article_labels[
                article_id
            ]

            for label in labels:

                subject_counts[label] = (
                    subject_counts.get(
                        label,
                        0
                    )
                    +
                    1
                )

        if not subject_counts:
            continue

        dominant_count = max(
            subject_counts.values()
        )

        purity = (
            dominant_count
            /
            len(cluster_article_ids)
        )

        cluster_purities.append(
            purity
        )

        total_correct += (
            dominant_count
        )

    weighted_purity = (
        total_correct
        /
        len(article_ids)
    )

    mean_cluster_purity = (
        np.mean(
            cluster_purities
        )
    )

    return (
        weighted_purity,
        mean_cluster_purity
    )


# ============================================================
# TEST
# ============================================================

results = []


for model_name, file_name in EMBEDDING_FILES.items():

    print("\n" + "=" * 100)
    print("MODEL:", model_name)
    print("=" * 100)

    row_embeddings = np.load(
        file_name
    ).astype(np.float32)


    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    norms = np.linalg.norm(
        row_embeddings,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1

    row_embeddings = (
        row_embeddings
        /
        norms
    )


    # --------------------------------------------------------
    # ARTICLE LEVEL
    # --------------------------------------------------------

    article_ids, X = (
        build_article_embeddings(
            row_embeddings
        )
    )


    print(
        "Kullanılan makale:",
        len(article_ids)
    )

    print(
        "Embedding boyutu:",
        X.shape[1]
    )


    # --------------------------------------------------------
    # K-MEANS
    # --------------------------------------------------------

    model = KMeans(
        n_clusters=K,
        random_state=RANDOM_STATE,
        n_init=10
    )

    labels = model.fit_predict(
        X
    )


    # --------------------------------------------------------
    # METRİKLER
    # --------------------------------------------------------

    silhouette = silhouette_score(
        X,
        labels,
        metric="cosine"
    )

    davies_bouldin = (
        davies_bouldin_score(
            X,
            labels
        )
    )

    calinski = (
        calinski_harabasz_score(
            X,
            labels
        )
    )


    weighted_purity, mean_purity = (
        calculate_multilabel_purity(
            article_ids,
            labels
        )
    )


    # --------------------------------------------------------
    # CLUSTER BOYUTLARI
    # --------------------------------------------------------

    sizes = (
        pd.Series(labels)
        .value_counts()
    )

    singleton = int(
        (sizes == 1).sum()
    )

    le5 = int(
        (sizes <= 5).sum()
    )

    le10 = int(
        (sizes <= 10).sum()
    )

    smallest = int(
        sizes.min()
    )

    largest = int(
        sizes.max()
    )

    average_size = float(
        sizes.mean()
    )


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        "Silhouette:",
        round(
            silhouette,
            6
        )
    )

    print(
        "Davies-Bouldin:",
        round(
            davies_bouldin,
            6
        )
    )

    print(
        "Calinski-Harabasz:",
        round(
            calinski,
            6
        )
    )

    print(
        "Weighted Purity:",
        round(
            weighted_purity,
            6
        )
    )

    print(
        "Mean Cluster Purity:",
        round(
            mean_purity,
            6
        )
    )

    print(
        "Singleton:",
        singleton
    )

    print(
        "<=5 cluster:",
        le5
    )

    print(
        "<=10 cluster:",
        le10
    )


    results.append(
        {
            "Model":
                model_name,

            "Embedding_Dim":
                X.shape[1],

            "K":
                K,

            "Articles":
                len(article_ids),

            "Silhouette":
                silhouette,

            "Davies_Bouldin":
                davies_bouldin,

            "Calinski_Harabasz":
                calinski,

            "Weighted_Purity":
                weighted_purity,

            "Mean_Cluster_Purity":
                mean_purity,

            "Singleton_Clusters":
                singleton,

            "Clusters_LE_5":
                le5,

            "Clusters_LE_10":
                le10,

            "Smallest_Cluster":
                smallest,

            "Largest_Cluster":
                largest,

            "Average_Cluster_Size":
                average_size
        }
    )


# ============================================================
# SONUÇ TABLOSU
# ============================================================

result_df = pd.DataFrame(
    results
)


print("\n" + "=" * 140)
print("QWEN3 vs MULTILINGUAL MPNET - K=190")
print("=" * 140)


print(
    result_df.to_string(
        index=False
    )
)


# ============================================================
# KAZANANLAR
# ============================================================

print("\n" + "=" * 100)
print("METRİK BAZINDA KAZANAN")
print("=" * 100)


print(
    "Silhouette:",
    result_df.loc[
        result_df[
            "Silhouette"
        ].idxmax(),
        "Model"
    ]
)

print(
    "Davies-Bouldin:",
    result_df.loc[
        result_df[
            "Davies_Bouldin"
        ].idxmin(),
        "Model"
    ]
)

print(
    "Calinski-Harabasz:",
    result_df.loc[
        result_df[
            "Calinski_Harabasz"
        ].idxmax(),
        "Model"
    ]
)

print(
    "Weighted Purity:",
    result_df.loc[
        result_df[
            "Weighted_Purity"
        ].idxmax(),
        "Model"
    ]
)

print(
    "Mean Cluster Purity:",
    result_df.loc[
        result_df[
            "Mean_Cluster_Purity"
        ].idxmax(),
        "Model"
    ]
)


# ============================================================
# KAYDET
# ============================================================

result_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


print(
    "\nDosya oluşturuldu:",
    OUTPUT_FILE
)