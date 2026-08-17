import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# DOSYALAR
# ============================================================

TEXT_FILE = "real_trdizin_texts.csv"
SUBJECT_FILE = "trdizin_subject_hierarchy.csv"
EMBEDDING_FILE = "embeddings/Qwen3_embeddings.npy"

OUTPUT_FILE = "kmeans_new_dataset_k_results.csv"


# ============================================================
# TEST EDİLECEK K DEĞERLERİ
# ============================================================

K_VALUES = [
    100,
    125,
    150,
    175,
    195,
    200,
    225,
    250,
    300,
    350,
    400
]

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
print("YENİ VERİ SETİ - QWEN3 + K-MEANS K ANALİZİ")
print("=" * 110)

print("Metin satırı:", len(texts))
print("Embedding shape:", embeddings.shape)


# ============================================================
# GÜVENLİK KONTROLÜ
# ============================================================

if len(texts) != len(embeddings):

    raise ValueError(
        f"Metin ve embedding satır sayıları uyuşmuyor! "
        f"Metin={len(texts)}, embedding={len(embeddings)}"
    )


# ============================================================
# SADECE EN ALT KONULAR
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
# MAKALE -> GERÇEK KONU SETİ
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


print(
    "Etiketli makale:",
    len(article_labels)
)


# ============================================================
# ARTICLE LEVEL EMBEDDING
# ============================================================
#
# Aynı makalenin TUR / ENG kayıtları varsa
# embedding ortalaması alınıyor.
# ============================================================

article_ids = []
article_vectors = []


for article_id, group in texts.groupby(
    "article_id"
):

    if article_id not in article_labels:
        continue


    indices = group.index.to_numpy()

    vectors = embeddings[
        indices
    ]


    vector = vectors.mean(
        axis=0
    )


    norm = np.linalg.norm(
        vector
    )

    if norm > 0:
        vector = vector / norm


    article_ids.append(
        article_id
    )

    article_vectors.append(
        vector
    )


X = np.vstack(
    article_vectors
).astype(np.float32)


print(
    "K-Means için kullanılan makale:",
    len(article_ids)
)

print(
    "Article embedding shape:",
    X.shape
)


# ============================================================
# MULTI-LABEL WEIGHTED PURITY
# ============================================================

def calculate_multilabel_purity(
    article_ids,
    cluster_labels
):

    total_correct = 0
    total_articles = len(
        article_ids
    )

    cluster_purities = []


    unique_clusters = np.unique(
        cluster_labels
    )


    for cluster_id in unique_clusters:

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

            labels = article_labels.get(
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
            len(
                cluster_article_ids
            )
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
        total_articles
        if total_articles
        else 0
    )


    mean_purity = (
        np.mean(
            cluster_purities
        )
        if cluster_purities
        else 0
    )


    return (
        weighted_purity,
        mean_purity
    )


# ============================================================
# K TESTLERİ
# ============================================================

results = []


for k in K_VALUES:

    print("\n" + "=" * 100)
    print("K TEST EDİLİYOR:", k)
    print("=" * 100)


    if k >= len(X):

        print("K makale sayısından büyük, atlandı.")
        continue


    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10
    )


    labels = model.fit_predict(
        X
    )


    # --------------------------------------------------------
    # COSINE SILHOUETTE
    # --------------------------------------------------------

    silhouette = silhouette_score(
        X,
        labels,
        metric="cosine"
    )


    # --------------------------------------------------------
    # PURITY
    # --------------------------------------------------------

    weighted_purity, mean_purity = (
        calculate_multilabel_purity(
            article_ids,
            labels
        )
    )


    # --------------------------------------------------------
    # CLUSTER DAĞILIMI
    # --------------------------------------------------------

    cluster_sizes = pd.Series(
        labels
    ).value_counts()


    singleton = int(
        (
            cluster_sizes == 1
        ).sum()
    )


    smallest = int(
        cluster_sizes.min()
    )


    largest = int(
        cluster_sizes.max()
    )


    average_size = float(
        cluster_sizes.mean()
    )


    median_size = float(
        cluster_sizes.median()
    )


    clusters_le_5 = int(
        (
            cluster_sizes <= 5
        ).sum()
    )


    clusters_le_10 = int(
        (
            cluster_sizes <= 10
        ).sum()
    )


    print(
        "Cosine Silhouette:",
        round(
            silhouette,
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
        "Mean Cluster Purity:",
        round(
            mean_purity,
            4
        )
    )

    print(
        "Singleton:",
        singleton
    )

    print(
        "En küçük cluster:",
        smallest
    )

    print(
        "En büyük cluster:",
        largest
    )

    print(
        "Ortalama cluster büyüklüğü:",
        round(
            average_size,
            2
        )
    )

    print(
        "Medyan cluster büyüklüğü:",
        round(
            median_size,
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


    results.append(
        {
            "K":
                k,

            "Silhouette":
                round(
                    silhouette,
                    6
                ),

            "Weighted_Purity":
                round(
                    weighted_purity,
                    6
                ),

            "Mean_Cluster_Purity":
                round(
                    mean_purity,
                    6
                ),

            "Singleton_Clusters":
                singleton,

            "Singleton_Ratio":
                round(
                    singleton / k,
                    6
                ),

            "Smallest_Cluster":
                smallest,

            "Largest_Cluster":
                largest,

            "Average_Cluster_Size":
                round(
                    average_size,
                    2
                ),

            "Median_Cluster_Size":
                round(
                    median_size,
                    2
                ),

            "Clusters_LE_5":
                clusters_le_5,

            "Clusters_LE_10":
                clusters_le_10
        }
    )


# ============================================================
# SONUÇ TABLOSU
# ============================================================

result_df = pd.DataFrame(
    results
)


print("\n" + "=" * 150)
print("GENEL K KARŞILAŞTIRMASI")
print("=" * 150)


pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    250
)


print(
    result_df.to_string(
        index=False
    )
)


# ============================================================
# EN YÜKSEK SILHOUETTE
# ============================================================

best_silhouette = result_df.loc[
    result_df[
        "Silhouette"
    ].idxmax()
]


print("\n" + "=" * 110)
print("EN YÜKSEK SILHOUETTE")
print("=" * 110)

print(
    best_silhouette.to_string()
)


# ============================================================
# EN YÜKSEK PURITY
# ============================================================

best_purity = result_df.loc[
    result_df[
        "Weighted_Purity"
    ].idxmax()
]


print("\n" + "=" * 110)
print("EN YÜKSEK WEIGHTED PURITY")
print("=" * 110)

print(
    best_purity.to_string()
)


# ============================================================
# CSV
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