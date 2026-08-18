import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)


# ============================================================
# DOSYALAR
# ============================================================

ARTICLE_EMBEDDING_FILE = "models/final_article_embeddings.npy"

OUTPUT_FILE = "results/k_literature_metrics.csv"


# ============================================================
# TEST EDİLECEK K DEĞERLERİ
# ============================================================

K_VALUES = [
    150,
    160,
    170,
    180,
    190,
    195,
    200,
    210,
    220,
    225,
    230,
    240,
    250
]

RANDOM_STATE = 42


# ============================================================
# EMBEDDINGLERİ OKU
# ============================================================

X = np.load(
    ARTICLE_EMBEDDING_FILE
).astype(np.float32)


print("=" * 110)
print("K-MEANS LİTERATÜR METRİKLERİ ANALİZİ")
print("=" * 110)

print(
    "Makale embedding shape:",
    X.shape
)


# ============================================================
# K TESTLERİ
# ============================================================

results = []


for k in K_VALUES:

    print("\n" + "=" * 100)
    print("K TEST EDİLİYOR:", k)
    print("=" * 100)


    # --------------------------------------------------------
    # K-MEANS
    # --------------------------------------------------------

    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10
    )


    labels = model.fit_predict(
        X
    )


    # --------------------------------------------------------
    # SILHOUETTE
    # Yüksek olması daha iyi
    # --------------------------------------------------------

    silhouette = silhouette_score(
        X,
        labels,
        metric="cosine"
    )


    # --------------------------------------------------------
    # DAVIES-BOULDIN INDEX
    # Düşük olması daha iyi
    # --------------------------------------------------------

    davies_bouldin = davies_bouldin_score(
        X,
        labels
    )


    # --------------------------------------------------------
    # CALINSKI-HARABASZ
    # Yüksek olması daha iyi
    # --------------------------------------------------------

    calinski_harabasz = (
        calinski_harabasz_score(
            X,
            labels
        )
    )


    # --------------------------------------------------------
    # CLUSTER BOYUTLARI
    # --------------------------------------------------------

    cluster_sizes = (
        pd.Series(labels)
        .value_counts()
    )


    singleton = int(
        (
            cluster_sizes == 1
        ).sum()
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


    smallest_cluster = int(
        cluster_sizes.min()
    )


    largest_cluster = int(
        cluster_sizes.max()
    )


    average_cluster_size = float(
        cluster_sizes.mean()
    )


    median_cluster_size = float(
        cluster_sizes.median()
    )


    # --------------------------------------------------------
    # SONUÇ
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
            calinski_harabasz,
            4
        )
    )

    print(
        "Singleton:",
        singleton
    )

    print(
        "Clusters <= 5:",
        clusters_le_5
    )

    print(
        "Clusters <= 10:",
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

            "Davies_Bouldin":
                round(
                    davies_bouldin,
                    6
                ),

            "Calinski_Harabasz":
                round(
                    calinski_harabasz,
                    6
                ),

            "Singleton_Clusters":
                singleton,

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
    )


# ============================================================
# DATAFRAME
# ============================================================

result_df = pd.DataFrame(
    results
)


# ============================================================
# GENEL TABLO
# ============================================================

print("\n" + "=" * 150)
print("GENEL LİTERATÜR METRİKLERİ KARŞILAŞTIRMASI")
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
# EN İYİ SILHOUETTE
# ============================================================

best_silhouette = result_df.loc[
    result_df[
        "Silhouette"
    ].idxmax()
]


print("\n" + "=" * 110)
print("EN İYİ SILHOUETTE")
print("=" * 110)

print(
    best_silhouette.to_string()
)


# ============================================================
# EN İYİ DAVIES-BOULDIN
# ============================================================

best_db = result_df.loc[
    result_df[
        "Davies_Bouldin"
    ].idxmin()
]


print("\n" + "=" * 110)
print("EN İYİ DAVIES-BOULDIN")
print("=" * 110)

print(
    best_db.to_string()
)


# ============================================================
# EN İYİ CALINSKI-HARABASZ
# ============================================================

best_ch = result_df.loc[
    result_df[
        "Calinski_Harabasz"
    ].idxmax()
]


print("\n" + "=" * 110)
print("EN İYİ CALINSKI-HARABASZ")
print("=" * 110)

print(
    best_ch.to_string()
)


# ============================================================
# NORMALİZE DENGE PUANI
# ============================================================
#
# Bu resmi bir akademik clustering metriği değildir.
# Sadece farklı metrikleri birlikte görmek için
# karar destek puanı olarak kullanılmaktadır.
# ============================================================

def normalize(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:

        return pd.Series(
            [1.0] * len(series),
            index=series.index
        )

    return (
        series - minimum
    ) / (
        maximum - minimum
    )


# Yüksek olması iyi
result_df[
    "Silhouette_N"
] = normalize(
    result_df[
        "Silhouette"
    ]
)


# Davies-Bouldin düşük olması iyi olduğu için ters çeviriyoruz
result_df[
    "Davies_Bouldin_N"
] = (
    1
    -
    normalize(
        result_df[
            "Davies_Bouldin"
        ]
    )
)


# Yüksek olması iyi
result_df[
    "Calinski_Harabasz_N"
] = normalize(
    result_df[
        "Calinski_Harabasz"
    ]
)


# Küçük cluster sayısı düşük olması iyi
result_df[
    "Small_Cluster_N"
] = (
    1
    -
    normalize(
        result_df[
            "Clusters_LE_10"
        ]
    )
)


# ============================================================
# KARAR DESTEK PUANI
# ============================================================
#
# %35 Silhouette
# %25 Davies-Bouldin
# %20 Calinski-Harabasz
# %20 küçük cluster dengesi
# ============================================================

result_df[
    "Literature_Balance_Score"
] = (

    0.35
    *
    result_df[
        "Silhouette_N"
    ]

    +

    0.25
    *
    result_df[
        "Davies_Bouldin_N"
    ]

    +

    0.20
    *
    result_df[
        "Calinski_Harabasz_N"
    ]

    +

    0.20
    *
    result_df[
        "Small_Cluster_N"
    ]
)


# ============================================================
# DENGE PUANI SIRALAMASI
# ============================================================

ranking = result_df.sort_values(
    "Literature_Balance_Score",
    ascending=False
)


print("\n" + "=" * 130)
print("LİTERATÜR METRİKLERİNE GÖRE DENGE PUANI")
print("=" * 130)


print(
    ranking[
        [
            "K",
            "Silhouette",
            "Davies_Bouldin",
            "Calinski_Harabasz",
            "Clusters_LE_10",
            "Literature_Balance_Score"
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# K = 190 ÖZEL
# ============================================================

k190 = result_df[
    result_df["K"] == 190
]


if not k190.empty:

    print("\n" + "=" * 110)
    print("MEVCUT FINAL K = 190")
    print("=" * 110)

    print(
        k190[
            [
                "K",
                "Silhouette",
                "Davies_Bouldin",
                "Calinski_Harabasz",
                "Singleton_Clusters",
                "Clusters_LE_5",
                "Clusters_LE_10",
                "Literature_Balance_Score"
            ]
        ]
        .to_string(
            index=False
        )
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