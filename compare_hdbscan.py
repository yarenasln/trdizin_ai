import numpy as np
import pandas as pd
import hdbscan

from sklearn.metrics import silhouette_score


# ============================================================
# 1. AYARLAR
# ============================================================

EMBEDDING_FILES = {
    "E5": "embeddings/E5_embeddings.npy",
    "BGE-M3": "embeddings/BGE-M3_embeddings.npy",
    "Qwen3": "embeddings/Qwen3_embeddings.npy",
}

# HDBSCAN'de K-Means'teki gibi K vermiyoruz.
# Bunun yerine minimum küme büyüklüğünü deniyoruz.
MIN_CLUSTER_SIZES = [
    5,
    10,
    15,
    20,
    30,
    40,
    50
]


# ============================================================
# 2. SONUÇLAR
# ============================================================

results = []


# ============================================================
# 3. HER EMBEDDING MODELİNİ TEST ET
# ============================================================

for model_name, embedding_file in EMBEDDING_FILES.items():

    print("\n")
    print("=" * 80)
    print("MODEL:", model_name)
    print("=" * 80)

    embeddings = np.load(
        embedding_file
    ).astype(np.float32)

    print(
        "Embedding shape:",
        embeddings.shape
    )


    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1

    embeddings = (
        embeddings / norms
    )


    # ========================================================
    # 4. FARKLI MIN_CLUSTER_SIZE DEĞERLERİNİ DENE
    # ========================================================

    for min_cluster_size in MIN_CLUSTER_SIZES:

        print(
            f"\nmin_cluster_size = "
            f"{min_cluster_size}"
        )


        # ----------------------------------------------------
        # HDBSCAN
        # ----------------------------------------------------

        clusterer = hdbscan.HDBSCAN(

            min_cluster_size=min_cluster_size,

            # Cosine benzerliğine yakın davranması için
            # normalize edilmiş embeddinglerde
            # Euclidean kullanıyoruz.
            metric="euclidean",

            cluster_selection_method="eom",

            prediction_data=False
        )


        labels = clusterer.fit_predict(
            embeddings
        )


        # ----------------------------------------------------
        # KÜME SAYISI
        # ----------------------------------------------------
        #
        # HDBSCAN -1 değerini noise (gürültü)
        # olarak işaretler.
        #

        unique_labels = set(
            labels
        )

        cluster_count = len(
            unique_labels - {-1}
        )


        # ----------------------------------------------------
        # NOISE
        # ----------------------------------------------------

        noise_count = int(
            np.sum(labels == -1)
        )

        noise_ratio = (
            noise_count / len(labels)
        )


        # ----------------------------------------------------
        # SILHOUETTE
        # --------------------------------------------------------
        #
        # Gürültü noktalarını Silhouette hesabına
        # dahil etmiyoruz.
        #

        non_noise_mask = (
            labels != -1
        )

        clean_embeddings = embeddings[
            non_noise_mask
        ]

        clean_labels = labels[
            non_noise_mask
        ]


        unique_clean_labels = np.unique(
            clean_labels
        )


        if (
            len(unique_clean_labels) >= 2
            and
            len(clean_embeddings)
            > len(unique_clean_labels)
        ):

            silhouette = silhouette_score(
                clean_embeddings,
                clean_labels,
                metric="euclidean"
            )

        else:

            silhouette = np.nan


        print(
            "Küme sayısı:",
            cluster_count
        )

        print(
            f"Noise (gürültü): "
            f"{noise_count} "
            f"({noise_ratio:.2%})"
        )

        if np.isnan(silhouette):

            print(
                "Silhouette (Silüet skoru): "
                "hesaplanamadı"
            )

        else:

            print(
                f"Silhouette (Silüet skoru): "
                f"{silhouette:.4f}"
            )


        # ----------------------------------------------------
        # SONUÇ
        # ----------------------------------------------------

        results.append({

            "Model":
                model_name,

            "Min_Cluster_Size":
                min_cluster_size,

            "Cluster_Count":
                cluster_count,

            "Noise_Count":
                noise_count,

            "Noise_Ratio":
                round(
                    noise_ratio,
                    4
                ),

            "Silhouette":
                (
                    round(
                        float(silhouette),
                        4
                    )
                    if not np.isnan(
                        silhouette
                    )
                    else np.nan
                )
        })


# ============================================================
# 5. TÜM SONUÇLAR
# ============================================================

results_df = pd.DataFrame(
    results
)

print("\n\n")
print("=" * 100)

print(
    "HDBSCAN MODEL KARŞILAŞTIRMASI"
)

print("=" * 100)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 6. HER MODELİN EN İYİ SONUCU
# ============================================================

valid_results = results_df.dropna(
    subset=["Silhouette"]
)


if not valid_results.empty:

    best_rows = (
        valid_results
        .sort_values(
            "Silhouette",
            ascending=False
        )
        .groupby(
            "Model",
            as_index=False
        )
        .first()
    )


    print("\n\n")
    print("=" * 100)

    print(
        "SILHOUETTE'A GÖRE "
        "HER MODELİN EN İYİ HDBSCAN SONUCU"
    )

    print("=" * 100)

    print(
        best_rows[
            [
                "Model",
                "Min_Cluster_Size",
                "Cluster_Count",
                "Noise_Ratio",
                "Silhouette"
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# 7. CSV KAYDET
# ============================================================

results_df.to_csv(
    "hdbscan_model_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nSonuç dosyası oluşturuldu:"
    " hdbscan_model_results.csv"
)