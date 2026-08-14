import os
import time

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)


# --------------------------------------------------
# 1. AYARLAR
# --------------------------------------------------

EMBEDDING_FILES = {
    "E5": "embeddings/E5_embeddings.npy",
    "BGE-M3": "embeddings/BGE-M3_embeddings.npy",
    "Qwen3": "embeddings/Qwen3_embeddings.npy",
}

K_VALUES = [
    100,
    125,
    150,
    175,
    200,
    225,
    250
]

RANDOM_STATE = 42


# --------------------------------------------------
# 2. SONUÇLAR
# --------------------------------------------------

results = []


# --------------------------------------------------
# 3. HER EMBEDDING MODELİNİ TEST ET
# --------------------------------------------------

for model_name, file_path in EMBEDDING_FILES.items():

    print("\n")
    print("=" * 75)
    print("EMBEDDING MODELİ:", model_name)
    print("=" * 75)

    embeddings = np.load(file_path)

    print("Orijinal shape:", embeddings.shape)
    print("Orijinal dtype:", embeddings.dtype)

    # Üç modeli clustering açısından aynı veri tipinde
    # değerlendirmek için float32 yapıyoruz.
    embeddings = embeddings.astype(np.float32)

    print("Clustering dtype:", embeddings.dtype)


    # --------------------------------------------------
    # 4. FARKLI K DEĞERLERİ
    # --------------------------------------------------

    for k in K_VALUES:

        print(f"\n{model_name} | K = {k}")

        start_time = time.perf_counter()

        kmeans = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10
        )

        cluster_labels = kmeans.fit_predict(
            embeddings
        )

        clustering_time = (
            time.perf_counter() - start_time
        )


        # --------------------------------------------------
        # 5. METRİKLER
        # --------------------------------------------------

        silhouette = silhouette_score(
            embeddings,
            cluster_labels,
            metric="cosine"
        )

        davies_bouldin = davies_bouldin_score(
            embeddings,
            cluster_labels
        )

        calinski_harabasz = calinski_harabasz_score(
            embeddings,
            cluster_labels
        )


        # --------------------------------------------------
        # 6. KÜME BOYUTLARI
        # --------------------------------------------------

        unique_clusters, cluster_counts = np.unique(
            cluster_labels,
            return_counts=True
        )

        smallest_cluster = int(
            cluster_counts.min()
        )

        largest_cluster = int(
            cluster_counts.max()
        )

        average_cluster_size = float(
            cluster_counts.mean()
        )


        # --------------------------------------------------
        # 7. EKRANA YAZ
        # --------------------------------------------------

        print(
            f"Silhouette       : {silhouette:.4f}"
        )

        print(
            f"Davies-Bouldin   : {davies_bouldin:.4f}"
        )

        print(
            f"Calinski-Harabasz: {calinski_harabasz:.2f}"
        )

        print(
            f"Süre             : {clustering_time:.2f} sn"
        )

        print(
            f"En küçük küme    : {smallest_cluster}"
        )

        print(
            f"En büyük küme    : {largest_cluster}"
        )


        # --------------------------------------------------
        # 8. SONUÇ TABLOSUNA EKLE
        # --------------------------------------------------

        results.append({
            "Model": model_name,
            "K": k,
            "Silhouette": round(
                silhouette,
                4
            ),
            "Davies_Bouldin": round(
                davies_bouldin,
                4
            ),
            "Calinski_Harabasz": round(
                calinski_harabasz,
                2
            ),
            "Time_sec": round(
                clustering_time,
                2
            ),
            "Smallest_Cluster": smallest_cluster,
            "Largest_Cluster": largest_cluster,
            "Average_Cluster_Size": round(
                average_cluster_size,
                2
            )
        })


# --------------------------------------------------
# 9. TÜM SONUÇLARI GÖSTER
# --------------------------------------------------

results_df = pd.DataFrame(
    results
)

print("\n\n")
print("=" * 110)
print("K-MEANS KARŞILAŞTIRMA SONUÇLARI")
print("=" * 110)

print(
    results_df.to_string(
        index=False
    )
)


# --------------------------------------------------
# 10. HER MODELİN EN İYİ K DEĞERİ
# --------------------------------------------------

print("\n")
print("=" * 110)
print("SILHOUETTE'A GÖRE HER MODELİN EN İYİ K DEĞERİ")
print("=" * 110)

for model_name in EMBEDDING_FILES.keys():

    model_results = results_df[
        results_df["Model"] == model_name
    ]

    best_row = model_results.loc[
        model_results["Silhouette"].idxmax()
    ]

    print(
        f"{model_name}: "
        f"K={int(best_row['K'])} | "
        f"Silhouette={best_row['Silhouette']}"
    )


# --------------------------------------------------
# 11. CSV'YE KAYDET
# --------------------------------------------------

results_df.to_csv(
    "kmeans_model_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nSonuç dosyası oluşturuldu:"
    " kmeans_model_results.csv"
)