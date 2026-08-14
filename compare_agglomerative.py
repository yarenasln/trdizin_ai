import os
import time
import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


# ============================================================
# 1. AYARLAR
# ============================================================

EMBEDDING_FILES = {
    "E5": "embeddings/E5_embeddings.npy",
    "BGE-M3": "embeddings/BGE-M3_embeddings.npy",
    "Qwen3": "embeddings/Qwen3_embeddings.npy",
}

# K-Means ile karşılaştırabilmek için
# aynı K değerlerini deniyoruz.
K_VALUES = [
    200,
    225,
    250,
    275,
    300,
    325,
    350,
    400
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


    # ========================================================
    # 4. NORMALIZE
    # ========================================================
    #
    # Embeddingleri birim uzunluğa getiriyoruz.
    #

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1

    embeddings = embeddings / norms


    # ========================================================
    # 5. FARKLI K DEĞERLERİNİ TEST ET
    # ========================================================

    for k in K_VALUES:

        print(
            f"\nK = {k}"
        )

        start_time = time.perf_counter()


        # ----------------------------------------------------
        # AGGLOMERATIVE CLUSTERING
        # ----------------------------------------------------
        #
        # metric="cosine":
        # Vektörlerin yönlerine göre uzaklık.
        #
        # linkage="average":
        # İki küme arasındaki ortalama uzaklığı kullanır.
        #

        clustering = AgglomerativeClustering(
            n_clusters=k,
            metric="cosine",
            linkage="average"
        )

        labels = clustering.fit_predict(
            embeddings
        )


        clustering_time = (
            time.perf_counter()
            - start_time
        )


        # ----------------------------------------------------
        # SILHOUETTE SCORE
        # ----------------------------------------------------

        silhouette = silhouette_score(
            embeddings,
            labels,
            metric="cosine"
        )


        # ----------------------------------------------------
        # KÜME BOYUTLARI
        # ----------------------------------------------------

        unique_labels, counts = np.unique(
            labels,
            return_counts=True
        )

        smallest_cluster = int(
            counts.min()
        )

        largest_cluster = int(
            counts.max()
        )

        average_cluster_size = float(
            counts.mean()
        )


        print(
            f"Silhouette (Silüet skoru): "
            f"{silhouette:.4f}"
        )

        print(
            f"Süre: "
            f"{clustering_time:.2f} saniye"
        )

        print(
            f"En küçük küme: "
            f"{smallest_cluster}"
        )

        print(
            f"En büyük küme: "
            f"{largest_cluster}"
        )


        # ----------------------------------------------------
        # SONUÇ EKLE
        # ----------------------------------------------------

        results.append({

            "Model":
                model_name,

            "K":
                k,

            "Silhouette":
                round(
                    float(silhouette),
                    4
                ),

            "Time_sec":
                round(
                    clustering_time,
                    2
                ),

            "Smallest_Cluster":
                smallest_cluster,

            "Largest_Cluster":
                largest_cluster,

            "Average_Cluster_Size":
                round(
                    average_cluster_size,
                    2
                )
        })


# ============================================================
# 6. TÜM SONUÇLAR
# ============================================================

results_df = pd.DataFrame(
    results
)

print("\n\n")
print("=" * 105)

print(
    "AGGLOMERATIVE CLUSTERING "
    "(BİRLEŞTİRİCİ HİYERARŞİK KÜMELEME) "
    "KARŞILAŞTIRMASI"
)

print("=" * 105)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 7. HER MODELİN EN İYİ K DEĞERİ
# ============================================================

best_rows = (
    results_df
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
    "SILHOUETTE'A GÖRE HER MODELİN "
    "EN İYİ AGGLOMERATIVE SONUCU"
)

print("=" * 100)

print(
    best_rows[
        [
            "Model",
            "K",
            "Silhouette",
            "Time_sec",
            "Smallest_Cluster",
            "Largest_Cluster"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 8. CSV KAYDET
# ============================================================

results_df.to_csv(
    "agglomerative_model_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nSonuç dosyası oluşturuldu:"
    " agglomerative_model_results.csv"
)