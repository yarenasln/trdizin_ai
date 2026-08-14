import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


# ============================================================
# 1. AYARLAR
# ============================================================

TEXT_FILE = "real_trdizin_texts.csv"
SUBJECT_FILE = "trdizin_subject_hierarchy.csv"

EMBEDDING_FILES = {
    "E5": "embeddings/E5_embeddings.npy",
    "BGE-M3": "embeddings/BGE-M3_embeddings.npy",
    "Qwen3": "embeddings/Qwen3_embeddings.npy",
}

# 195 gerçek leaf subject (en-alt konu) civarını özellikle
# merkez alıyoruz; ayrıca alt/üst değerleri de kontrol ediyoruz.
K_VALUES = [
    100,
    150,
    175,
    195,
    200,
    225,
    250,
    300,
    400
]

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

subjects["leaf_subject"] = (
    subjects["leaf_subject"]
    .fillna("")
    .astype(str)
    .str.strip()
)

# Sadece gerçek en-alt konular
subjects = subjects[
    subjects["leaf_subject"] != ""
].copy()


# ============================================================
# 3. HER MAKALEYİ GERÇEK KONU SETİYLE EŞLEŞTİR
# ============================================================

article_labels = (
    subjects
    .groupby("article_id")["subject_fullname"]
    .apply(
        lambda values: set(
            values.dropna().astype(str)
        )
    )
    .to_dict()
)


# ============================================================
# 4. ARTICLE-LEVEL EMBEDDING
# ============================================================
#
# TUR ve ENG aynı makaleyse embedding ortalamasını alıyoruz.
# Böylece aynı makale iki ayrı veri noktası olmuyor.
#

def build_article_embeddings(row_embeddings):

    article_vectors = {}
    article_ids = []

    for article_id, group in texts.groupby("article_id"):

        if article_id not in article_labels:
            continue

        indices = group.index.to_numpy()

        vectors = row_embeddings[indices]

        vector = vectors.mean(axis=0)

        norm = np.linalg.norm(vector)

        if norm > 0:
            vector = vector / norm

        article_vectors[article_id] = vector
        article_ids.append(article_id)

    matrix = np.vstack([
        article_vectors[article_id]
        for article_id in article_ids
    ]).astype(np.float32)

    return article_ids, matrix


# ============================================================
# 5. MULTI-LABEL CLUSTER PURITY
# ============================================================
#
# Her kümede:
# En çok makalede görülen TR Dizin konusunu buluyoruz.
#
# Örneğin:
#
# Küme 8:
# A -> Yapay Zeka, Robotik
# B -> Yapay Zeka
# C -> Yapay Zeka, Bilgi Sistemleri
#
# dominant subject = Yapay Zeka
#
# purity = 3 / 3 = 1.0
#

def calculate_multilabel_purity(
    article_ids,
    cluster_labels
):

    total_correct = 0
    total_articles = len(article_ids)

    cluster_purities = []

    unique_clusters = np.unique(
        cluster_labels
    )

    singleton_clusters = 0


    for cluster_id in unique_clusters:

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

            labels = article_labels[
                article_id
            ]

            for label in labels:

                subject_counts[label] = (
                    subject_counts.get(
                        label,
                        0
                    ) + 1
                )


        if not subject_counts:
            continue


        dominant_count = max(
            subject_counts.values()
        )

        cluster_purity = (
            dominant_count
            / len(cluster_article_ids)
        )

        cluster_purities.append(
            cluster_purity
        )

        total_correct += dominant_count


    weighted_purity = (
        total_correct / total_articles
        if total_articles > 0
        else 0
    )


    mean_cluster_purity = (
        np.mean(cluster_purities)
        if cluster_purities
        else 0
    )


    return (
        weighted_purity,
        mean_cluster_purity,
        singleton_clusters
    )


# ============================================================
# 6. SONUÇLAR
# ============================================================

results = []


# ============================================================
# 7. HER EMBEDDING MODELİNİ TEST ET
# ============================================================

for model_name, file_name in EMBEDDING_FILES.items():

    print("\n")
    print("=" * 80)
    print("MODEL:", model_name)
    print("=" * 80)

    embeddings = np.load(
        file_name
    ).astype(np.float32)


    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1

    embeddings = embeddings / norms


    # --------------------------------------------------------
    # Makale seviyesine indir
    # --------------------------------------------------------

    article_ids, X = build_article_embeddings(
        embeddings
    )

    print(
        "Kullanılan makale:",
        len(article_ids)
    )


    # ========================================================
    # 8. ALGORİTMALAR
    # ========================================================

    for algorithm in [
        "K-Means",
        "Agglomerative"
    ]:

        print("\nALGORİTMA:", algorithm)


        for k in K_VALUES:

            if k >= len(X):
                continue


            # ------------------------------------------------
            # KÜMELEME
            # ------------------------------------------------

            if algorithm == "K-Means":

                model = KMeans(
                    n_clusters=k,
                    random_state=RANDOM_STATE,
                    n_init=10
                )

                labels = model.fit_predict(
                    X
                )

            else:

                model = AgglomerativeClustering(
                    n_clusters=k,
                    metric="cosine",
                    linkage="average"
                )

                labels = model.fit_predict(
                    X
                )


            # ------------------------------------------------
            # SILHOUETTE
            # ------------------------------------------------

            silhouette = silhouette_score(
                X,
                labels,
                metric="cosine"
            )


            # ------------------------------------------------
            # PURITY
            # ------------------------------------------------

            (
                weighted_purity,
                mean_purity,
                singleton_clusters
            ) = calculate_multilabel_purity(
                article_ids,
                labels
            )


            # ------------------------------------------------
            # KÜME BOYUTLARI
            # ------------------------------------------------

            _, counts = np.unique(
                labels,
                return_counts=True
            )

            smallest_cluster = int(
                counts.min()
            )

            largest_cluster = int(
                counts.max()
            )

            singleton_ratio = (
                singleton_clusters / k
            )


            print(
                f"{algorithm} | "
                f"K={k} | "
                f"Silhouette={silhouette:.4f} | "
                f"Purity={weighted_purity:.4f} | "
                f"Singleton={singleton_clusters}"
            )


            results.append({

                "Model": model_name,

                "Algorithm": algorithm,

                "K": k,

                "Silhouette": round(
                    float(silhouette),
                    4
                ),

                "Weighted_Purity": round(
                    float(weighted_purity),
                    4
                ),

                "Mean_Cluster_Purity": round(
                    float(mean_purity),
                    4
                ),

                "Singleton_Clusters":
                    singleton_clusters,

                "Singleton_Ratio": round(
                    singleton_ratio,
                    4
                ),

                "Smallest_Cluster":
                    smallest_cluster,

                "Largest_Cluster":
                    largest_cluster
            })


# ============================================================
# 9. SONUÇ TABLOSU
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n\n")
print("=" * 120)

print(
    "KÜMELEME ↔ TR DİZİN KONU UYUMU"
)

print("=" * 120)


print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 10. 195 K İÇİN ÖZEL KARŞILAŞTIRMA
# ============================================================

k195 = results_df[
    results_df["K"] == 195
].copy()


print("\n\n")
print("=" * 110)

print(
    "K=195 ÖZEL KARŞILAŞTIRMA "
    "(GERÇEK EN-ALT KONU SAYISINA YAKIN)"
)

print("=" * 110)


print(
    k195[
        [
            "Model",
            "Algorithm",
            "Silhouette",
            "Weighted_Purity",
            "Singleton_Clusters",
            "Largest_Cluster"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 11. CSV
# ============================================================

results_df.to_csv(
    "cluster_subject_alignment_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nSonuç dosyası oluşturuldu: "
    "cluster_subject_alignment_results.csv"
)