import numpy as np
import pandas as pd
import skfuzzy as fuzz

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

# Önce makul K değerlerini deniyoruz.
# 400'ü şimdilik eklemiyoruz çünkü önceki deneylerde
# aşırı kümeleme (over-clustering) problemi gördük.
K_VALUES = [
    100,
    150,
    195,
    225,
    250
]

# Fuzzy C-Means bulanıklık katsayısı
FUZZINESS_M = 2.0

MAX_ITER = 300
ERROR = 0.005
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

# Sadece gerçek en-alt konu etiketleri
subjects = subjects[
    subjects["leaf_subject"] != ""
].copy()


# ============================================================
# 3. MAKALE -> GERÇEK TR DİZİN KONULARI
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
# 4. MAKALE SEVİYESİNDE EMBEDDING
# ============================================================
#
# Aynı makalenin TUR + ENG metinleri varsa
# embedding ortalamasını alıyoruz.
#

def build_article_embeddings(row_embeddings):

    article_ids = []
    article_vectors = []

    for article_id, group in texts.groupby("article_id"):

        if article_id not in article_labels:
            continue

        indices = group.index.to_numpy()

        vectors = row_embeddings[
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

    matrix = np.vstack(
        article_vectors
    ).astype(np.float32)

    return article_ids, matrix


# ============================================================
# 5. MULTI-LABEL PURITY
#    (ÇOK ETİKETLİ KÜME SAFLIĞI)
# ============================================================

def calculate_multilabel_purity(
    article_ids,
    cluster_labels
):

    total_correct = 0

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
                    )
                    + 1
                )


        if not subject_counts:
            continue


        dominant_count = max(
            subject_counts.values()
        )

        total_correct += dominant_count


    weighted_purity = (
        total_correct / len(article_ids)
    )

    return (
        weighted_purity,
        singleton_clusters
    )


# ============================================================
# 6. FUZZY ÜYELİK ANALİZİ
# ============================================================

def analyze_memberships(u):

    # u şekli:
    # cluster sayısı x makale sayısı

    sorted_u = np.sort(
        u,
        axis=0
    )[::-1]

    top1 = sorted_u[0]
    top2 = sorted_u[1]


    mean_top1 = float(
        np.mean(top1)
    )

    mean_top2 = float(
        np.mean(top2)
    )


    # İkinci güçlü üyelik,
    # birinci üyeliğin en az %75'i ise
    # makaleyi "birden fazla kümeye yakın"
    # kabul ediyoruz.
    ambiguous = (
        top2
        >=
        (top1 * 0.75)
    )

    ambiguous_ratio = float(
        np.mean(ambiguous)
    )


    # Normalize edilmiş Partition Entropy
    # (Bölüm Entropisi)
    #
    # 0'a yakın -> çok keskin üyelik
    # 1'e yakın -> çok bulanık üyelik

    eps = 1e-12

    entropy = -np.sum(
        u * np.log(u + eps)
    ) / u.shape[1]

    normalized_entropy = (
        entropy
        /
        np.log(u.shape[0])
    )


    return (
        mean_top1,
        mean_top2,
        ambiguous_ratio,
        float(normalized_entropy)
    )


# ============================================================
# 7. SONUÇLAR
# ============================================================

results = []


# ============================================================
# 8. MODELLER
# ============================================================

for model_name, file_name in EMBEDDING_FILES.items():

    print("\n")
    print("=" * 85)
    print("MODEL:", model_name)
    print("=" * 85)


    embeddings = np.load(
        file_name
    ).astype(np.float32)


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


    # --------------------------------------------------------
    # MAKALE SEVİYESİ
    # --------------------------------------------------------

    article_ids, X = (
        build_article_embeddings(
            embeddings
        )
    )

    print(
        "Kullanılan makale:",
        len(article_ids)
    )


    # scikit-fuzzy:
    # feature x sample formatı istiyor.
    fuzzy_data = X.T


    # ========================================================
    # 9. FARKLI K DEĞERLERİ
    # ========================================================

    for k in K_VALUES:

        print("\n")
        print("-" * 70)

        print(
            f"Fuzzy C-Means "
            f"(Bulanık C-Ortalamalar) | K={k}"
        )

        print("-" * 70)


        (
            centers,
            u,
            u0,
            distances,
            objective_history,
            iterations,
            fpc
        ) = fuzz.cluster.cmeans(

            data=fuzzy_data,

            c=k,

            m=FUZZINESS_M,

            error=ERROR,

            maxiter=MAX_ITER,

            init=None,

            seed=RANDOM_STATE
        )


        # ----------------------------------------------------
        # HARD LABEL
        # (EN GÜÇLÜ ÜYELİĞE GÖRE TEK KÜME)
        # ----------------------------------------------------

        hard_labels = np.argmax(
            u,
            axis=0
        )


        # ----------------------------------------------------
        # SILHOUETTE
        # ----------------------------------------------------

        unique_labels = np.unique(
            hard_labels
        )

        if len(unique_labels) >= 2:

            silhouette = silhouette_score(
                X,
                hard_labels,
                metric="cosine"
            )

        else:

            silhouette = np.nan


        # ----------------------------------------------------
        # PURITY
        # ----------------------------------------------------

        (
            weighted_purity,
            singleton_clusters
        ) = calculate_multilabel_purity(
            article_ids,
            hard_labels
        )


        # ----------------------------------------------------
        # KÜME BOYUTLARI
        # ----------------------------------------------------

        _, counts = np.unique(
            hard_labels,
            return_counts=True
        )

        smallest_cluster = int(
            counts.min()
        )

        largest_cluster = int(
            counts.max()
        )


        # ----------------------------------------------------
        # FUZZY MEMBERSHIP
        # (BULANIK ÜYELİK)
        # ----------------------------------------------------

        (
            mean_top1,
            mean_top2,
            ambiguous_ratio,
            normalized_entropy
        ) = analyze_memberships(
            u
        )


        # ----------------------------------------------------
        # EKRANA YAZ
        # ----------------------------------------------------

        print(
            f"FPC "
            f"(Bulanık Bölüm Katsayısı): "
            f"{fpc:.4f}"
        )

        print(
            f"Silhouette "
            f"(Silüet skoru): "
            f"{silhouette:.4f}"
        )

        print(
            f"Weighted Purity "
            f"(Ağırlıklı saflık): "
            f"{weighted_purity:.4f}"
        )

        print(
            f"Ortalama Top-1 Membership "
            f"(birinci üyelik): "
            f"{mean_top1:.4f}"
        )

        print(
            f"Ortalama Top-2 Membership "
            f"(ikinci üyelik): "
            f"{mean_top2:.4f}"
        )

        print(
            f"Çoklu kümeye yakın makale oranı: "
            f"{ambiguous_ratio:.2%}"
        )

        print(
            f"Normalized Partition Entropy "
            f"(normalize bölüm entropisi): "
            f"{normalized_entropy:.4f}"
        )

        print(
            "Singleton Cluster "
            "(tek elemanlı küme):",
            singleton_clusters
        )

        print(
            "En büyük küme:",
            largest_cluster
        )

        print(
            "İterasyon:",
            iterations
        )


        # ----------------------------------------------------
        # SONUÇ KAYDET
        # ----------------------------------------------------

        results.append({

            "Model":
                model_name,

            "K":
                k,

            "FPC":
                round(
                    float(fpc),
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
                ),

            "Weighted_Purity":
                round(
                    float(weighted_purity),
                    4
                ),

            "Mean_Top1_Membership":
                round(
                    mean_top1,
                    4
                ),

            "Mean_Top2_Membership":
                round(
                    mean_top2,
                    4
                ),

            "Ambiguous_Ratio":
                round(
                    ambiguous_ratio,
                    4
                ),

            "Normalized_Entropy":
                round(
                    normalized_entropy,
                    4
                ),

            "Singleton_Clusters":
                singleton_clusters,

            "Smallest_Cluster":
                smallest_cluster,

            "Largest_Cluster":
                largest_cluster,

            "Iterations":
                iterations
        })


# ============================================================
# 10. SONUÇ TABLOSU
# ============================================================

results_df = pd.DataFrame(
    results
)

print("\n\n")
print("=" * 125)

print(
    "FUZZY C-MEANS "
    "(BULANIK C-ORTALAMALAR) "
    "KARŞILAŞTIRMASI"
)

print("=" * 125)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 11. K=195 ÖZEL KARŞILAŞTIRMA
# ============================================================

k195 = results_df[
    results_df["K"] == 195
]

print("\n\n")
print("=" * 115)

print(
    "K=195 FUZZY C-MEANS "
    "(BULANIK C-ORTALAMALAR) "
    "ÖZEL KARŞILAŞTIRMA"
)

print("=" * 115)

print(
    k195[
        [
            "Model",
            "FPC",
            "Silhouette",
            "Weighted_Purity",
            "Mean_Top1_Membership",
            "Mean_Top2_Membership",
            "Ambiguous_Ratio",
            "Singleton_Clusters"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 12. CSV KAYDET
# ============================================================

results_df.to_csv(
    "fuzzy_cmeans_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nSonuç dosyası oluşturuldu: "
    "fuzzy_cmeans_results.csv"
)