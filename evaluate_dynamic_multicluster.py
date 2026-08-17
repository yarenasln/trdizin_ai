import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# DOSYALAR
# ============================================================

TEXT_FILE = "real_trdizin_texts.csv"
EMBEDDING_FILE = "embeddings/Qwen3_embeddings.npy"

CLUSTER_FILE = "kmeans_article_clusters.csv"
CENTROID_FILE = "kmeans_centroids.npy"
SUBJECT_FILE = "trdizin_subject_hierarchy.csv"


# ============================================================
# DENEYECEĞİMİZ DEĞERLER
# ============================================================

# En iyi centroid'e göre izin verilen similarity farkı

CLUSTER_MARGINS = [
    0.00,
    0.05,
    0.08,
    0.10,
    0.12,
    0.15,
    0.18,
    0.20,
    0.25
]
# Konu kabul eşikleri

SUBJECT_THRESHOLDS = [
    0.15,
    0.20,
    0.225,
    0.25,
    0.275,
    0.30,
    0.325,
    0.35
]

# ============================================================
# VERİLERİ OKU
# ============================================================

texts = pd.read_csv(
    TEXT_FILE,
    encoding="utf-8-sig"
)

clusters = pd.read_csv(
    CLUSTER_FILE,
    encoding="utf-8-sig"
)

subjects = pd.read_csv(
    SUBJECT_FILE,
    encoding="utf-8-sig"
)

embeddings = np.load(
    EMBEDDING_FILE
).astype(np.float32)

centroids = np.load(
    CENTROID_FILE
).astype(np.float32)


print("Metin satırı:", len(texts))
print("Embedding shape:", embeddings.shape)
print("Cluster makalesi:", len(clusters))
print("Centroid shape:", centroids.shape)


# ============================================================
# SADECE GERÇEK EN-ALT KONULARI KULLAN
# ============================================================

subjects["leaf_subject"] = (
    subjects["leaf_subject"]
    .fillna("")
    .astype(str)
    .str.strip()
)

subjects = subjects[
    subjects["leaf_subject"] != ""
].copy()


subjects["subject_fullname"] = (
    subjects["subject_fullname"]
    .fillna("")
    .astype(str)
    .str.strip()
)

subjects = subjects[
    subjects["subject_fullname"] != ""
].copy()


print(
    "En-alt konu satırı:",
    len(subjects)
)


# ============================================================
# MAKALE SEVİYESİNDE EMBEDDING
# ============================================================
#
# Aynı article_id birden fazla metin satırına sahipse
# embeddinglerin ortalamasını alıyoruz.
# ============================================================

article_vectors = {}

for article_id, group in texts.groupby("article_id"):

    indices = group.index.to_numpy()

    vectors = embeddings[
        indices
    ]

    vector = vectors.mean(
        axis=0
    )

    # Cosine similarity için normalize
    norm = np.linalg.norm(
        vector
    )

    if norm > 0:
        vector = vector / norm

    article_vectors[
        article_id
    ] = vector


print(
    "Makale embedding sayısı:",
    len(article_vectors)
)


# ============================================================
# CENTROIDLERİ NORMALIZE ET
# ============================================================

centroid_norms = np.linalg.norm(
    centroids,
    axis=1,
    keepdims=True
)

centroid_norms[
    centroid_norms == 0
] = 1

normalized_centroids = (
    centroids
    /
    centroid_norms
)


# ============================================================
# MAKALE -> GERÇEK TR DİZİN EN-ALT KONULARI
# ============================================================

article_subjects = (
    subjects
    .groupby(
        "article_id"
    )["subject_fullname"]
    .apply(set)
    .to_dict()
)


print(
    "Etiketli makale sayısı:",
    len(article_subjects)
)


# ============================================================
# CLUSTER -> MAKALELER
# ============================================================

cluster_members = (
    clusters
    .groupby(
        "cluster_id"
    )["article_id"]
    .apply(list)
    .to_dict()
)


# ============================================================
# HER MAKALE -> TÜM CENTROID BENZERLİKLERİ
# ============================================================

similarity_cache = {}

for article_id, vector in article_vectors.items():

    similarities = cosine_similarity(
        vector.reshape(1, -1),
        normalized_centroids
    )[0]

    similarity_cache[
        article_id
    ] = similarities


# ============================================================
# DENEYLER
# ============================================================

results = []


for cluster_margin in CLUSTER_MARGINS:

    for subject_threshold in SUBJECT_THRESHOLDS:

        tp = 0
        fp = 0
        fn = 0

        hit = 0
        exact = 0

        predicted_counts = []
        selected_cluster_counts = []

        evaluated = 0


        # ====================================================
        # SADECE GERÇEK ETİKETİ OLAN MAKALELER
        # ====================================================

        for article_id, real_subjects in article_subjects.items():

            if article_id not in article_vectors:
                continue

            if article_id not in similarity_cache:
                continue


            similarities = similarity_cache[
                article_id
            ]


            # =================================================
            # EN İYİ CENTROID
            # =================================================

            best_similarity = float(
                similarities.max()
            )


            # =================================================
            # DİNAMİK CLUSTER SEÇİMİ
            # =================================================
            #
            # Sabit Top-2 / Top-3 YOK.
            #
            # En iyi cluster'a yeterince yakın olan
            # kaç cluster varsa seçiliyor.
            # =================================================

            selected_clusters = np.where(

                similarities
                >=
                best_similarity
                -
                cluster_margin

            )[0]


            selected_cluster_counts.append(
                len(selected_clusters)
            )


            # =================================================
            # KONU KANITLARI
            # =================================================

            subject_scores = {}

            total_cluster_weight = 0.0


            for cluster_id in selected_clusters:

                members = cluster_members.get(
                    int(cluster_id),
                    []
                )


                # =============================================
                # LEAVE-ONE-OUT
                # =============================================
                #
                # Makalenin kendi gerçek etiketi
                # kendi tahminini üretirken kullanılmıyor.
                # =============================================

                other_members = [

                    member_id

                    for member_id
                    in members

                    if member_id
                    != article_id
                ]


                if not other_members:
                    continue


                # =============================================
                # CLUSTER AĞIRLIĞI
                # =============================================

                similarity = float(
                    similarities[
                        cluster_id
                    ]
                )


                cluster_weight = (

                    similarity
                    /
                    best_similarity

                    if best_similarity > 0

                    else 0
                )


                total_cluster_weight += (
                    cluster_weight
                )


                # =============================================
                # CLUSTER İÇİNDEKİ KONU SAYILARI
                # =============================================

                subject_counts = {}


                for other_id in other_members:

                    labels = article_subjects.get(
                        other_id,
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

                            + 1
                        )


                # =============================================
                # KONU DESTEK ORANI
                # =============================================

                for subject, count in subject_counts.items():

                    support_ratio = (

                        count
                        /
                        len(other_members)
                    )


                    weighted_evidence = (

                        support_ratio
                        *
                        cluster_weight
                    )


                    subject_scores[
                        subject
                    ] = (

                        subject_scores.get(
                            subject,
                            0
                        )

                        +
                        weighted_evidence
                    )


            # =================================================
            # KONU SKORLARINI NORMALIZE ET
            # =================================================

            if total_cluster_weight > 0:

                subject_scores = {

                    subject:
                    score
                    /
                    total_cluster_weight

                    for subject, score
                    in subject_scores.items()
                }


            # =================================================
            # DİNAMİK KONU SEÇİMİ
            # =================================================
            #
            # TOP-3 YOK.
            #
            # Eşiği geçen kaç konu varsa
            # o kadar konu veriliyor.
            # =================================================

            predicted_subjects = {

                subject

                for subject, score
                in subject_scores.items()

                if score
                >=
                subject_threshold
            }


            predicted_counts.append(
                len(predicted_subjects)
            )


            evaluated += 1


            # =================================================
            # TP / FP / FN
            # =================================================

            tp += len(

                predicted_subjects
                &
                real_subjects

            )


            fp += len(

                predicted_subjects
                -
                real_subjects

            )


            fn += len(

                real_subjects
                -
                predicted_subjects

            )


            # En az bir doğru konu
            if predicted_subjects & real_subjects:

                hit += 1


            # Birebir aynı konu seti
            if predicted_subjects == real_subjects:

                exact += 1


        # =====================================================
        # METRİKLER
        # =====================================================

        precision = (

            tp / (tp + fp)

            if tp + fp > 0

            else 0
        )


        recall = (

            tp / (tp + fn)

            if tp + fn > 0

            else 0
        )


        f1 = (

            2
            *
            precision
            *
            recall
            /
            (precision + recall)

            if precision + recall > 0

            else 0
        )


        hit_rate = (

            hit / evaluated

            if evaluated > 0

            else 0
        )


        exact_match = (

            exact / evaluated

            if evaluated > 0

            else 0
        )


        avg_labels = (

            np.mean(
                predicted_counts
            )

            if predicted_counts

            else 0
        )


        avg_clusters = (

            np.mean(
                selected_cluster_counts
            )

            if selected_cluster_counts

            else 0
        )


        no_prediction = (

            np.mean(

                np.array(
                    predicted_counts
                )
                ==
                0
            )

            if predicted_counts

            else 0
        )


        results.append({

            "Cluster_Margin":
                cluster_margin,

            "Subject_Threshold":
                subject_threshold,

            "Precision":
                round(
                    precision,
                    4
                ),

            "Recall":
                round(
                    recall,
                    4
                ),

            "F1":
                round(
                    f1,
                    4
                ),

            "Hit_Rate":
                round(
                    hit_rate,
                    4
                ),

            "Exact_Match":
                round(
                    exact_match,
                    4
                ),

            "Avg_Labels":
                round(
                    avg_labels,
                    2
                ),

            "Avg_Selected_Clusters":
                round(
                    avg_clusters,
                    2
                ),

            "No_Prediction_Ratio":
                round(
                    no_prediction,
                    4
                ),

            "TP":
                tp,

            "FP":
                fp,

            "FN":
                fn,

            "Evaluated":
                evaluated
        })


# ============================================================
# SONUÇLAR
# ============================================================

result_df = pd.DataFrame(
    results
)


# ============================================================
# HER MARGIN İÇİN EN İYİ
# ============================================================

print("\n" + "=" * 120)

print(
    "HER CLUSTER MARGIN İÇİN EN İYİ SONUÇ"
)

print("=" * 120)


best_per_margin = (

    result_df

    .sort_values(
        "F1",
        ascending=False
    )

    .groupby(
        "Cluster_Margin",
        as_index=False
    )

    .first()

    .sort_values(
        "F1",
        ascending=False
    )
)


print(
    best_per_margin.to_string(
        index=False
    )
)


# ============================================================
# GENEL EN İYİ
# ============================================================

best = result_df.loc[
    result_df["F1"].idxmax()
]


print("\n" + "=" * 120)

print(
    "GENEL EN İYİ DİNAMİK MULTI-CLUSTER SONUCU"
)

print("=" * 120)


print(
    best.to_string()
)


# ============================================================
# EN İYİ 10
# ============================================================

print("\n" + "=" * 120)

print(
    "EN İYİ 10 KOMBİNASYON"
)

print("=" * 120)


print(

    result_df

    .sort_values(
        "F1",
        ascending=False
    )

    .head(10)

    .to_string(
        index=False
    )
)


# ============================================================
# DOSYAYA KAYDET
# ============================================================

result_df.to_csv(

    "dynamic_multicluster_results.csv",

    index=False,

    encoding="utf-8-sig"
)


print(
    "\nDosya oluşturuldu:"
    " dynamic_multicluster_results.csv"
)