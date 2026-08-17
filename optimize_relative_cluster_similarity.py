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
# SABİT AYARLAR
# ============================================================

CLUSTER_MARGIN = 0.15
SUBJECT_THRESHOLD = 0.225
MIN_SUPPORT = 2
MIN_CLUSTER_SIMILARITY = 0.45


# ============================================================
# TEST EDİLECEK GÖRELİ BENZERLİKLER
# ============================================================

RELATIVE_SIMILARITIES = [
    0.75,
    0.80,
    0.85,
    0.90,
    0.95
]


# ============================================================
# VERİLER
# ============================================================

texts = pd.read_csv(TEXT_FILE, encoding="utf-8-sig")
clusters = pd.read_csv(CLUSTER_FILE, encoding="utf-8-sig")
subjects = pd.read_csv(SUBJECT_FILE, encoding="utf-8-sig")

embeddings = np.load(
    EMBEDDING_FILE
).astype(np.float32)

centroids = np.load(
    CENTROID_FILE
).astype(np.float32)


print("=" * 110)
print("RELATIVE CLUSTER SIMILARITY OPTIMIZASYONU")
print("=" * 110)

print("Metin satırı:", len(texts))
print("Embedding:", embeddings.shape)
print("Cluster makalesi:", len(clusters))
print("Centroid:", centroids.shape)


# ============================================================
# EN-ALT KONULAR
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

leaf_subjects = subjects[
    (subjects["leaf_subject"] != "")
    &
    (subjects["subject_fullname"] != "")
].copy()


article_subjects = (
    leaf_subjects
    .groupby("article_id")["subject_fullname"]
    .apply(set)
    .to_dict()
)


# ============================================================
# MAKALE EMBEDDINGLERİ
# ============================================================

article_vectors = {}

for article_id, group in texts.groupby("article_id"):

    indices = group.index.to_numpy()

    vector = embeddings[
        indices
    ].mean(axis=0)

    norm = np.linalg.norm(vector)

    if norm > 0:
        vector = vector / norm

    article_vectors[article_id] = vector


# ============================================================
# CENTROID NORMALIZATION
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
    centroids / centroid_norms
)


# ============================================================
# CLUSTER -> MAKALELER
# ============================================================

cluster_members = (
    clusters
    .groupby("cluster_id")["article_id"]
    .apply(list)
    .to_dict()
)


# ============================================================
# SIMILARITY CACHE
# ============================================================

similarity_cache = {}

for article_id, vector in article_vectors.items():

    similarity_cache[article_id] = cosine_similarity(
        vector.reshape(1, -1),
        normalized_centroids
    )[0]


# ============================================================
# TESTLER
# ============================================================

results = []


for relative_similarity in RELATIVE_SIMILARITIES:

    tp = 0
    fp = 0
    fn = 0

    hit_count = 0
    exact_count = 0

    predicted_counts = []
    selected_cluster_counts = []

    evaluated = 0


    for article_id, real_subjects in article_subjects.items():

        if article_id not in article_vectors:
            continue


        similarities = similarity_cache[
            article_id
        ]

        best_similarity = float(
            similarities.max()
        )


        # ====================================================
        # 3 ŞARTLI CLUSTER SEÇİMİ
        # ====================================================
        #
        # 1) Best - margin sınırı
        # 2) Minimum mutlak similarity
        # 3) Best similarity'nin belirli oranı
        # ====================================================

        relative_limit = (
            best_similarity
            *
            relative_similarity
        )


        selected_clusters = np.where(

            (
                similarities
                >=
                best_similarity - CLUSTER_MARGIN
            )

            &

            (
                similarities
                >=
                MIN_CLUSTER_SIMILARITY
            )

            &

            (
                similarities
                >=
                relative_limit
            )

        )[0]


        selected_cluster_counts.append(
            len(selected_clusters)
        )


        # ====================================================
        # KONU SKORLARI
        # ====================================================

        subject_scores = {}
        subject_support_articles = {}

        total_cluster_weight = 0.0


        for cluster_id in selected_clusters:

            members = cluster_members.get(
                int(cluster_id),
                []
            )


            # Leave-one-out
            other_members = [
                member_id
                for member_id in members
                if member_id != article_id
            ]


            if not other_members:
                continue


            similarity = float(
                similarities[cluster_id]
            )


            cluster_weight = (
                similarity / best_similarity
                if best_similarity > 0
                else 0
            )


            total_cluster_weight += (
                cluster_weight
            )


            subject_counts = {}


            for other_id in other_members:

                labels = article_subjects.get(
                    other_id,
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


                    if label not in subject_support_articles:
                        subject_support_articles[label] = set()


                    subject_support_articles[
                        label
                    ].add(
                        other_id
                    )


            for subject, count in subject_counts.items():

                support_ratio = (
                    count
                    /
                    len(other_members)
                )


                evidence = (
                    support_ratio
                    *
                    cluster_weight
                )


                subject_scores[subject] = (
                    subject_scores.get(
                        subject,
                        0
                    )
                    +
                    evidence
                )


        # ====================================================
        # NORMALIZATION
        # ====================================================

        if total_cluster_weight > 0:

            subject_scores = {
                subject:
                score / total_cluster_weight

                for subject, score
                in subject_scores.items()
            }


        # ====================================================
        # KONU SEÇİMİ
        # ====================================================

        predicted_subjects = set()


        for subject, score in subject_scores.items():

            support_count = len(
                subject_support_articles.get(
                    subject,
                    set()
                )
            )


            if (
                score >= SUBJECT_THRESHOLD
                and
                support_count >= MIN_SUPPORT
            ):

                predicted_subjects.add(
                    subject
                )


        predicted_counts.append(
            len(predicted_subjects)
        )

        evaluated += 1


        # ====================================================
        # METRİKLER
        # ====================================================

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


        if predicted_subjects & real_subjects:
            hit_count += 1


        if predicted_subjects == real_subjects:
            exact_count += 1


    # ========================================================
    # SONUÇLAR
    # ========================================================

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
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0
    )

    hit_rate = (
        hit_count / evaluated
        if evaluated
        else 0
    )

    exact_match = (
        exact_count / evaluated
        if evaluated
        else 0
    )

    avg_labels = (
        np.mean(predicted_counts)
        if predicted_counts
        else 0
    )

    avg_clusters = (
        np.mean(selected_cluster_counts)
        if selected_cluster_counts
        else 0
    )

    max_clusters = (
        np.max(selected_cluster_counts)
        if selected_cluster_counts
        else 0
    )

    no_prediction_ratio = (
        np.mean(
            np.array(predicted_counts) == 0
        )
        if predicted_counts
        else 0
    )


    results.append({

        "Relative_Similarity":
            relative_similarity,

        "Precision":
            round(precision, 4),

        "Recall":
            round(recall, 4),

        "F1":
            round(f1, 4),

        "Hit_Rate":
            round(hit_rate, 4),

        "Exact_Match":
            round(exact_match, 4),

        "Avg_Labels":
            round(avg_labels, 2),

        "Avg_Selected_Clusters":
            round(avg_clusters, 2),

        "Max_Selected_Clusters":
            int(max_clusters),

        "No_Prediction_Ratio":
            round(no_prediction_ratio, 4),

        "TP": tp,
        "FP": fp,
        "FN": fn,

        "Evaluated":
            evaluated
    })


# ============================================================
# SONUÇ TABLOSU
# ============================================================

result_df = pd.DataFrame(
    results
)


print("\n" + "=" * 125)
print("RELATIVE SIMILARITY SONUÇLARI")
print("=" * 125)

print(
    result_df.to_string(
        index=False
    )
)


# ============================================================
# F1'E GÖRE EN İYİ
# ============================================================

best = result_df.loc[
    result_df["F1"].idxmax()
]


print("\n" + "=" * 125)
print("F1'E GÖRE EN İYİ RELATIVE SIMILARITY")
print("=" * 125)

print(
    best.to_string()
)


# ============================================================
# CSV
# ============================================================

result_df.to_csv(
    "relative_cluster_similarity_results.csv",
    index=False,
    encoding="utf-8-sig"
)


print(
    "\nDosya oluşturuldu:"
    " relative_cluster_similarity_results.csv"
)