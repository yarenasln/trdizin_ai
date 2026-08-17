import numpy as np
import pandas as pd
import umap.umap_ as umap

from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# DOSYALAR
# ============================================================

TEXT_FILE = "real_trdizin_texts.csv"
EMBEDDING_FILE = "embeddings/Qwen3_embeddings.npy"
SUBJECT_FILE = "trdizin_subject_hierarchy.csv"


# ============================================================
# UMAP + K-MEANS AYARLARI
# ============================================================

K = 195

UMAP_COMPONENTS = 25
UMAP_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1

RANDOM_STATE = 42


# ============================================================
# DİNAMİK KONU TAHMİN AYARLARI
# ============================================================

CLUSTER_MARGIN = 0.15
MIN_CLUSTER_SIMILARITY = 0.45
RELATIVE_SIMILARITY = 0.80

SUBJECT_THRESHOLD = 0.225
MIN_SUPPORT = 2


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

row_embeddings = np.load(
    EMBEDDING_FILE
).astype(np.float32)


print("=" * 110)
print("QWEN3 + UMAP25 + K-MEANS DİNAMİK KONU TAHMİN TESTİ")
print("=" * 110)

print("Metin satırı:", len(texts))
print("Orijinal embedding:", row_embeddings.shape)


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
# ROW LEVEL -> ARTICLE LEVEL EMBEDDING
# ============================================================

article_ids = []
article_vectors = []

for article_id, group in texts.groupby("article_id"):

    indices = group.index.to_numpy()

    vector = row_embeddings[
        indices
    ].mean(axis=0)

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


X_original = np.vstack(
    article_vectors
).astype(np.float32)

article_ids = np.array(
    article_ids
)


print(
    "Makale sayısı:",
    len(article_ids)
)

print(
    "Article embedding:",
    X_original.shape
)


# ============================================================
# UMAP 1024 -> 25
# ============================================================

print("\nUMAP 25 çalışıyor...")


reducer = umap.UMAP(
    n_components=UMAP_COMPONENTS,
    n_neighbors=UMAP_NEIGHBORS,
    min_dist=UMAP_MIN_DIST,
    metric="cosine",
    random_state=RANDOM_STATE
)


X_umap = reducer.fit_transform(
    X_original
).astype(np.float32)


print(
    "UMAP sonrası:",
    X_umap.shape
)


# ============================================================
# UMAP VEKTÖRLERİNİ NORMALIZE ET
# ============================================================

umap_norms = np.linalg.norm(
    X_umap,
    axis=1,
    keepdims=True
)

umap_norms[
    umap_norms == 0
] = 1

X_umap_normalized = (
    X_umap / umap_norms
)


# ============================================================
# K-MEANS
# ============================================================

print("\nK-Means çalışıyor...")


kmeans = KMeans(
    n_clusters=K,
    init="k-means++",
    n_init=10,
    max_iter=300,
    random_state=RANDOM_STATE
)


cluster_labels = kmeans.fit_predict(
    X_umap
)


centroids = kmeans.cluster_centers_.astype(
    np.float32
)


# ============================================================
# CENTROID NORMALIZE
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
# ARTICLE ID -> INDEX
# ============================================================

article_index = {
    article_id: index
    for index, article_id
    in enumerate(article_ids)
}


# ============================================================
# CLUSTER -> MAKALELER
# ============================================================

cluster_members = {}

for article_id, cluster_id in zip(
    article_ids,
    cluster_labels
):

    cluster_members.setdefault(
        int(cluster_id),
        []
    ).append(
        article_id
    )


# ============================================================
# DEĞERLENDİRME
# ============================================================

tp = 0
fp = 0
fn = 0

hit_count = 0
exact_count = 0

predicted_counts = []
selected_cluster_counts = []

evaluated = 0


# ============================================================
# SADECE ETİKETİ OLAN MAKALELER
# ============================================================

for article_id, real_subjects in article_subjects.items():

    if article_id not in article_index:
        continue


    idx = article_index[
        article_id
    ]


    vector = X_umap_normalized[
        idx
    ]


    similarities = cosine_similarity(
        vector.reshape(1, -1),
        normalized_centroids
    )[0]


    best_similarity = float(
        similarities.max()
    )


    relative_limit = (
        best_similarity
        *
        RELATIVE_SIMILARITY
    )


    # ========================================================
    # DİNAMİK CLUSTER SEÇİMİ
    # ========================================================

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


    # ========================================================
    # KONU KANITI
    # ========================================================

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
            similarities[
                cluster_id
            ]
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

                subject_counts[
                    label
                ] = (
                    subject_counts.get(
                        label,
                        0
                    )
                    + 1
                )


                if label not in subject_support_articles:

                    subject_support_articles[
                        label
                    ] = set()


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


            subject_scores[
                subject
            ] = (
                subject_scores.get(
                    subject,
                    0
                )
                +
                evidence
            )


    # ========================================================
    # NORMALIZE KONU SKORLARI
    # ========================================================

    if total_cluster_weight > 0:

        subject_scores = {
            subject:
            score / total_cluster_weight

            for subject, score
            in subject_scores.items()
        }


    # ========================================================
    # DİNAMİK ETİKET SEÇİMİ
    # ========================================================

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


    # ========================================================
    # METRİKLER
    # ========================================================

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


# ============================================================
# SONUÇ HESAPLARI
# ============================================================

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
    if evaluated > 0
    else 0
)

exact_match = (
    exact_count / evaluated
    if evaluated > 0
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
        np.array(
            predicted_counts
        ) == 0
    )
    if predicted_counts
    else 0
)


# ============================================================
# SONUÇ
# ============================================================

print("\n" + "=" * 120)
print("UMAP25 + K-MEANS KONU TAHMİN SONUCU")
print("=" * 120)


print(
    "Precision:",
    round(
        precision,
        4
    )
)

print(
    "Recall:",
    round(
        recall,
        4
    )
)

print(
    "F1:",
    round(
        f1,
        4
    )
)

print(
    "Hit Rate:",
    round(
        hit_rate,
        4
    )
)

print(
    "Exact Match:",
    round(
        exact_match,
        4
    )
)

print(
    "Ortalama tahmin edilen etiket:",
    round(
        avg_labels,
        2
    )
)

print(
    "Ortalama seçilen cluster:",
    round(
        avg_clusters,
        2
    )
)

print(
    "Maksimum seçilen cluster:",
    int(
        max_clusters
    )
)

print(
    "Tahmin yok oranı:",
    round(
        no_prediction_ratio,
        4
    )
)

print(
    "TP:",
    tp
)

print(
    "FP:",
    fp
)

print(
    "FN:",
    fn
)

print(
    "Değerlendirilen makale:",
    evaluated
)


# ============================================================
# MEVCUT 1024D SİSTEMLE KARŞILAŞTIR
# ============================================================

BASELINE_PRECISION = 0.4078
BASELINE_RECALL = 0.3490
BASELINE_F1 = 0.3762
BASELINE_HIT_RATE = 0.6141


print("\n" + "=" * 120)
print("1024D K-MEANS VS UMAP25 + K-MEANS")
print("=" * 120)


print(
    f"Precision: "
    f"{BASELINE_PRECISION:.4f}"
    f" -> {precision:.4f}"
)

print(
    f"Recall: "
    f"{BASELINE_RECALL:.4f}"
    f" -> {recall:.4f}"
)

print(
    f"F1: "
    f"{BASELINE_F1:.4f}"
    f" -> {f1:.4f}"
)

print(
    f"Hit Rate: "
    f"{BASELINE_HIT_RATE:.4f}"
    f" -> {hit_rate:.4f}"
)


# ============================================================
# DOSYALARI KAYDET
# ============================================================

np.save(
    "umap25_qwen3_embeddings.npy",
    X_umap
)

np.save(
    "umap25_kmeans_centroids.npy",
    centroids
)


pd.DataFrame({

    "article_id":
        article_ids,

    "cluster_id":
        cluster_labels

}).to_csv(

    "umap25_kmeans_article_clusters.csv",

    index=False,

    encoding="utf-8-sig"
)


summary = pd.DataFrame([
    {
        "Model":
            "Qwen3",

        "UMAP_Dimension":
            25,

        "K":
            K,

        "Cluster_Margin":
            CLUSTER_MARGIN,

        "Min_Cluster_Similarity":
            MIN_CLUSTER_SIMILARITY,

        "Relative_Similarity":
            RELATIVE_SIMILARITY,

        "Subject_Threshold":
            SUBJECT_THRESHOLD,

        "Min_Support":
            MIN_SUPPORT,

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

        "Max_Selected_Clusters":
            int(
                max_clusters
            ),

        "No_Prediction_Ratio":
            round(
                no_prediction_ratio,
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
    }
])


summary.to_csv(
    "umap25_topic_prediction_results.csv",
    index=False,
    encoding="utf-8-sig"
)


print(
    "\nDosyalar oluşturuldu:"
)

print(
    "umap25_qwen3_embeddings.npy"
)

print(
    "umap25_kmeans_centroids.npy"
)

print(
    "umap25_kmeans_article_clusters.csv"
)

print(
    "umap25_topic_prediction_results.csv"
)