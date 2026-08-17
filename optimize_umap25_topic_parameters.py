import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


# ============================================================
# DOSYALAR
# ============================================================

TEXT_FILE = "real_trdizin_texts.csv"
UMAP_FILE = "umap25_qwen3_embeddings.npy"
CLUSTER_FILE = "umap25_kmeans_article_clusters.csv"
CENTROID_FILE = "umap25_kmeans_centroids.npy"
SUBJECT_FILE = "trdizin_subject_hierarchy.csv"


# ============================================================
# TEST EDİLECEK PARAMETRELER
# ============================================================

DISTANCE_MARGINS = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40
]

SUBJECT_THRESHOLDS = [
    0.15,
    0.175,
    0.20,
    0.225,
    0.25,
    0.275,
    0.30
]

MIN_SUPPORT = 2


# ============================================================
# VERİLERİ YÜKLE
# ============================================================

texts = pd.read_csv(
    TEXT_FILE,
    encoding="utf-8-sig"
)

X = np.load(
    UMAP_FILE
).astype(np.float32)

centroids = np.load(
    CENTROID_FILE
).astype(np.float32)

clusters = pd.read_csv(
    CLUSTER_FILE,
    encoding="utf-8-sig"
)

subjects = pd.read_csv(
    SUBJECT_FILE,
    encoding="utf-8-sig"
)


print("=" * 120)
print("UMAP25 + K-MEANS EUCLIDEAN PARAMETRE OPTİMİZASYONU")
print("=" * 120)

print("UMAP embedding:", X.shape)
print("Centroid:", centroids.shape)
print("Cluster makalesi:", len(clusters))


# ============================================================
# EN ALT KONU ETİKETLERİ
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

subjects = subjects[
    (subjects["leaf_subject"] != "")
    &
    (subjects["subject_fullname"] != "")
].copy()


article_subjects = (
    subjects
    .groupby("article_id")["subject_fullname"]
    .apply(set)
    .to_dict()
)


# ============================================================
# ARTICLE ID SIRASI
# ============================================================

article_ids = (
    clusters["article_id"]
    .to_numpy()
)

cluster_labels = (
    clusters["cluster_id"]
    .to_numpy()
)


if len(article_ids) != len(X):
    raise ValueError(
        "Embedding ve cluster makale sayıları uyuşmuyor."
    )


article_index = {
    article_id: i
    for i, article_id
    in enumerate(article_ids)
}


# ============================================================
# CLUSTER -> ARTICLE ID
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
# DISTANCE MATRİSİNİ BİR KEZ HESAPLA
# ============================================================

print("\nEuclidean distance matrisi hesaplanıyor...")

distance_matrix = pairwise_distances(
    X,
    centroids,
    metric="euclidean"
)

best_distances = distance_matrix.min(
    axis=1
)

print(
    "Distance matrisi:",
    distance_matrix.shape
)


# ============================================================
# DEĞERLENDİRİLECEK MAKALELER
# ============================================================

evaluated_article_ids = [
    article_id
    for article_id in article_ids
    if article_id in article_subjects
]

print(
    "Etiketli/değerlendirilecek makale:",
    len(evaluated_article_ids)
)


# ============================================================
# TEK BİR PARAMETRE KOMBİNASYONUNU TEST ET
# ============================================================

def evaluate(distance_margin, subject_threshold):

    tp = 0
    fp = 0
    fn = 0

    hit = 0
    exact = 0

    predicted_label_counts = []
    selected_cluster_counts = []

    for article_id in evaluated_article_ids:

        idx = article_index[
            article_id
        ]

        real_labels = article_subjects[
            article_id
        ]

        distances = distance_matrix[
            idx
        ]

        best_distance = best_distances[
            idx
        ]


        # ====================================================
        # DİNAMİK CLUSTER SEÇ
        #
        # En yakın centroid + belirlenen margin içerisinde
        # kalan bütün clusterlar.
        # ====================================================

        selected_clusters = np.where(
            distances
            <=
            best_distance + distance_margin
        )[0]


        selected_cluster_counts.append(
            len(selected_clusters)
        )


        # ====================================================
        # KONU KANITI
        # ====================================================

        subject_weighted_score = {}
        subject_support_articles = {}

        total_cluster_weight = 0.0


        for cluster_id in selected_clusters:

            members = cluster_members.get(
                int(cluster_id),
                []
            )


            # Leave-one-out:
            # Test edilen makalenin kendi etiketini
            # tahminde kullanmıyoruz.
            other_members = [
                member_id
                for member_id in members
                if member_id != article_id
            ]


            if not other_members:
                continue


            cluster_distance = distances[
                cluster_id
            ]


            # =================================================
            # DISTANCE -> WEIGHT
            #
            # Yakın cluster daha yüksek ağırlık alır.
            # =================================================

            weight = 1.0 / (
                cluster_distance + 1e-6
            )


            total_cluster_weight += weight


            cluster_subject_counts = {}


            for other_id in other_members:

                labels = article_subjects.get(
                    other_id,
                    set()
                )


                for label in labels:

                    cluster_subject_counts[
                        label
                    ] = (
                        cluster_subject_counts.get(
                            label,
                            0
                        )
                        + 1
                    )


                    subject_support_articles.setdefault(
                        label,
                        set()
                    ).add(
                        other_id
                    )


            # =================================================
            # CLUSTER İÇİ KONU ORANI
            # =================================================

            for label, count in cluster_subject_counts.items():

                ratio = (
                    count
                    /
                    len(other_members)
                )


                evidence = (
                    ratio
                    *
                    weight
                )


                subject_weighted_score[
                    label
                ] = (
                    subject_weighted_score.get(
                        label,
                        0.0
                    )
                    +
                    evidence
                )


        # ====================================================
        # SKOR NORMALİZASYONU
        # ====================================================

        if total_cluster_weight > 0:

            subject_weighted_score = {
                label:
                score / total_cluster_weight

                for label, score
                in subject_weighted_score.items()
            }


        # ====================================================
        # DİNAMİK KONU SEÇİMİ
        # ====================================================

        predicted_labels = set()


        for label, score in subject_weighted_score.items():

            support = len(
                subject_support_articles.get(
                    label,
                    set()
                )
            )


            if (
                score >= subject_threshold
                and
                support >= MIN_SUPPORT
            ):

                predicted_labels.add(
                    label
                )


        predicted_label_counts.append(
            len(predicted_labels)
        )


        # ====================================================
        # METRİKLER
        # ====================================================

        tp += len(
            predicted_labels
            &
            real_labels
        )

        fp += len(
            predicted_labels
            -
            real_labels
        )

        fn += len(
            real_labels
            -
            predicted_labels
        )


        if predicted_labels & real_labels:
            hit += 1


        if predicted_labels == real_labels:
            exact += 1


    evaluated = len(
        evaluated_article_ids
    )


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
        /
        (precision + recall)
        if precision + recall > 0
        else 0
    )

    hit_rate = (
        hit / evaluated
        if evaluated
        else 0
    )

    exact_match = (
        exact / evaluated
        if evaluated
        else 0
    )

    avg_labels = (
        np.mean(
            predicted_label_counts
        )
        if predicted_label_counts
        else 0
    )

    no_prediction = (
        np.mean(
            np.array(
                predicted_label_counts
            ) == 0
        )
        if predicted_label_counts
        else 0
    )

    avg_clusters = (
        np.mean(
            selected_cluster_counts
        )
        if selected_cluster_counts
        else 0
    )

    max_clusters = (
        np.max(
            selected_cluster_counts
        )
        if selected_cluster_counts
        else 0
    )


    return {
        "Distance_Margin":
            distance_margin,

        "Subject_Threshold":
            subject_threshold,

        "Min_Support":
            MIN_SUPPORT,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1":
            f1,

        "Hit_Rate":
            hit_rate,

        "Exact_Match":
            exact_match,

        "Avg_Labels":
            avg_labels,

        "Avg_Selected_Clusters":
            avg_clusters,

        "Max_Selected_Clusters":
            max_clusters,

        "No_Prediction_Ratio":
            no_prediction,

        "TP":
            tp,

        "FP":
            fp,

        "FN":
            fn,

        "Evaluated":
            evaluated
    }


# ============================================================
# TÜM KOMBİNASYONLAR
# ============================================================

results = []

total_tests = (
    len(DISTANCE_MARGINS)
    *
    len(SUBJECT_THRESHOLDS)
)

current_test = 0


for margin in DISTANCE_MARGINS:

    for threshold in SUBJECT_THRESHOLDS:

        current_test += 1

        print(
            f"Test {current_test}/{total_tests} | "
            f"margin={margin:.3f} | "
            f"threshold={threshold:.3f}"
        )

        result = evaluate(
            margin,
            threshold
        )

        results.append(
            result
        )


# ============================================================
# SONUÇ TABLOSU
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df = results_df.sort_values(
    "F1",
    ascending=False
).reset_index(
    drop=True
)


pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    250
)


# ============================================================
# HER MARGIN İÇİN EN İYİ THRESHOLD
# ============================================================

best_per_margin = (
    results_df
    .sort_values(
        "F1",
        ascending=False
    )
    .groupby(
        "Distance_Margin",
        as_index=False
    )
    .first()
    .sort_values(
        "F1",
        ascending=False
    )
)


print("\n" + "=" * 150)
print("HER DISTANCE MARGIN İÇİN EN İYİ SONUÇ")
print("=" * 150)

print(
    best_per_margin.to_string(
        index=False
    )
)


# ============================================================
# GENEL EN İYİ
# ============================================================

best = results_df.iloc[0]


print("\n" + "=" * 120)
print("GENEL EN İYİ UMAP25 + K-MEANS SONUCU")
print("=" * 120)

print(
    best.to_string()
)


# ============================================================
# EN İYİ 15
# ============================================================

print("\n" + "=" * 150)
print("EN İYİ 15 KOMBİNASYON")
print("=" * 150)

print(
    results_df.head(
        15
    ).to_string(
        index=False
    )
)


# ============================================================
# 1024D BASELINE KARŞILAŞTIRMASI
# ============================================================

BASELINE_PRECISION = 0.4078
BASELINE_RECALL = 0.3490
BASELINE_F1 = 0.3762
BASELINE_HIT_RATE = 0.6141


print("\n" + "=" * 120)
print("MEVCUT 1024D K-MEANS İLE KARŞILAŞTIRMA")
print("=" * 120)

print(
    f"Precision: "
    f"{BASELINE_PRECISION:.4f}"
    f" -> {best['Precision']:.4f}"
)

print(
    f"Recall: "
    f"{BASELINE_RECALL:.4f}"
    f" -> {best['Recall']:.4f}"
)

print(
    f"F1: "
    f"{BASELINE_F1:.4f}"
    f" -> {best['F1']:.4f}"
)

print(
    f"Hit Rate: "
    f"{BASELINE_HIT_RATE:.4f}"
    f" -> {best['Hit_Rate']:.4f}"
)


# ============================================================
# CSV
# ============================================================

results_df.to_csv(
    "umap25_euclidean_parameter_results.csv",
    index=False,
    encoding="utf-8-sig"
)


print(
    "\nDosya oluşturuldu: "
    "umap25_euclidean_parameter_results.csv"
)