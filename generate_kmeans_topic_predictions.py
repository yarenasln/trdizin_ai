import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# AYARLAR
# ============================================================

TEXT_FILE = "real_trdizin_texts.csv"
EMBEDDING_FILE = "embeddings/Qwen3_embeddings.npy"

CLUSTER_FILE = "kmeans_article_clusters.csv"
CENTROID_FILE = "kmeans_centroids.npy"
SUBJECT_FILE = "trdizin_subject_hierarchy.csv"

# Pilot ayarlar
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


print("=" * 100)
print("K-MEANS DİNAMİK KONU TAHMİN SİSTEMİ")
print("=" * 100)

print("Metin satırı:", len(texts))
print("Embedding:", embeddings.shape)
print("Cluster makalesi:", len(clusters))
print("Centroid:", centroids.shape)

print(
    "Ayarlar:",
    f"margin={CLUSTER_MARGIN}",
    f"min_cluster_similarity={MIN_CLUSTER_SIMILARITY}",
    f"relative_similarity={RELATIVE_SIMILARITY}",
    f"subject_threshold={SUBJECT_THRESHOLD}",
    f"min_support={MIN_SUPPORT}"
)


# ============================================================
# SADECE EN-ALT KONULAR
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


# ============================================================
# GERÇEK TR DİZİN ETİKETLERİ
# ============================================================

article_subjects = (
    leaf_subjects
    .groupby("article_id")["subject_fullname"]
    .apply(set)
    .to_dict()
)

print(
    "Etiketli makale:",
    len(article_subjects)
)


# ============================================================
# MAKALE SEVİYESİNDE EMBEDDING
# ============================================================

article_vectors = {}

for article_id, group in texts.groupby("article_id"):

    indices = group.index.to_numpy()

    vector = embeddings[
        indices
    ].mean(axis=0)

    norm = np.linalg.norm(
        vector
    )

    if norm > 0:
        vector = vector / norm

    article_vectors[
        article_id
    ] = vector


print(
    "Makale embedding:",
    len(article_vectors)
)


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
    centroids
    /
    centroid_norms
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
# BAŞLIKLAR
# ============================================================

title_column = None

for candidate in [
    "title",
    "article_title",
    "name"
]:
    if candidate in texts.columns:
        title_column = candidate
        break


article_titles = {}

if title_column:

    article_titles = (
        texts
        .groupby("article_id")[title_column]
        .first()
        .fillna("")
        .astype(str)
        .to_dict()
    )


# ============================================================
# ÇIKTI
# ============================================================

output_rows = []


# ============================================================
# HER MAKALE İÇİN
# ============================================================

for article_id, vector in article_vectors.items():

    similarities = cosine_similarity(
        vector.reshape(1, -1),
        normalized_centroids
    )[0]


    best_similarity = float(
        similarities.max()
    )

    best_cluster = int(
        np.argmax(similarities)
    )


    # ========================================================
    # DİNAMİK CLUSTER SEÇİMİ
    # ========================================================
    #
    # Cluster'ın seçilmesi için 3 şart:
    #
    # 1) En iyi cluster'a 0.15 margin içinde olması
    #
    # 2) Mutlak cosine similarity >= 0.45 olması
    #
    # 3) En iyi similarity'nin en az %80'i olması
    # ========================================================

    relative_limit = (
        best_similarity
        *
        RELATIVE_SIMILARITY
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


    # ========================================================
    # KONU KANITLARI
    # ========================================================

    subject_scores = {}

    # Bir konuyu kaç FARKLI makale destekliyor?
    subject_support_articles = {}

    total_cluster_weight = 0.0


    for cluster_id in selected_clusters:

        members = cluster_members.get(
            int(cluster_id),
            []
        )


        # ====================================================
        # LEAVE-ONE-OUT
        # ====================================================

        other_members = [
            member_id
            for member_id in members
            if member_id != article_id
        ]


        if not other_members:
            continue


        # ====================================================
        # CLUSTER AĞIRLIĞI
        # ====================================================

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


        # ====================================================
        # CLUSTER'DAKİ KONU SAYILARI
        # ====================================================

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


        # ====================================================
        # KONU KANITI
        # ====================================================

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
    # KONU SKORLARINI NORMALIZE ET
    # ========================================================

    if total_cluster_weight > 0:

        subject_scores = {

            subject:
            score
            /
            total_cluster_weight

            for subject, score
            in subject_scores.items()
        }


    # ========================================================
    # DİNAMİK KONU SEÇİMİ
    # ========================================================
    #
    # Top-3 / Top-5 YOK.
    #
    # İki koşulu geçen kaç konu varsa
    # o kadar etiket veriyoruz.
    # ========================================================

    predictions = []


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

            predictions.append(
                (
                    subject,
                    score,
                    support_count
                )
            )


    predictions.sort(
        key=lambda x: x[1],
        reverse=True
    )


    # ========================================================
    # TR DİZİN ETİKETLERİ
    # ========================================================

    real_subjects = article_subjects.get(
        article_id,
        set()
    )


    predicted_subjects = {
        subject
        for subject, _, _
        in predictions
    }


    matched = (
        predicted_subjects
        &
        real_subjects
    )


    missing = (
        real_subjects
        -
        predicted_subjects
    )


    additional = (
        predicted_subjects
        -
        real_subjects
    )


    # ========================================================
    # DURUM
    # ========================================================

    if not real_subjects:

        status = (
            "TR_DIZIN_ETIKETI_YOK"
        )


    elif (
        predicted_subjects
        ==
        real_subjects
    ):

        status = (
            "TAM_ESLESME"
        )


    elif matched:

        status = (
            "KISMI_ESLESME"
        )


    elif not predicted_subjects:

        status = (
            "TAHMIN_YOK"
        )


    else:

        status = (
            "UYUSMAZLIK"
        )


    # ========================================================
    # TAHMİN VAR
    # ========================================================

    if predictions:

        for rank, (
            subject,
            score,
            support
        ) in enumerate(
            predictions,
            start=1
        ):

            output_rows.append({

                "article_id":
                    article_id,

                "title":
                    article_titles.get(
                        article_id,
                        ""
                    ),

                "best_cluster":
                    best_cluster,

                "best_similarity":
                    round(
                        best_similarity,
                        4
                    ),

                "selected_cluster_count":
                    len(
                        selected_clusters
                    ),

                "prediction_rank":
                    rank,

                "predicted_subject":
                    subject,

                "prediction_score":
                    round(
                        score,
                        4
                    ),

                "support_count":
                    support,

                "is_in_trdizin":
                    subject in real_subjects,

                "trdizin_subjects":
                    " || ".join(
                        sorted(
                            real_subjects
                        )
                    ),

                "matched_subjects":
                    " || ".join(
                        sorted(
                            matched
                        )
                    ),

                "missing_subjects":
                    " || ".join(
                        sorted(
                            missing
                        )
                    ),

                "additional_subjects":
                    " || ".join(
                        sorted(
                            additional
                        )
                    ),

                "status":
                    status
            })


    # ========================================================
    # TAHMİN YOK
    # ========================================================

    else:

        output_rows.append({

            "article_id":
                article_id,

            "title":
                article_titles.get(
                    article_id,
                    ""
                ),

            "best_cluster":
                best_cluster,

            "best_similarity":
                round(
                    best_similarity,
                    4
                ),

            "selected_cluster_count":
                len(
                    selected_clusters
                ),

            "prediction_rank":
                "",

            "predicted_subject":
                "",

            "prediction_score":
                "",

            "support_count":
                "",

            "is_in_trdizin":
                False,

            "trdizin_subjects":
                " || ".join(
                    sorted(
                        real_subjects
                    )
                ),

            "matched_subjects":
                "",

            "missing_subjects":
                " || ".join(
                    sorted(
                        real_subjects
                    )
                ),

            "additional_subjects":
                "",

            "status":
                status
        })


# ============================================================
# DATAFRAME
# ============================================================

result = pd.DataFrame(
    output_rows
)


# ============================================================
# CSV
# ============================================================

result.to_csv(
    "kmeans_final_topic_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# MAKALE BAZLI ÖZET
# ============================================================

article_summary = (
    result[
        [
            "article_id",
            "status"
        ]
    ]
    .drop_duplicates(
        "article_id"
    )
)


print("\n" + "=" * 100)
print("SONUÇ ÖZETİ")
print("=" * 100)


print(
    article_summary[
        "status"
    ]
    .value_counts()
    .to_string()
)


print(
    "\nToplam makale:",
    len(article_summary)
)


# ============================================================
# ETİKETSİZ
# ============================================================

unlabeled = result[
    result["status"]
    ==
    "TR_DIZIN_ETIKETI_YOK"
]


print(
    "TR Dizin etiketi olmayan makale:",
    unlabeled[
        "article_id"
    ].nunique()
)


# ============================================================
# UYUŞMAZLIK
# ============================================================

mismatch = result[
    result["status"]
    ==
    "UYUSMAZLIK"
]


print(
    "Tam uyuşmazlık olan makale:",
    mismatch[
        "article_id"
    ].nunique()
)


# ============================================================
# EK KONU ADAYI
# ============================================================

additional = result[
    result[
        "additional_subjects"
    ]
    .fillna("")
    != ""
]


print(
    "Ek konu adayı bulunan makale:",
    additional[
        "article_id"
    ].nunique()
)


# ============================================================
# CLUSTER DAĞILIMI
# ============================================================

article_cluster_counts = (
    result[
        [
            "article_id",
            "selected_cluster_count"
        ]
    ]
    .drop_duplicates(
        "article_id"
    )
)


print(
    "Ortalama seçilen cluster:",
    round(
        article_cluster_counts[
            "selected_cluster_count"
        ].mean(),
        2
    )
)


print(
    "Maksimum seçilen cluster:",
    article_cluster_counts[
        "selected_cluster_count"
    ].max()
)


# ============================================================
# ÖRNEKLER
# ============================================================

print("\n" + "=" * 100)
print("ÖRNEK TAHMİNLER")
print("=" * 100)


example_ids = (
    result[
        "article_id"
    ]
    .drop_duplicates()
    .head(10)
)


for article_id in example_ids:

    group = result[
        result["article_id"]
        ==
        article_id
    ]


    first = group.iloc[0]


    print("\n" + "-" * 100)

    print(
        "Article ID:",
        article_id
    )


    if first["title"]:

        print(
            "Başlık:",
            first["title"]
        )


    print(
        "Durum:",
        first["status"]
    )


    print(
        "Seçilen cluster sayısı:",
        first["selected_cluster_count"]
    )


    print(
        "En iyi centroid benzerliği:",
        first["best_similarity"]
    )


    print("\nTR Dizin:")


    if first["trdizin_subjects"]:

        for subject in str(
            first[
                "trdizin_subjects"
            ]
        ).split(
            " || "
        ):

            print(
                "  -",
                subject
            )

    else:

        print(
            "  Etiket bulunamadı"
        )


    print("\nBizim sistem:")


    predicted = group[
        group[
            "predicted_subject"
        ]
        .fillna("")
        != ""
    ]


    if predicted.empty:

        print(
            "  Tahmin üretilemedi"
        )


    else:

        for _, prediction in predicted.iterrows():

            marker = (
                "DOGRU"
                if prediction[
                    "is_in_trdizin"
                ]
                else "ADAY"
            )


            print(
                f"  - "
                f"{prediction['predicted_subject']} "
                f"| skor={prediction['prediction_score']} "
                f"| destek={prediction['support_count']} "
                f"| {marker}"
            )


print(
    "\nDosya oluşturuldu:"
    " kmeans_final_topic_predictions.csv"
)