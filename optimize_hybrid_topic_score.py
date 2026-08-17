import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# DOSYALAR
# ============================================================

TEXT_FILE = "real_trdizin_texts.csv"
ARTICLE_EMBEDDING_FILE = "embeddings/Qwen3_embeddings.npy"

PREDICTION_FILE = "kmeans_final_topic_predictions.csv"

SUBJECT_EMBEDDING_FILE = "embeddings/Qwen3_subject_embeddings.npy"
SUBJECT_METADATA_FILE = "Qwen3_subject_metadata.csv"


# ============================================================
# DENEYECEĞİMİZ AYARLAR
# ============================================================

# Final score:
# alpha * KMeansScore + (1-alpha) * SemanticScore
ALPHAS = [
    0.20,
    0.30,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60
]

FINAL_THRESHOLDS = [
    0.18,
    0.20,
    0.21,
    0.22,
    0.225,
    0.23,
    0.24,
    0.25,
    0.26
]



# ============================================================
# VERİLERİ OKU
# ============================================================

texts = pd.read_csv(
    TEXT_FILE,
    encoding="utf-8-sig"
)

predictions = pd.read_csv(
    PREDICTION_FILE,
    encoding="utf-8-sig"
)

subject_metadata = pd.read_csv(
    SUBJECT_METADATA_FILE,
    encoding="utf-8-sig"
)

row_embeddings = np.load(
    ARTICLE_EMBEDDING_FILE
).astype(np.float32)

subject_embeddings = np.load(
    SUBJECT_EMBEDDING_FILE
).astype(np.float32)


print("=" * 115)
print("K-MEANS + SEMANTIC HİBRİT KONU SKORU OPTİMİZASYONU")
print("=" * 115)

print(
    "Makale embedding:",
    row_embeddings.shape
)

print(
    "Konu embedding:",
    subject_embeddings.shape
)


# ============================================================
# MAKALE SEVİYESİNDE EMBEDDING
# ============================================================

article_vectors = {}

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

    article_vectors[
        article_id
    ] = vector


print(
    "Makale sayısı:",
    len(article_vectors)
)


# ============================================================
# SUBJECT -> EMBEDDING INDEX
# ============================================================

subject_index = {}

for _, row in subject_metadata.iterrows():

    subject_index[
        row["subject_fullname"]
    ] = int(
        row["subject_embedding_id"]
    )


print(
    "Konu sayısı:",
    len(subject_index)
)


# ============================================================
# TR DİZİN ETİKETLERİNİ PARÇALA
# ============================================================

def parse_subjects(value):

    if pd.isna(value):
        return set()

    value = str(value).strip()

    if not value:
        return set()

    return {
        x.strip()
        for x in value.split("||")
        if x.strip()
    }


# ============================================================
# MAKALE BAZLI ADAYLARI HAZIRLA
# ============================================================

article_groups = []


for article_id, group in predictions.groupby("article_id"):

    if article_id not in article_vectors:
        continue


    real_subjects = parse_subjects(
        group.iloc[0][
            "trdizin_subjects"
        ]
    )


    # Ground truth yoksa optimizasyona katma
    if not real_subjects:
        continue


    candidates = group[
        group[
            "predicted_subject"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        != ""
    ].copy()


    article_groups.append(
        (
            article_id,
            candidates,
            real_subjects
        )
    )


print(
    "Değerlendirilecek makale:",
    len(article_groups)
)


# ============================================================
# HER ADAY İÇİN
# K-MEANS SCORE + SEMANTIC SCORE HESAPLA
# ============================================================

candidate_cache = {}


for article_id, group, real_subjects in article_groups:

    article_vector = article_vectors[
        article_id
    ]

    rows = []


    for _, row in group.iterrows():

        subject = row[
            "predicted_subject"
        ]


        if subject not in subject_index:
            continue


        # --------------------------------------------
        # K-Means konu skoru
        # --------------------------------------------

        kmeans_score = float(
            row[
                "prediction_score"
            ]
        )


        # --------------------------------------------
        # Semantic similarity
        # --------------------------------------------

        subject_idx = subject_index[
            subject
        ]

        subject_vector = subject_embeddings[
            subject_idx
        ]


        semantic_score = float(
            cosine_similarity(
                article_vector.reshape(
                    1,
                    -1
                ),
                subject_vector.reshape(
                    1,
                    -1
                )
            )[0][0]
        )


        rows.append(
            {
                "subject":
                    subject,

                "kmeans_score":
                    kmeans_score,

                "semantic_score":
                    semantic_score
            }
        )


    candidate_cache[
        article_id
    ] = rows


# ============================================================
# ALPHA + THRESHOLD TARAMASI
# ============================================================

results = []


for alpha in ALPHAS:

    for threshold in FINAL_THRESHOLDS:

        tp = 0
        fp = 0
        fn = 0

        hit = 0
        exact = 0

        predicted_counts = []


        for (
            article_id,
            group,
            real_subjects
        ) in article_groups:

            candidates = candidate_cache.get(
                article_id,
                []
            )


            predicted_subjects = set()


            for candidate in candidates:

                kmeans_score = candidate[
                    "kmeans_score"
                ]

                semantic_score = candidate[
                    "semantic_score"
                ]


                # ============================================
                # HİBRİT SKOR
                # ============================================

                final_score = (

                    alpha
                    *
                    kmeans_score

                    +

                    (1 - alpha)
                    *
                    semantic_score
                )


                if final_score >= threshold:

                    predicted_subjects.add(
                        candidate[
                            "subject"
                        ]
                    )


            predicted_counts.append(
                len(predicted_subjects)
            )


            # ================================================
            # METRİKLER
            # ================================================

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
                hit += 1


            if predicted_subjects == real_subjects:
                exact += 1


        evaluated = len(
            article_groups
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
                predicted_counts
            )
            if predicted_counts
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


        results.append(
            {
                "Alpha":
                    alpha,

                "Final_Threshold":
                    threshold,

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
        )


# ============================================================
# SONUÇ TABLOSU
# ============================================================

result_df = pd.DataFrame(
    results
)


result_df = result_df.sort_values(
    "F1",
    ascending=False
).reset_index(
    drop=True
)


pd.set_option(
    "display.width",
    250
)

pd.set_option(
    "display.max_columns",
    None
)


# ============================================================
# HER ALPHA İÇİN EN İYİ
# ============================================================

best_per_alpha = (
    result_df
    .sort_values(
        "F1",
        ascending=False
    )
    .groupby(
        "Alpha",
        as_index=False
    )
    .first()
    .sort_values(
        "F1",
        ascending=False
    )
)


print("\n" + "=" * 135)
print("HER ALPHA İÇİN EN İYİ SONUÇ")
print("=" * 135)

print(
    best_per_alpha.to_string(
        index=False
    )
)


# ============================================================
# GENEL EN İYİ
# ============================================================

best = result_df.iloc[0]


print("\n" + "=" * 120)
print("GENEL EN İYİ HİBRİT SONUÇ")
print("=" * 120)

print(
    best.to_string()
)


# ============================================================
# EN İYİ 15
# ============================================================

print("\n" + "=" * 135)
print("EN İYİ 15 KOMBİNASYON")
print("=" * 135)

print(
    result_df
    .head(15)
    .to_string(
        index=False
    )
)


# ============================================================
# BASELINE KARŞILAŞTIRMA
# ============================================================

BASELINE_PRECISION = 0.4078
BASELINE_RECALL = 0.3490
BASELINE_F1 = 0.3762
BASELINE_HIT_RATE = 0.6141


print("\n" + "=" * 120)
print("MEVCUT K-MEANS VS HİBRİT SİSTEM")
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

result_df.to_csv(
    "hybrid_topic_score_results.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# EN İYİ AYARLA ÖRNEKLER
# ============================================================

BEST_ALPHA = float(
    best[
        "Alpha"
    ]
)

BEST_THRESHOLD = float(
    best[
        "Final_Threshold"
    ]
)


print("\n" + "=" * 120)
print("EN İYİ AYARLA ÖRNEK HİBRİT SKORLAR")
print("=" * 120)


shown = 0


for (
    article_id,
    group,
    real_subjects
) in article_groups:

    candidates = candidate_cache.get(
        article_id,
        []
    )


    if not candidates:
        continue


    print(
        "\nArticle:",
        article_id
    )


    if not group.empty:

        print(
            "Başlık:",
            group.iloc[0].get(
                "title",
                ""
            )
        )


    scored_candidates = []


    for candidate in candidates:

        final_score = (

            BEST_ALPHA
            *
            candidate[
                "kmeans_score"
            ]

            +

            (1 - BEST_ALPHA)
            *
            candidate[
                "semantic_score"
            ]
        )


        scored_candidates.append(
            (
                candidate[
                    "subject"
                ],

                candidate[
                    "kmeans_score"
                ],

                candidate[
                    "semantic_score"
                ],

                final_score
            )
        )


    for (
        subject,
        kmeans_score,
        semantic_score,
        final_score
    ) in sorted(
        scored_candidates,
        key=lambda x: x[3],
        reverse=True
    ):

        selected = (
            final_score
            >=
            BEST_THRESHOLD
        )


        truth = (
            "TR_DIZINDE_VAR"
            if subject in real_subjects
            else "EK_ADAY"
        )


        print(
            f"  {subject}"
            f" | KMeans={kmeans_score:.4f}"
            f" | Semantic={semantic_score:.4f}"
            f" | Final={final_score:.4f}"
            f" | {'SECILDI' if selected else 'ELENDI'}"
            f" | {truth}"
        )


    shown += 1

    if shown >= 10:
        break


print(
    "\nDosya oluşturuldu:"
    " hybrid_topic_score_results.csv"
)