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
# DENEYECEĞİMİZ SEMANTIC THRESHOLD'LAR
# ============================================================

SEMANTIC_THRESHOLDS = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70
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


print("=" * 110)
print("K-MEANS + SEMANTIC SUBJECT VALIDATION")
print("=" * 110)

print(
    "Makale embedding:",
    row_embeddings.shape
)

print(
    "Konu embedding:",
    subject_embeddings.shape
)


# ============================================================
# ARTICLE LEVEL EMBEDDING
# ============================================================
#
# TUR + ENG varsa ortalama alıyoruz.
# ============================================================

article_vectors = {}


for article_id, group in texts.groupby(
    "article_id"
):

    indices = group.index.to_numpy()

    vector = row_embeddings[
        indices
    ].mean(
        axis=0
    )

    norm = np.linalg.norm(
        vector
    )

    if norm > 0:

        vector = (
            vector / norm
        )


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
        subject.strip()

        for subject
        in value.split("||")

        if subject.strip()
    }


# ============================================================
# HER MAKALE İÇİN K-MEANS ADAYLARI
# ============================================================

article_prediction_groups = []


for article_id, group in predictions.groupby(
    "article_id"
):

    if article_id not in article_vectors:
        continue


    real_subjects = parse_subjects(
        group.iloc[0][
            "trdizin_subjects"
        ]
    )


    # Ground truth yoksa
    # threshold optimizasyonuna katma.
    if not real_subjects:
        continue


    candidate_rows = group[
        group[
            "predicted_subject"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        != ""
    ].copy()


    article_prediction_groups.append(
        (
            article_id,
            candidate_rows,
            real_subjects
        )
    )


print(
    "Değerlendirilecek makale:",
    len(article_prediction_groups)
)


# ============================================================
# SEMANTIC SIMILARITY HESAPLA
# ============================================================

semantic_cache = {}


for article_id, group, real_subjects in article_prediction_groups:

    article_vector = article_vectors[
        article_id
    ]


    scores = {}


    for _, row in group.iterrows():

        subject = row[
            "predicted_subject"
        ]


        if subject not in subject_index:
            continue


        idx = subject_index[
            subject
        ]


        subject_vector = subject_embeddings[
            idx
        ]


        similarity = float(
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


        scores[
            subject
        ] = similarity


    semantic_cache[
        article_id
    ] = scores


# ============================================================
# THRESHOLD TESTLERİ
# ============================================================

results = []


for threshold in SEMANTIC_THRESHOLDS:

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
    ) in article_prediction_groups:

        semantic_scores = semantic_cache.get(
            article_id,
            {}
        )


        # ====================================================
        # K-MEANS ADAYLARI İÇİNDEN
        # SEMANTIC EŞİĞİ GEÇENLER
        # ====================================================

        predicted_subjects = {

            subject

            for subject, similarity
            in semantic_scores.items()

            if similarity >= threshold
        }


        predicted_counts.append(
            len(predicted_subjects)
        )


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
        article_prediction_groups
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


    results.append({

        "Semantic_Threshold":
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
    })


# ============================================================
# SONUÇ TABLOSU
# ============================================================

result_df = pd.DataFrame(
    results
)


print("\n" + "=" * 120)
print("SEMANTIC THRESHOLD SONUÇLARI")
print("=" * 120)


print(
    result_df.to_string(
        index=False
    )
)


# ============================================================
# EN İYİ F1
# ============================================================

best = result_df.loc[
    result_df[
        "F1"
    ].idxmax()
]


print("\n" + "=" * 120)
print("F1'E GÖRE EN İYİ SEMANTIC THRESHOLD")
print("=" * 120)


print(
    best.to_string()
)


# ============================================================
# BASELINE KARŞILAŞTIRMASI
# ============================================================

BASELINE_PRECISION = 0.4078
BASELINE_RECALL = 0.3490
BASELINE_F1 = 0.3762
BASELINE_HIT_RATE = 0.6141


print("\n" + "=" * 120)
print("MEVCUT K-MEANS VS SEMANTIC DOĞRULAMA")
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
# KAYDET
# ============================================================

result_df.to_csv(
    "semantic_subject_validation_results.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# ÖRNEK SEMANTIC SKORLAR
# ============================================================

print("\n" + "=" * 120)
print("ÖRNEK K-MEANS ADAYLARI + SEMANTIC SIMILARITY")
print("=" * 120)


shown = 0


for (
    article_id,
    group,
    real_subjects
) in article_prediction_groups:

    semantic_scores = semantic_cache.get(
        article_id,
        {}
    )


    if not semantic_scores:
        continue


    title = group.iloc[0].get(
        "title",
        ""
    )


    print(
        "\nArticle:",
        article_id
    )

    print(
        "Başlık:",
        title
    )


    for subject, similarity in sorted(
        semantic_scores.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        status = (
            "TR_DIZINDE_VAR"
            if subject in real_subjects
            else "EK_ADAY"
        )


        print(
            f"  {subject}"
            f" | semantic={similarity:.4f}"
            f" | {status}"
        )


    shown += 1

    if shown >= 10:
        break


print(
    "\nDosya oluşturuldu:"
    " semantic_subject_validation_results.csv"
)