import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# DOSYALAR
# ============================================================

PREDICTION_FILE = "final_hybrid_predictions.csv"

SUBJECT_METADATA_FILE = "Qwen3_subject_metadata.csv"

SUBJECT_EMBEDDING_FILE = (
    "embeddings/Qwen3_subject_embeddings.npy"
)

OUTPUT_FILE = (
    "semantic_aware_evaluation_results.csv"
)


# ============================================================
# BENZERLİK SINIRLARI
# ============================================================
#
# Bunlar exact label yerine
# anlamsal yakınlığı değerlendirmek için.
# ============================================================

SEMANTIC_THRESHOLDS = [
    0.60,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90
]


# ============================================================
# VERİLER
# ============================================================

predictions = pd.read_csv(
    PREDICTION_FILE,
    encoding="utf-8-sig"
)

subject_metadata = pd.read_csv(
    SUBJECT_METADATA_FILE,
    encoding="utf-8-sig"
)

subject_embeddings = np.load(
    SUBJECT_EMBEDDING_FILE
).astype(np.float32)


print("=" * 115)
print("SEMANTIC-AWARE KONU TAHMİN DEĞERLENDİRMESİ")
print("=" * 115)

print(
    "Prediction satırı:",
    len(predictions)
)

print(
    "Makale sayısı:",
    predictions[
        "article_id"
    ].nunique()
)

print(
    "Konu embedding:",
    subject_embeddings.shape
)


# ============================================================
# SUBJECT -> EMBEDDING INDEX
# ============================================================

subject_index = {}


for _, row in subject_metadata.iterrows():

    subject = str(
        row[
            "subject_fullname"
        ]
    ).strip()

    subject_index[
        subject
    ] = int(
        row[
            "subject_embedding_id"
        ]
    )


# ============================================================
# YARDIMCI FONKSİYON
# ============================================================

def parse_subjects(value):

    if pd.isna(value):
        return set()

    value = str(
        value
    ).strip()

    if not value:
        return set()

    return {
        x.strip()
        for x in value.split("||")
        if x.strip()
    }


def subject_similarity(
    subject_a,
    subject_b
):

    if (
        subject_a not in subject_index
        or
        subject_b not in subject_index
    ):

        return np.nan


    vec_a = subject_embeddings[
        subject_index[
            subject_a
        ]
    ]

    vec_b = subject_embeddings[
        subject_index[
            subject_b
        ]
    ]


    return float(
        cosine_similarity(

            vec_a.reshape(
                1,
                -1
            ),

            vec_b.reshape(
                1,
                -1
            )

        )[0][0]
    )


# ============================================================
# ETİKETLİ MAKALELER
# ============================================================

article_groups = []


for article_id, group in predictions.groupby(
    "article_id"
):

    first = group.iloc[0]


    real_subjects = parse_subjects(
        first.get(
            "trdizin_subjects",
            ""
        )
    )


    # TR Dizin etiketi olmayan 49 makale
    # değerlendirmeye girmiyor.
    if not real_subjects:
        continue


    predicted_subjects = set(

        group[
            "predicted_subject"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )


    predicted_subjects.discard(
        ""
    )


    article_groups.append(
        (
            article_id,
            real_subjects,
            predicted_subjects
        )
    )


print(
    "Değerlendirilen makale:",
    len(article_groups)
)


# ============================================================
# EXACT BASELINE
# ============================================================

exact_tp = 0
exact_fp = 0
exact_fn = 0

exact_hit = 0


for (
    article_id,
    real_subjects,
    predicted_subjects
) in article_groups:

    exact_tp += len(
        predicted_subjects
        &
        real_subjects
    )

    exact_fp += len(
        predicted_subjects
        -
        real_subjects
    )

    exact_fn += len(
        real_subjects
        -
        predicted_subjects
    )


    if (
        predicted_subjects
        &
        real_subjects
    ):

        exact_hit += 1


exact_precision = (

    exact_tp
    /
    (
        exact_tp
        +
        exact_fp
    )

    if (
        exact_tp
        +
        exact_fp
    ) > 0

    else 0
)


exact_recall = (

    exact_tp
    /
    (
        exact_tp
        +
        exact_fn
    )

    if (
        exact_tp
        +
        exact_fn
    ) > 0

    else 0
)


exact_f1 = (

    2
    *
    exact_precision
    *
    exact_recall

    /
    (
        exact_precision
        +
        exact_recall
    )

    if (
        exact_precision
        +
        exact_recall
    ) > 0

    else 0
)


exact_hit_rate = (

    exact_hit
    /
    len(
        article_groups
    )
)


print("\n" + "=" * 115)
print("EXACT LABEL BASELINE")
print("=" * 115)

print(
    "Precision:",
    round(
        exact_precision,
        4
    )
)

print(
    "Recall:",
    round(
        exact_recall,
        4
    )
)

print(
    "F1:",
    round(
        exact_f1,
        4
    )
)

print(
    "Hit Rate:",
    round(
        exact_hit_rate,
        4
    )
)


# ============================================================
# SEMANTIC-AWARE DEĞERLENDİRME
# ============================================================
#
# Bir tahmin, herhangi bir gerçek etikete
# threshold kadar yakınsa
# semantic doğru sayılıyor.
#
# Aynı gerçek etiket birden fazla tahmin tarafından
# tekrar tekrar kullanılmasın diye greedy matching
# (açgözlü eşleştirme) yapıyoruz.
# ============================================================

results = []


for threshold in SEMANTIC_THRESHOLDS:

    total_tp = 0
    total_fp = 0
    total_fn = 0

    hit_count = 0

    exact_article_match = 0

    similarities_of_matches = []


    for (
        article_id,
        real_subjects,
        predicted_subjects
    ) in article_groups:

        real_list = list(
            real_subjects
        )

        predicted_list = list(
            predicted_subjects
        )


        # ----------------------------------------------------
        # TÜM PREDICTED-REAL PAIR'LARI
        # ----------------------------------------------------

        candidate_pairs = []


        for predicted in predicted_list:

            for real in real_list:

                # Exact match zaten similarity=1 olarak düşünülebilir

                if predicted == real:

                    similarity = 1.0

                else:

                    similarity = subject_similarity(
                        predicted,
                        real
                    )


                if pd.isna(
                    similarity
                ):

                    continue


                candidate_pairs.append(
                    (
                        similarity,
                        predicted,
                        real
                    )
                )


        # ----------------------------------------------------
        # EN YÜKSEK SIMILARITY'DEN BAŞLA
        # ----------------------------------------------------

        candidate_pairs.sort(
            key=lambda x: x[0],
            reverse=True
        )


        used_predicted = set()
        used_real = set()

        matched_pairs = []


        for (
            similarity,
            predicted,
            real
        ) in candidate_pairs:

            if similarity < threshold:
                continue


            if predicted in used_predicted:
                continue


            if real in used_real:
                continue


            used_predicted.add(
                predicted
            )

            used_real.add(
                real
            )


            matched_pairs.append(
                (
                    predicted,
                    real,
                    similarity
                )
            )


            similarities_of_matches.append(
                similarity
            )


        # ----------------------------------------------------
        # SEMANTIC TP / FP / FN
        # ----------------------------------------------------

        article_tp = len(
            matched_pairs
        )

        article_fp = (
            len(
                predicted_list
            )
            -
            article_tp
        )

        article_fn = (
            len(
                real_list
            )
            -
            article_tp
        )


        total_tp += article_tp
        total_fp += article_fp
        total_fn += article_fn


        if article_tp > 0:

            hit_count += 1


        # Bütün gerçek + tahminler eşleşmiş mi?
        if (
            article_fp == 0
            and
            article_fn == 0
        ):

            exact_article_match += 1


    # ========================================================
    # METRİKLER
    # ========================================================

    precision = (

        total_tp
        /
        (
            total_tp
            +
            total_fp
        )

        if (
            total_tp
            +
            total_fp
        ) > 0

        else 0
    )


    recall = (

        total_tp
        /
        (
            total_tp
            +
            total_fn
        )

        if (
            total_tp
            +
            total_fn
        ) > 0

        else 0
    )


    f1 = (

        2
        *
        precision
        *
        recall

        /
        (
            precision
            +
            recall
        )

        if (
            precision
            +
            recall
        ) > 0

        else 0
    )


    hit_rate = (

        hit_count
        /
        len(
            article_groups
        )
    )


    semantic_exact_match = (

        exact_article_match
        /
        len(
            article_groups
        )
    )


    mean_matched_similarity = (

        np.mean(
            similarities_of_matches
        )

        if similarities_of_matches

        else 0
    )


    results.append(
        {
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

            "Semantic_Exact_Match":
                round(
                    semantic_exact_match,
                    4
                ),

            "Mean_Matched_Similarity":
                round(
                    mean_matched_similarity,
                    4
                ),

            "TP":
                total_tp,

            "FP":
                total_fp,

            "FN":
                total_fn,

            "Evaluated":
                len(
                    article_groups
                )
        }
    )


# ============================================================
# SONUÇ
# ============================================================

result_df = pd.DataFrame(
    results
)


print("\n" + "=" * 135)
print("SEMANTIC-AWARE SONUÇLAR")
print("=" * 135)


print(
    result_df.to_string(
        index=False
    )
)


# ============================================================
# EXACT VS SEMANTIC
# ============================================================

print("\n" + "=" * 120)
print("EXACT LABEL VS SEMANTIC-AWARE")
print("=" * 120)


for _, row in result_df.iterrows():

    print(
        f"Threshold >= "
        f"{row['Semantic_Threshold']:.2f}"
        f" | "
        f"F1: "
        f"{exact_f1:.4f}"
        f" -> "
        f"{row['F1']:.4f}"
        f" | "
        f"Hit Rate: "
        f"{exact_hit_rate:.4f}"
        f" -> "
        f"{row['Hit_Rate']:.4f}"
    )


# ============================================================
# ÖNERİLEN RAPORLAMA NOKTALARI
# ============================================================

print("\n" + "=" * 120)
print("ÖNERİLEN RAPORLAMA NOKTALARI")
print("=" * 120)


for threshold in [
    0.75,
    0.80,
    0.90
]:

    selected = result_df[
        result_df[
            "Semantic_Threshold"
        ]
        ==
        threshold
    ]


    if selected.empty:
        continue


    row = selected.iloc[0]


    print(
        f"\nSemantic threshold = "
        f"{threshold}"
    )

    print(
        "Precision:",
        row[
            "Precision"
        ]
    )

    print(
        "Recall:",
        row[
            "Recall"
        ]
    )

    print(
        "F1:",
        row[
            "F1"
        ]
    )

    print(
        "Hit Rate:",
        row[
            "Hit_Rate"
        ]
    )

    print(
        "Semantic Exact Match:",
        row[
            "Semantic_Exact_Match"
        ]
    )


# ============================================================
# CSV
# ============================================================

result_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


print(
    "\nDosya oluşturuldu:",
    OUTPUT_FILE
)