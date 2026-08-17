import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# DOSYALAR
# ============================================================

TEXT_FILE = "real_trdizin_texts.csv"

ARTICLE_EMBEDDING_FILE = "embeddings/Qwen3_embeddings.npy"

KMEANS_PREDICTION_FILE = "kmeans_final_topic_predictions.csv"

SUBJECT_EMBEDDING_FILE = "embeddings/Qwen3_subject_embeddings.npy"

SUBJECT_METADATA_FILE = "Qwen3_subject_metadata.csv"


# ============================================================
# MEVCUT EN İYİ HİBRİT AYARLAR
# ============================================================

ALPHA = 0.55
FINAL_THRESHOLD = 0.26


# ============================================================
# SOFT HİYERARŞİ MARJINI TEST EDECEĞİZ
# ============================================================
#
# Örnek:
#
# Fen = 0.50
# Sosyal = 0.47
#
# margin = 0.05 ise:
# ikisi de tutulabilir.
#
# margin = 0 ise:
# sadece Fen tutulur.
# ============================================================

HIERARCHY_MARGINS = [
    0.00,
    0.03,
    0.05,
    0.08,
    0.10
]


# ============================================================
# VERİLER
# ============================================================

texts = pd.read_csv(
    TEXT_FILE,
    encoding="utf-8-sig"
)

predictions = pd.read_csv(
    KMEANS_PREDICTION_FILE,
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


print("=" * 120)
print("HİYERARŞİK K-MEANS + SEMANTIC HİBRİT DENEYİ")
print("=" * 120)

print(
    "Makale embedding:",
    row_embeddings.shape
)

print(
    "Konu embedding:",
    subject_embeddings.shape
)

print(
    "Alpha:",
    ALPHA
)

print(
    "Final threshold:",
    FINAL_THRESHOLD
)


# ============================================================
# MAKALE SEVİYESİNDE EMBEDDING
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
        str(
            row["subject_fullname"]
        ).strip()
    ] = int(
        row["subject_embedding_id"]
    )


print(
    "Konu sayısı:",
    len(subject_index)
)


# ============================================================
# YARDIMCI FONKSİYONLAR
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


def hierarchy_parts(subject):

    return [
        x.strip()
        for x in str(subject).split(">")
        if x.strip()
    ]


def root_of(subject):
    """
    Örnek:
    Fen > Tıp > Cerrahi
    ↓
    Fen
    """

    parts = hierarchy_parts(
        subject
    )

    return (
        parts[0]
        if len(parts) >= 1
        else ""
    )


def level2_of(subject):
    """
    Örnek:
    Fen > Tıp > Cerrahi
    ↓
    Fen > Tıp
    """

    parts = hierarchy_parts(
        subject
    )

    if len(parts) >= 2:

        return (
            parts[0]
            +
            " > "
            +
            parts[1]
        )

    elif len(parts) == 1:

        return parts[0]

    return ""


# ============================================================
# MAKALE ADAYLARINI HAZIRLA
# ============================================================

article_data = {}


for article_id, group in predictions.groupby(
    "article_id"
):

    if article_id not in article_vectors:
        continue


    real_subjects = parse_subjects(
        group.iloc[0].get(
            "trdizin_subjects",
            ""
        )
    )


    # Etiketi olmayan 49 makale
    # optimizasyon değerlendirmesine alınmıyor.
    if not real_subjects:
        continue


    article_vector = article_vectors[
        article_id
    ]


    candidate_rows = group[
        group[
            "predicted_subject"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        != ""
    ]


    candidates = []


    for _, row in candidate_rows.iterrows():

        subject = str(
            row[
                "predicted_subject"
            ]
        ).strip()


        if subject not in subject_index:
            continue


        # ----------------------------------------------------
        # K-MEANS SKOR
        # ----------------------------------------------------

        kmeans_score = float(
            row[
                "prediction_score"
            ]
        )


        # ----------------------------------------------------
        # SEMANTIC SKOR
        # ----------------------------------------------------

        subject_vector = subject_embeddings[
            subject_index[
                subject
            ]
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


        # ----------------------------------------------------
        # HİBRİT FINAL SKOR
        # ----------------------------------------------------

        final_score = (

            ALPHA
            *
            kmeans_score

            +

            (1 - ALPHA)
            *
            semantic_score
        )


        candidates.append(
            {
                "subject":
                    subject,

                "root":
                    root_of(
                        subject
                    ),

                "level2":
                    level2_of(
                        subject
                    ),

                "kmeans_score":
                    kmeans_score,

                "semantic_score":
                    semantic_score,

                "final_score":
                    final_score
            }
        )


    article_data[
        article_id
    ] = {
        "real_subjects":
            real_subjects,

        "candidates":
            candidates
    }


print(
    "Değerlendirilecek makale:",
    len(article_data)
)


# ============================================================
# BASELINE
# ============================================================
#
# Hiyerarşik filtre YOK.
# Sadece mevcut hibrit final_score >= 0.26
# ============================================================

def baseline_prediction(candidates):

    return {
        candidate[
            "subject"
        ]

        for candidate in candidates

        if candidate[
            "final_score"
        ]
        >=
        FINAL_THRESHOLD
    }


# ============================================================
# ROOT HİYERARŞİ FİLTRESİ
# ============================================================

def root_prediction(
    candidates,
    margin
):

    eligible = [
        candidate
        for candidate in candidates
        if candidate[
            "final_score"
        ] >= FINAL_THRESHOLD
    ]


    if not eligible:
        return set()


    # Her root için en güçlü konu skorunu alıyoruz.
    #
    # SUM kullanmıyoruz.
    # Çünkü çok fazla adayı olan branch
    # haksız avantaj kazanabilir.

    root_scores = {}


    for candidate in eligible:

        root = candidate[
            "root"
        ]

        score = candidate[
            "final_score"
        ]


        if (
            root not in root_scores
            or
            score > root_scores[root]
        ):

            root_scores[
                root
            ] = score


    best_root_score = max(
        root_scores.values()
    )


    selected_roots = {

        root

        for root, score
        in root_scores.items()

        if score
        >=
        best_root_score
        -
        margin
    }


    return {

        candidate[
            "subject"
        ]

        for candidate in eligible

        if candidate[
            "root"
        ]
        in
        selected_roots
    }


# ============================================================
# LEVEL-2 HİYERARŞİ FİLTRESİ
# ============================================================

def level2_prediction(
    candidates,
    margin
):

    eligible = [
        candidate
        for candidate in candidates
        if candidate[
            "final_score"
        ]
        >=
        FINAL_THRESHOLD
    ]


    if not eligible:
        return set()


    # ========================================================
    # 1. ROOT SEÇ
    # ========================================================

    root_scores = {}


    for candidate in eligible:

        root = candidate[
            "root"
        ]

        score = candidate[
            "final_score"
        ]


        if (
            root not in root_scores
            or
            score > root_scores[root]
        ):

            root_scores[
                root
            ] = score


    best_root_score = max(
        root_scores.values()
    )


    selected_roots = {

        root

        for root, score
        in root_scores.items()

        if score
        >=
        best_root_score
        -
        margin
    }


    root_filtered = [

        candidate

        for candidate in eligible

        if candidate[
            "root"
        ]
        in
        selected_roots
    ]


    if not root_filtered:
        return set()


    # ========================================================
    # 2. LEVEL-2 SEÇ
    # ========================================================

    level2_scores = {}


    for candidate in root_filtered:

        level2 = candidate[
            "level2"
        ]

        score = candidate[
            "final_score"
        ]


        if (
            level2 not in level2_scores
            or
            score > level2_scores[
                level2
            ]
        ):

            level2_scores[
                level2
            ] = score


    best_level2_score = max(
        level2_scores.values()
    )


    selected_level2 = {

        level2

        for level2, score
        in level2_scores.items()

        if score
        >=
        best_level2_score
        -
        margin
    }


    return {

        candidate[
            "subject"
        ]

        for candidate
        in root_filtered

        if candidate[
            "level2"
        ]
        in
        selected_level2
    }


# ============================================================
# DEĞERLENDİRME FONKSİYONU
# ============================================================

def evaluate(
    method_name,
    margin=None
):

    tp = 0
    fp = 0
    fn = 0

    hit = 0
    exact = 0

    predicted_counts = []


    for article_id, data in article_data.items():

        real_subjects = data[
            "real_subjects"
        ]

        candidates = data[
            "candidates"
        ]


        # ----------------------------------------------------
        # STRATEJİ
        # ----------------------------------------------------

        if method_name == "BASELINE":

            predicted_subjects = (
                baseline_prediction(
                    candidates
                )
            )


        elif method_name == "ROOT":

            predicted_subjects = (
                root_prediction(
                    candidates,
                    margin
                )
            )


        elif method_name == "LEVEL2":

            predicted_subjects = (
                level2_prediction(
                    candidates,
                    margin
                )
            )


        else:

            raise ValueError(
                "Bilinmeyen method"
            )


        predicted_counts.append(
            len(
                predicted_subjects
            )
        )


        # ----------------------------------------------------
        # TP FP FN
        # ----------------------------------------------------

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


        if (
            predicted_subjects
            &
            real_subjects
        ):

            hit += 1


        if (
            predicted_subjects
            ==
            real_subjects
        ):

            exact += 1


    evaluated = len(
        article_data
    )


    precision = (

        tp
        /
        (tp + fp)

        if tp + fp > 0

        else 0
    )


    recall = (

        tp
        /
        (tp + fn)

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

        hit
        /
        evaluated

        if evaluated

        else 0
    )


    exact_match = (

        exact
        /
        evaluated

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
            )
            ==
            0
        )

        if predicted_counts

        else 0
    )


    return {

        "Method":
            method_name,

        "Hierarchy_Margin":
            margin,

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


# ============================================================
# TÜM DENEYLER
# ============================================================

results = []


# ------------------------------------------------------------
# BASELINE
# ------------------------------------------------------------

results.append(
    evaluate(
        "BASELINE"
    )
)


# ------------------------------------------------------------
# ROOT
# ------------------------------------------------------------

for margin in HIERARCHY_MARGINS:

    results.append(
        evaluate(
            "ROOT",
            margin
        )
    )


# ------------------------------------------------------------
# LEVEL 2
# ------------------------------------------------------------

for margin in HIERARCHY_MARGINS:

    results.append(
        evaluate(
            "LEVEL2",
            margin
        )
    )


# ============================================================
# SONUÇLAR
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
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    250
)


print("\n" + "=" * 150)
print("HİYERARŞİK YÖNTEM KARŞILAŞTIRMASI")
print("=" * 150)


print(
    result_df.to_string(
        index=False
    )
)


# ============================================================
# GENEL EN İYİ
# ============================================================

best = result_df.iloc[0]


print("\n" + "=" * 120)
print("GENEL EN İYİ HİYERARŞİK SONUÇ")
print("=" * 120)


print(
    best.to_string()
)


# ============================================================
# BASELINE
# ============================================================

baseline = result_df[
    result_df[
        "Method"
    ]
    ==
    "BASELINE"
].iloc[0]


print("\n" + "=" * 120)
print("BASELINE HİBRİT VS EN İYİ HİYERARŞİK")
print("=" * 120)


print(
    f"Precision: "
    f"{baseline['Precision']:.4f}"
    f" -> {best['Precision']:.4f}"
)

print(
    f"Recall: "
    f"{baseline['Recall']:.4f}"
    f" -> {best['Recall']:.4f}"
)

print(
    f"F1: "
    f"{baseline['F1']:.4f}"
    f" -> {best['F1']:.4f}"
)

print(
    f"Hit Rate: "
    f"{baseline['Hit_Rate']:.4f}"
    f" -> {best['Hit_Rate']:.4f}"
)

print(
    f"Exact Match: "
    f"{baseline['Exact_Match']:.4f}"
    f" -> {best['Exact_Match']:.4f}"
)


# ============================================================
# EN İYİ ROOT
# ============================================================

root_results = result_df[
    result_df[
        "Method"
    ]
    ==
    "ROOT"
]


best_root = root_results.iloc[0]


print("\n" + "=" * 120)
print("EN İYİ ROOT FİLTRESİ")
print("=" * 120)


print(
    best_root.to_string()
)


# ============================================================
# EN İYİ LEVEL2
# ============================================================

level2_results = result_df[
    result_df[
        "Method"
    ]
    ==
    "LEVEL2"
]


best_level2 = level2_results.iloc[0]


print("\n" + "=" * 120)
print("EN İYİ LEVEL-2 FİLTRESİ")
print("=" * 120)


print(
    best_level2.to_string()
)


# ============================================================
# CSV
# ============================================================

result_df.to_csv(
    "hierarchical_hybrid_results.csv",
    index=False,
    encoding="utf-8-sig"
)


print(
    "\nDosya oluşturuldu:"
    " hierarchical_hybrid_results.csv"
)