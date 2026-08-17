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
# FINAL AYARLAR
# ============================================================

ALPHA = 0.55
FINAL_THRESHOLD = 0.26


# ============================================================
# VERİLERİ OKU
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


print("=" * 115)
print("FINAL K-MEANS + SEMANTIC HİBRİT KONU TAHMİN SİSTEMİ")
print("=" * 115)

print(
    "Makale embedding:",
    row_embeddings.shape
)

print(
    "Konu embedding:",
    subject_embeddings.shape
)

print(
    f"Alpha: {ALPHA}"
)

print(
    f"Final threshold: {FINAL_THRESHOLD}"
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
# SONUÇ SATIRLARI
# ============================================================

output_rows = []


# ============================================================
# HER MAKALEYİ İŞLE
# ============================================================

for article_id, group in predictions.groupby(
    "article_id"
):

    if article_id not in article_vectors:
        continue


    first_row = group.iloc[0]


    title = (
        str(
            first_row.get(
                "title",
                ""
            )
        )
    )


    trdizin_subjects = parse_subjects(
        first_row.get(
            "trdizin_subjects",
            ""
        )
    )


    article_vector = article_vectors[
        article_id
    ]


    # ========================================================
    # K-MEANS ADAYLARI
    # ========================================================

    candidate_rows = group[
        group[
            "predicted_subject"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        != ""
    ].copy()


    final_predictions = []


    for _, row in candidate_rows.iterrows():

        subject = str(
            row[
                "predicted_subject"
            ]
        ).strip()


        if subject not in subject_index:
            continue


        # ----------------------------------------------------
        # K-MEANS SKORU
        # ----------------------------------------------------

        kmeans_score = float(
            row[
                "prediction_score"
            ]
        )


        # ----------------------------------------------------
        # SUBJECT EMBEDDING
        # ----------------------------------------------------

        subject_idx = subject_index[
            subject
        ]

        subject_vector = subject_embeddings[
            subject_idx
        ]


        # ----------------------------------------------------
        # SEMANTIC SIMILARITY
        # ----------------------------------------------------

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
        # HİBRİT FINAL SCORE
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


        selected = (
            final_score
            >=
            FINAL_THRESHOLD
        )


        final_predictions.append(
            {
                "subject":
                    subject,

                "kmeans_score":
                    kmeans_score,

                "semantic_score":
                    semantic_score,

                "final_score":
                    final_score,

                "support_count":
                    row.get(
                        "support_count",
                        0
                    ),

                "selected":
                    selected,

                "is_in_trdizin":
                    subject
                    in
                    trdizin_subjects
            }
        )


    # ========================================================
    # FINAL SEÇİLEN KONULAR
    # ========================================================

    selected_predictions = [

        x

        for x in final_predictions

        if x[
            "selected"
        ]
    ]


    selected_predictions = sorted(

        selected_predictions,

        key=lambda x:
        x[
            "final_score"
        ],

        reverse=True
    )


    predicted_subjects = {

        x[
            "subject"
        ]

        for x in selected_predictions
    }


    # ========================================================
    # DURUM
    # ========================================================

    if not trdizin_subjects:

        status = (
            "TR_DIZIN_ETIKETI_YOK"
        )


    elif (
        predicted_subjects
        ==
        trdizin_subjects
    ):

        status = (
            "TAM_ESLESME"
        )


    elif (
        predicted_subjects
        &
        trdizin_subjects
    ):

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
    # SEÇİLENLER VARSA
    # ========================================================

    if selected_predictions:

        for rank, prediction in enumerate(
            selected_predictions,
            start=1
        ):

            output_rows.append(
                {
                    "article_id":
                        article_id,

                    "title":
                        title,

                    "status":
                        status,

                    "prediction_rank":
                        rank,

                    "predicted_subject":
                        prediction[
                            "subject"
                        ],

                    "kmeans_score":
                        round(
                            prediction[
                                "kmeans_score"
                            ],
                            4
                        ),

                    "semantic_score":
                        round(
                            prediction[
                                "semantic_score"
                            ],
                            4
                        ),

                    "final_score":
                        round(
                            prediction[
                                "final_score"
                            ],
                            4
                        ),

                    "support_count":
                        prediction[
                            "support_count"
                        ],

                    "is_in_trdizin":
                        prediction[
                            "is_in_trdizin"
                        ],

                    "trdizin_subjects":
                        " || ".join(
                            sorted(
                                trdizin_subjects
                            )
                        ),

                    "best_cluster":
                        first_row.get(
                            "best_cluster",
                            ""
                        ),

                    "best_similarity":
                        first_row.get(
                            "best_similarity",
                            ""
                        ),

                    "selected_cluster_count":
                        first_row.get(
                            "selected_cluster_count",
                            ""
                        )
                }
            )


    # ========================================================
    # HİÇ FINAL TAHMİN YOKSA
    # ========================================================

    else:

        output_rows.append(
            {
                "article_id":
                    article_id,

                "title":
                    title,

                "status":
                    status,

                "prediction_rank":
                    "",

                "predicted_subject":
                    "",

                "kmeans_score":
                    "",

                "semantic_score":
                    "",

                "final_score":
                    "",

                "support_count":
                    "",

                "is_in_trdizin":
                    False,

                "trdizin_subjects":
                    " || ".join(
                        sorted(
                            trdizin_subjects
                        )
                    ),

                "best_cluster":
                    first_row.get(
                        "best_cluster",
                        ""
                    ),

                "best_similarity":
                    first_row.get(
                        "best_similarity",
                        ""
                    ),

                "selected_cluster_count":
                    first_row.get(
                        "selected_cluster_count",
                        ""
                    )
            }
        )


# ============================================================
# DATAFRAME
# ============================================================

result = pd.DataFrame(
    output_rows
)


# ============================================================
# TÜM SONUÇLARI KAYDET
# ============================================================

result.to_csv(
    "final_hybrid_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SADECE TR DİZİN ETİKETİ OLMAYANLAR
# ============================================================

unlabeled = result[
    result[
        "status"
    ]
    ==
    "TR_DIZIN_ETIKETI_YOK"
].copy()


unlabeled.to_csv(
    "final_hybrid_unlabeled_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SADECE UYUŞMAZLIKLAR
# ============================================================

mismatches = result[
    result[
        "status"
    ]
    ==
    "UYUSMAZLIK"
].copy()


mismatches.to_csv(
    "final_hybrid_mismatches.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# ÖZET
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


print("\n" + "=" * 110)
print("FINAL SİSTEM ÖZETİ")
print("=" * 110)


print(
    article_summary[
        "status"
    ]
    .value_counts()
    .to_string()
)


print(
    "\nToplam makale:",
    article_summary[
        "article_id"
    ].nunique()
)


print(
    "TR Dizin etiketi olmayan:",
    unlabeled[
        "article_id"
    ].nunique()
)


print(
    "Uyuşmazlık:",
    mismatches[
        "article_id"
    ].nunique()
)


# ============================================================
# ETİKETSİZ MAKALE ÖRNEKLERİ
# ============================================================

print("\n" + "=" * 120)
print("TR DİZİN ETİKETİ OLMAYAN MAKALELER İÇİN HİBRİT TAHMİNLER")
print("=" * 120)


unlabeled_ids = (
    unlabeled[
        "article_id"
    ]
    .drop_duplicates()
    .tolist()
)


for article_id in unlabeled_ids:

    group = unlabeled[
        unlabeled[
            "article_id"
        ]
        ==
        article_id
    ]


    first = group.iloc[0]


    print("\n" + "-" * 110)

    print(
        "Article ID:",
        article_id
    )

    print(
        "Başlık:",
        first[
            "title"
        ]
    )


    predicted = group[
        group[
            "predicted_subject"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        != ""
    ]


    if predicted.empty:

        print(
            "Tahmin üretilemedi."
        )

        continue


    for _, row in predicted.iterrows():

        print(
            f"  - {row['predicted_subject']}"
            f" | KMeans={row['kmeans_score']}"
            f" | Semantic={row['semantic_score']}"
            f" | Final={row['final_score']}"
            f" | Support={row['support_count']}"
        )


print("\nDosyalar oluşturuldu:")

print(
    "final_hybrid_predictions.csv"
)

print(
    "final_hybrid_unlabeled_predictions.csv"
)

print(
    "final_hybrid_mismatches.csv"
)