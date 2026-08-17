import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# DOSYALAR
# ============================================================

SUBJECT_METADATA_FILE = "Qwen3_subject_metadata.csv"
SUBJECT_EMBEDDING_FILE = "embeddings/Qwen3_subject_embeddings.npy"

OUTPUT_PAIRS = "subject_similarity_pairs.csv"
OUTPUT_SUMMARY = "subject_similarity_summary.csv"


# ============================================================
# VERİLERİ OKU
# ============================================================

subjects = pd.read_csv(
    SUBJECT_METADATA_FILE,
    encoding="utf-8-sig"
)

embeddings = np.load(
    SUBJECT_EMBEDDING_FILE
).astype(np.float32)


print("=" * 110)
print("TR DİZİN KONU-KONU SEMANTIC SIMILARITY ANALİZİ")
print("=" * 110)

print("Konu sayısı:", len(subjects))
print("Embedding shape:", embeddings.shape)


# ============================================================
# SUBJECT PARÇALAMA
# ============================================================

def parse_hierarchy(subject):

    parts = [
        x.strip()
        for x in str(subject).split(">")
    ]

    root = (
        parts[0]
        if len(parts) > 0
        else ""
    )

    level2 = (
        parts[1]
        if len(parts) > 1
        else ""
    )

    leaf = (
        parts[-1]
        if len(parts) > 0
        else ""
    )

    return root, level2, leaf


# ============================================================
# COSINE SIMILARITY MATRİSİ
# ============================================================

similarity_matrix = cosine_similarity(
    embeddings
)

print(
    "Similarity matrix:",
    similarity_matrix.shape
)


# ============================================================
# TÜM BENZERSİZ KONU ÇİFTLERİ
# ============================================================

rows = []


for i in range(len(subjects)):

    subject_a = str(
        subjects.iloc[i][
            "subject_fullname"
        ]
    ).strip()

    root_a, level2_a, leaf_a = parse_hierarchy(
        subject_a
    )


    for j in range(i + 1, len(subjects)):

        subject_b = str(
            subjects.iloc[j][
                "subject_fullname"
            ]
        ).strip()

        root_b, level2_b, leaf_b = parse_hierarchy(
            subject_b
        )


        similarity = float(
            similarity_matrix[i, j]
        )


        # ----------------------------------------------------
        # TAKSONOMİK İLİŞKİ
        # ----------------------------------------------------

        if (
            root_a == root_b
            and
            level2_a == level2_b
        ):

            relation = "AYNI_LEVEL2"

        elif root_a == root_b:

            relation = "AYNI_ROOT"

        else:

            relation = "FARKLI_ROOT"


        rows.append(
            {
                "subject_a": subject_a,
                "subject_b": subject_b,

                "root_a": root_a,
                "root_b": root_b,

                "level2_a": level2_a,
                "level2_b": level2_b,

                "relation": relation,

                "similarity": round(
                    similarity,
                    4
                )
            }
        )


pairs = pd.DataFrame(
    rows
)


print(
    "Toplam benzersiz konu çifti:",
    len(pairs)
)


# ============================================================
# GENEL DAĞILIM
# ============================================================

print("\n" + "=" * 110)
print("GENEL KONU BENZERLİĞİ DAĞILIMI")
print("=" * 110)


print(
    "Ortalama:",
    round(
        pairs["similarity"].mean(),
        4
    )
)

print(
    "Medyan:",
    round(
        pairs["similarity"].median(),
        4
    )
)

print(
    "Minimum:",
    round(
        pairs["similarity"].min(),
        4
    )
)

print(
    "Maksimum:",
    round(
        pairs["similarity"].max(),
        4
    )
)


print("\nPercentiller:")

for p in [
    10,
    25,
    50,
    75,
    90,
    95,
    99
]:

    value = np.percentile(
        pairs["similarity"],
        p
    )

    print(
        f"P{p}:",
        round(
            float(value),
            4
        )
    )


# ============================================================
# TAKSONOMİK GRUPLARA GÖRE
# ============================================================

print("\n" + "=" * 125)
print("TAKSONOMİK İLİŞKİYE GÖRE BENZERLİK")
print("=" * 125)


summary_rows = []


for relation, group in pairs.groupby(
    "relation"
):

    stats = {
        "Relation": relation,

        "Count":
            len(group),

        "Mean":
            round(
                group[
                    "similarity"
                ].mean(),
                4
            ),

        "Median":
            round(
                group[
                    "similarity"
                ].median(),
                4
            ),

        "P25":
            round(
                np.percentile(
                    group[
                        "similarity"
                    ],
                    25
                ),
                4
            ),

        "P75":
            round(
                np.percentile(
                    group[
                        "similarity"
                    ],
                    75
                ),
                4
            ),

        "P90":
            round(
                np.percentile(
                    group[
                        "similarity"
                    ],
                    90
                ),
                4
            ),

        "Min":
            round(
                group[
                    "similarity"
                ].min(),
                4
            ),

        "Max":
            round(
                group[
                    "similarity"
                ].max(),
                4
            )
    }


    summary_rows.append(
        stats
    )


summary = pd.DataFrame(
    summary_rows
)


print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# THRESHOLD ANALİZİ
# ============================================================

print("\n" + "=" * 135)
print("THRESHOLD ANALİZİ")
print("=" * 135)


threshold_rows = []


for threshold in [
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90
]:

    print(
        f"\nSimilarity >= {threshold:.2f}"
    )


    for relation in [
        "AYNI_LEVEL2",
        "AYNI_ROOT",
        "FARKLI_ROOT"
    ]:

        group = pairs[
            pairs[
                "relation"
            ]
            ==
            relation
        ]


        count = int(
            (
                group[
                    "similarity"
                ]
                >=
                threshold
            ).sum()
        )


        ratio = (
            count
            /
            len(group)
            if len(group)
            else 0
        )


        print(
            f"  {relation:15s}: "
            f"{count:5d} / "
            f"{len(group):5d} "
            f"({ratio * 100:6.2f}%)"
        )


        threshold_rows.append(
            {
                "Threshold":
                    threshold,

                "Relation":
                    relation,

                "Count":
                    count,

                "Total":
                    len(group),

                "Ratio":
                    round(
                        ratio,
                        4
                    )
            }
        )


threshold_df = pd.DataFrame(
    threshold_rows
)


# ============================================================
# EN BENZER KONU ÇİFTLERİ
# ============================================================

print("\n" + "=" * 140)
print("EN BENZER 25 FARKLI KONU ÇİFTİ")
print("=" * 140)


print(
    pairs
    .sort_values(
        "similarity",
        ascending=False
    )
    [
        [
            "subject_a",
            "subject_b",
            "relation",
            "similarity"
        ]
    ]
    .head(25)
    .to_string(
        index=False
    )
)


# ============================================================
# 0.75 ÜSTÜNDEKİ FARKLI ROOT'LAR
# ============================================================

cross_root_high = pairs[
    (
        pairs[
            "relation"
        ]
        ==
        "FARKLI_ROOT"
    )
    &
    (
        pairs[
            "similarity"
        ]
        >=
        0.75
    )
].copy()


print("\n" + "=" * 140)
print("0.75 ÜSTÜ FARKLI ROOT KONU ÇİFTLERİ")
print("=" * 140)


print(
    "Çift sayısı:",
    len(
        cross_root_high
    )
)


if not cross_root_high.empty:

    print(
        cross_root_high
        .sort_values(
            "similarity",
            ascending=False
        )
        [
            [
                "subject_a",
                "subject_b",
                "similarity"
            ]
        ]
        .head(30)
        .to_string(
            index=False
        )
    )


# ============================================================
# 0.75 EŞİĞİNİN AYIRICILIĞI
# ============================================================

threshold = 0.75


same_level2 = pairs[
    pairs[
        "relation"
    ]
    ==
    "AYNI_LEVEL2"
]


different_root = pairs[
    pairs[
        "relation"
    ]
    ==
    "FARKLI_ROOT"
]


same_level2_above = (
    same_level2[
        "similarity"
    ]
    >=
    threshold
).mean()


different_root_above = (
    different_root[
        "similarity"
    ]
    >=
    threshold
).mean()


print("\n" + "=" * 120)
print("0.75 EŞİĞİ AYIRICILIK ÖZETİ")
print("=" * 120)


print(
    "Aynı Level-2 konuların >=0.75 oranı:",
    round(
        same_level2_above,
        4
    )
)

print(
    "Farklı root konuların >=0.75 oranı:",
    round(
        different_root_above,
        4
    )
)


# ============================================================
# DOSYALARI KAYDET
# ============================================================

pairs.to_csv(
    OUTPUT_PAIRS,
    index=False,
    encoding="utf-8-sig"
)


summary.to_csv(
    OUTPUT_SUMMARY,
    index=False,
    encoding="utf-8-sig"
)


threshold_df.to_csv(
    "subject_similarity_threshold_analysis.csv",
    index=False,
    encoding="utf-8-sig"
)


print("\n" + "=" * 110)
print("DOSYALAR OLUŞTURULDU")
print("=" * 110)

print(OUTPUT_PAIRS)
print(OUTPUT_SUMMARY)
print(
    "subject_similarity_threshold_analysis.csv"
)