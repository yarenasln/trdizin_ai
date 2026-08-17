import os
import pandas as pd


CLUSTER_FILE = "results/final_article_clusters.csv"
SUBJECT_FILE = "trdizin_subject_hierarchy.csv"

OUTPUT_FILE = "results/final_coherent_cluster_profiles.csv"


clusters = pd.read_csv(
    CLUSTER_FILE,
    encoding="utf-8-sig"
)

subjects = pd.read_csv(
    SUBJECT_FILE,
    encoding="utf-8-sig"
)


subjects["main_field"] = (
    subjects["main_field"]
    .fillna("")
    .astype(str)
    .str.strip()
)

subjects["sub_field"] = (
    subjects["sub_field"]
    .fillna("")
    .astype(str)
    .str.strip()
)

subjects["leaf_subject"] = (
    subjects["leaf_subject"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# Sadece gerçek leaf konu kayıtları
subjects = subjects[
    subjects["leaf_subject"] != ""
].copy()


merged = clusters.merge(
    subjects[
        [
            "article_id",
            "main_field",
            "sub_field",
            "leaf_subject",
            "subject_fullname"
        ]
    ],
    on="article_id",
    how="left"
)


print("=" * 110)
print("HİYERARŞİK OLARAK TUTARLI CLUSTER PROFİLLERİ")
print("=" * 110)

print(
    "Makale:",
    clusters["article_id"].nunique()
)

print(
    "Cluster:",
    clusters["cluster_id"].nunique()
)


rows = []


for cluster_id, group in merged.groupby(
    "cluster_id"
):

    cluster_size = (
        group["article_id"]
        .nunique()
    )


    # ========================================================
    # 1. ANA ALAN
    # ========================================================

    valid_main = group[
        group["main_field"] != ""
    ]


    main_counts = (
        valid_main
        .groupby("main_field")["article_id"]
        .nunique()
        .sort_values(
            ascending=False
        )
    )


    if main_counts.empty:

        dominant_main = ""
        main_count = 0
        main_ratio = 0

    else:

        dominant_main = (
            main_counts.index[0]
        )

        main_count = int(
            main_counts.iloc[0]
        )

        main_ratio = (
            main_count
            /
            cluster_size
        )


    # ========================================================
    # 2. ALT ALAN
    # Sadece dominant ana alan içinden seç
    # ========================================================

    main_filtered = group[
        group["main_field"]
        ==
        dominant_main
    ]


    valid_sub = main_filtered[
        main_filtered["sub_field"] != ""
    ]


    sub_counts = (
        valid_sub
        .groupby("sub_field")["article_id"]
        .nunique()
        .sort_values(
            ascending=False
        )
    )


    if sub_counts.empty:

        dominant_sub = ""
        sub_count = 0
        sub_ratio = 0

    else:

        dominant_sub = (
            sub_counts.index[0]
        )

        sub_count = int(
            sub_counts.iloc[0]
        )

        # Ana alan içindeki oran
        sub_ratio = (
            sub_count
            /
            main_count
            if main_count > 0
            else 0
        )


    # ========================================================
    # 3. LEAF SUBJECT
    # Sadece dominant main + dominant sub içinden seç
    # ========================================================

    branch_filtered = group[
        (
            group["main_field"]
            ==
            dominant_main
        )
        &
        (
            group["sub_field"]
            ==
            dominant_sub
        )
    ]


    leaf_counts = (
        branch_filtered[
            branch_filtered[
                "subject_fullname"
            ].notna()
        ]
        .groupby(
            "subject_fullname"
        )["article_id"]
        .nunique()
        .sort_values(
            ascending=False
        )
    )


    if leaf_counts.empty:

        top_subject = ""
        leaf_count = 0
        leaf_ratio = 0

    else:

        top_subject = (
            leaf_counts.index[0]
        )

        leaf_count = int(
            leaf_counts.iloc[0]
        )

        # Seçilmiş alt alan içindeki oran
        leaf_ratio = (
            leaf_count
            /
            sub_count
            if sub_count > 0
            else 0
        )


    # ========================================================
    # İKİNCİ VE ÜÇÜNCÜ LEAF
    # ========================================================

    second_subject = (
        leaf_counts.index[1]
        if len(leaf_counts) >= 2
        else ""
    )

    third_subject = (
        leaf_counts.index[2]
        if len(leaf_counts) >= 3
        else ""
    )


    rows.append(
        {
            "cluster_id":
                cluster_id,

            "cluster_size":
                cluster_size,

            "dominant_main_field":
                dominant_main,

            "main_support":
                main_count,

            "main_confidence":
                round(
                    main_ratio,
                    4
                ),

            "dominant_sub_field":
                dominant_sub,

            "sub_support":
                sub_count,

            "sub_confidence_within_main":
                round(
                    sub_ratio,
                    4
                ),

            "top_subject":
                top_subject,

            "leaf_support":
                leaf_count,

            "leaf_confidence_within_sub":
                round(
                    leaf_ratio,
                    4
                ),

            "second_subject":
                second_subject,

            "third_subject":
                third_subject
        }
    )


profiles = pd.DataFrame(
    rows
).sort_values(
    "cluster_id"
)


# ============================================================
# HİYERARŞİ KONTROLÜ
# ============================================================

def hierarchy_valid(row):

    subject = str(
        row["top_subject"]
    )

    if not subject:
        return True

    parts = [
        x.strip()
        for x in subject.split(">")
        if x.strip()
    ]

    if len(parts) < 3:
        return False

    return (
        parts[0]
        ==
        row["dominant_main_field"]
        and
        parts[1]
        ==
        row["dominant_sub_field"]
    )


profiles[
    "hierarchy_consistent"
] = profiles.apply(
    hierarchy_valid,
    axis=1
)


print("\n" + "=" * 110)
print("GENEL SONUÇ")
print("=" * 110)

print(
    "Profil sayısı:",
    len(profiles)
)

print(
    "Hiyerarşisi tutarlı:",
    int(
        profiles[
            "hierarchy_consistent"
        ].sum()
    ),
    "/",
    len(profiles)
)

print(
    "Ortalama ana alan confidence:",
    round(
        profiles[
            "main_confidence"
        ].mean(),
        4
    )
)

print(
    "Ortalama alt alan confidence:",
    round(
        profiles[
            "sub_confidence_within_main"
        ].mean(),
        4
    )
)

print(
    "Ortalama leaf confidence:",
    round(
        profiles[
            "leaf_confidence_within_sub"
        ].mean(),
        4
    )
)


# ============================================================
# ÖRNEKLER
# ============================================================

print("\n" + "=" * 130)
print("İLK 20 CLUSTER")
print("=" * 130)

print(
    profiles[
        [
            "cluster_id",
            "cluster_size",
            "dominant_main_field",
            "main_confidence",
            "dominant_sub_field",
            "sub_confidence_within_main",
            "top_subject",
            "leaf_confidence_within_sub",
            "hierarchy_consistent"
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# KAYDET
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)

profiles.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


print(
    "\nDosya oluşturuldu:",
    OUTPUT_FILE
)