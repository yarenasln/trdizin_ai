import os
import pandas as pd


# ============================================================
# DOSYALAR
# ============================================================

CLUSTER_FILE = "results/final_article_clusters.csv"
SUBJECT_FILE = "trdizin_subject_hierarchy.csv"

OUTPUT_DIR = "results"

PROFILE_FILE = os.path.join(
    OUTPUT_DIR,
    "final_cluster_profiles.csv"
)

TOPIC_FILE = os.path.join(
    OUTPUT_DIR,
    "final_cluster_topic_distribution.csv"
)


# ============================================================
# VERİLERİ OKU
# ============================================================

clusters = pd.read_csv(
    CLUSTER_FILE,
    encoding="utf-8-sig"
)

subjects = pd.read_csv(
    SUBJECT_FILE,
    encoding="utf-8-sig"
)


print("=" * 110)
print("FINAL CLUSTER KONU PROFİLLERİ")
print("=" * 110)

print(
    "Cluster makalesi:",
    clusters["article_id"].nunique()
)

print(
    "Makale-konu ilişkisi:",
    len(subjects)
)


# ============================================================
# SADECE GERÇEK LEAF SUBJECT KAYITLARI
# ============================================================

subjects["leaf_subject"] = (
    subjects["leaf_subject"]
    .fillna("")
    .astype(str)
    .str.strip()
)

leaf_subjects = subjects[
    subjects["leaf_subject"] != ""
].copy()


# ============================================================
# CLUSTER + SUBJECT BİRLEŞTİR
# ============================================================

merged = clusters.merge(
    leaf_subjects[
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


print(
    "Birleşmiş satır:",
    len(merged)
)


# ============================================================
# CLUSTER PROFİLLERİ
# ============================================================

profile_rows = []
topic_rows = []


for cluster_id, group in merged.groupby(
    "cluster_id"
):

    cluster_article_ids = (
        group["article_id"]
        .drop_duplicates()
        .tolist()
    )

    cluster_size = len(
        cluster_article_ids
    )


    # --------------------------------------------------------
    # ETİKETLİ MAKALELER
    # --------------------------------------------------------

    labeled_article_ids = (
        group[
            group["subject_fullname"]
            .notna()
        ]["article_id"]
        .drop_duplicates()
    )

    labeled_count = len(
        labeled_article_ids
    )


    # --------------------------------------------------------
    # ANA ALAN DAĞILIMI
    # --------------------------------------------------------

    main_field_article_counts = (
        group
        .dropna(
            subset=["main_field"]
        )
        .groupby("main_field")["article_id"]
        .nunique()
        .sort_values(
            ascending=False
        )
    )


    if not main_field_article_counts.empty:

        dominant_main = (
            main_field_article_counts.index[0]
        )

        dominant_main_count = int(
            main_field_article_counts.iloc[0]
        )

        dominant_main_ratio = (
            dominant_main_count
            /
            cluster_size
        )

    else:

        dominant_main = ""
        dominant_main_count = 0
        dominant_main_ratio = 0


    # --------------------------------------------------------
    # ALT ALAN DAĞILIMI
    # --------------------------------------------------------

    sub_field_article_counts = (
        group
        .dropna(
            subset=["sub_field"]
        )
    )

    sub_field_article_counts = (
        sub_field_article_counts[
            sub_field_article_counts[
                "sub_field"
            ]
            .astype(str)
            .str.strip()
            != ""
        ]
        .groupby(
            [
                "main_field",
                "sub_field"
            ]
        )["article_id"]
        .nunique()
        .sort_values(
            ascending=False
        )
    )


    if not sub_field_article_counts.empty:

        dominant_sub_pair = (
            sub_field_article_counts.index[0]
        )

        dominant_sub_main = (
            dominant_sub_pair[0]
        )

        dominant_sub = (
            dominant_sub_pair[1]
        )

        dominant_sub_count = int(
            sub_field_article_counts.iloc[0]
        )

        dominant_sub_ratio = (
            dominant_sub_count
            /
            cluster_size
        )

    else:

        dominant_sub_main = ""
        dominant_sub = ""
        dominant_sub_count = 0
        dominant_sub_ratio = 0


    # --------------------------------------------------------
    # LEAF SUBJECT DAĞILIMI
    # --------------------------------------------------------

    leaf_counts = (
        group
        .dropna(
            subset=["subject_fullname"]
        )
        .groupby("subject_fullname")[
            "article_id"
        ]
        .nunique()
        .sort_values(
            ascending=False
        )
    )


    top_subject = ""
    top_subject_count = 0
    top_subject_ratio = 0

    second_subject = ""
    second_subject_count = 0
    second_subject_ratio = 0

    third_subject = ""
    third_subject_count = 0
    third_subject_ratio = 0


    if len(leaf_counts) >= 1:

        top_subject = (
            leaf_counts.index[0]
        )

        top_subject_count = int(
            leaf_counts.iloc[0]
        )

        top_subject_ratio = (
            top_subject_count
            /
            cluster_size
        )


    if len(leaf_counts) >= 2:

        second_subject = (
            leaf_counts.index[1]
        )

        second_subject_count = int(
            leaf_counts.iloc[1]
        )

        second_subject_ratio = (
            second_subject_count
            /
            cluster_size
        )


    if len(leaf_counts) >= 3:

        third_subject = (
            leaf_counts.index[2]
        )

        third_subject_count = int(
            leaf_counts.iloc[2]
        )

        third_subject_ratio = (
            third_subject_count
            /
            cluster_size
        )


    # --------------------------------------------------------
    # TÜM KONU DAĞILIMINI AYRI TABLOYA YAZ
    # --------------------------------------------------------

    for subject_name, count in (
        leaf_counts.items()
    ):

        topic_rows.append(
            {
                "cluster_id":
                    cluster_id,

                "cluster_size":
                    cluster_size,

                "subject_fullname":
                    subject_name,

                "article_support":
                    int(count),

                "support_ratio":
                    round(
                        count
                        /
                        cluster_size,
                        4
                    )
            }
        )


    # --------------------------------------------------------
    # CLUSTER ÖZET PROFİLİ
    # --------------------------------------------------------

    profile_rows.append(
        {
            "cluster_id":
                cluster_id,

            "cluster_size":
                cluster_size,

            "labeled_articles":
                labeled_count,

            "labeled_ratio":
                round(
                    labeled_count
                    /
                    cluster_size,
                    4
                ),

            "dominant_main_field":
                dominant_main,

            "dominant_main_count":
                dominant_main_count,

            "dominant_main_ratio":
                round(
                    dominant_main_ratio,
                    4
                ),

            "dominant_sub_main":
                dominant_sub_main,

            "dominant_sub_field":
                dominant_sub,

            "dominant_sub_count":
                dominant_sub_count,

            "dominant_sub_ratio":
                round(
                    dominant_sub_ratio,
                    4
                ),

            "top_subject":
                top_subject,

            "top_subject_count":
                top_subject_count,

            "top_subject_ratio":
                round(
                    top_subject_ratio,
                    4
                ),

            "second_subject":
                second_subject,

            "second_subject_count":
                second_subject_count,

            "second_subject_ratio":
                round(
                    second_subject_ratio,
                    4
                ),

            "third_subject":
                third_subject,

            "third_subject_count":
                third_subject_count,

            "third_subject_ratio":
                round(
                    third_subject_ratio,
                    4
                )
        }
    )


# ============================================================
# DATAFRAME
# ============================================================

profiles = pd.DataFrame(
    profile_rows
)

topics = pd.DataFrame(
    topic_rows
)


profiles = profiles.sort_values(
    "cluster_id"
)

topics = topics.sort_values(
    [
        "cluster_id",
        "article_support"
    ],
    ascending=[
        True,
        False
    ]
)


# ============================================================
# GENEL İSTATİSTİKLER
# ============================================================

print("\n" + "=" * 110)
print("CLUSTER PROFİL İSTATİSTİKLERİ")
print("=" * 110)

print(
    "Profil oluşturulan cluster:",
    len(profiles)
)

print(
    "Ortalama dominant ana alan oranı:",
    round(
        profiles[
            "dominant_main_ratio"
        ].mean(),
        4
    )
)

print(
    "Ortalama dominant alt alan oranı:",
    round(
        profiles[
            "dominant_sub_ratio"
        ].mean(),
        4
    )
)

print(
    "Ortalama en güçlü leaf konu oranı:",
    round(
        profiles[
            "top_subject_ratio"
        ].mean(),
        4
    )
)


# ============================================================
# ANA ALANI ÇOK NET OLAN CLUSTER'LAR
# ============================================================

print("\n" + "=" * 110)
print("ANA ALANI EN BELİRGİN 10 CLUSTER")
print("=" * 110)

print(
    profiles
    .sort_values(
        "dominant_main_ratio",
        ascending=False
    )
    [
        [
            "cluster_id",
            "cluster_size",
            "dominant_main_field",
            "dominant_main_ratio",
            "dominant_sub_field",
            "top_subject"
        ]
    ]
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================
# ANA ALANI EN KARIŞIK CLUSTER'LAR
# ============================================================

print("\n" + "=" * 110)
print("ANA ALANI EN KARIŞIK 10 CLUSTER")
print("=" * 110)

print(
    profiles
    .sort_values(
        "dominant_main_ratio",
        ascending=True
    )
    [
        [
            "cluster_id",
            "cluster_size",
            "dominant_main_field",
            "dominant_main_ratio",
            "dominant_sub_field",
            "top_subject"
        ]
    ]
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================
# KAYDET
# ============================================================

profiles.to_csv(
    PROFILE_FILE,
    index=False,
    encoding="utf-8-sig"
)

topics.to_csv(
    TOPIC_FILE,
    index=False,
    encoding="utf-8-sig"
)


print("\n" + "=" * 110)
print("DOSYALAR OLUŞTURULDU")
print("=" * 110)

print(PROFILE_FILE)
print(TOPIC_FILE)