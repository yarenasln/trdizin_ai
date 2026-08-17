import pandas as pd


# ============================================================
# DOSYALAR
# ============================================================

CLUSTER_FILE = "kmeans_article_clusters.csv"
SUBJECT_FILE = "trdizin_subject_hierarchy.csv"

clusters = pd.read_csv(
    CLUSTER_FILE,
    encoding="utf-8-sig"
)

subjects = pd.read_csv(
    SUBJECT_FILE,
    encoding="utf-8-sig"
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

subjects = subjects[
    subjects["leaf_subject"] != ""
].copy()


# ============================================================
# MAKALE -> GERÇEK TR DİZİN KONULARI
# ============================================================

article_subjects = (
    subjects
    .groupby("article_id")["subject_fullname"]
    .apply(
        lambda x: set(
            x.dropna().astype(str)
        )
    )
    .to_dict()
)


# ============================================================
# CLUSTER -> MAKALELER
# ============================================================

cluster_articles = (
    clusters
    .groupby("cluster_id")["article_id"]
    .apply(list)
    .to_dict()
)


# ============================================================
# LEAVE-ONE-OUT
# ============================================================
#
# Makale X'i değerlendirirken:
#
# Makale X'in kendi etiketi kullanılmayacak.
#
# Sadece aynı cluster'daki DİĞER makalelerin
# konularına bakılacak.
# ============================================================

rows = []


for _, row in clusters.iterrows():

    article_id = row["article_id"]
    cluster_id = row["cluster_id"]

    members = cluster_articles[
        cluster_id
    ]

    # ----------------------------------------
    # Kendisi hariç komşu makaleler
    # ----------------------------------------

    other_articles = [
        x for x in members
        if x != article_id
    ]

    support_size = len(
        other_articles
    )

    subject_counts = {}


    # ----------------------------------------
    # Diğer makalelerin konularını say
    # ----------------------------------------

    for other_id in other_articles:

        labels = article_subjects.get(
            other_id,
            set()
        )

        for label in labels:

            subject_counts[label] = (
                subject_counts.get(
                    label,
                    0
                )
                + 1
            )


    # ----------------------------------------
    # Makalenin mevcut TR Dizin konuları
    # ----------------------------------------

    real_subjects = article_subjects.get(
        article_id,
        set()
    )


    # ----------------------------------------
    # Konuları desteğe göre sırala
    # ----------------------------------------

    sorted_subjects = sorted(
        subject_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )


    for rank, (subject, count) in enumerate(
        sorted_subjects,
        start=1
    ):

        ratio = (
            count / support_size
            if support_size > 0
            else 0
        )

        rows.append({

            "article_id":
                article_id,

            "cluster_id":
                cluster_id,

            "cluster_size":
                len(members),

            "other_article_count":
                support_size,

            "rank":
                rank,

            "candidate_subject":
                subject,

            "support_count":
                count,

            "support_ratio":
                round(ratio, 4),

            # Bu aday zaten makalenin
            # mevcut TR Dizin konularından biri mi?
            "exists_in_trdizin":
                subject in real_subjects,

            "current_trdizin_subjects":
                " || ".join(
                    sorted(real_subjects)
                )
        })


result = pd.DataFrame(
    rows
)


# ============================================================
# KAYDET
# ============================================================

result.to_csv(
    "kmeans_leave_one_out_topics.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# ÖRNEKLER
# ============================================================

print("\n" + "=" * 100)
print("K-MEANS LEAVE-ONE-OUT KONU ADAYLARI")
print("=" * 100)


# En az 5 başka makale tarafından desteklenen
# kümelerden birkaç örnek gösterelim.

valid = result[
    result["other_article_count"] >= 5
]


sample_ids = (
    valid["article_id"]
    .drop_duplicates()
    .head(10)
)


for article_id in sample_ids:

    article_rows = result[
        result["article_id"] == article_id
    ].sort_values(
        "support_ratio",
        ascending=False
    )

    if article_rows.empty:
        continue

    first = article_rows.iloc[0]

    print("\n" + "-" * 100)

    print(
        "Article ID:",
        article_id
    )

    print(
        "Cluster:",
        first["cluster_id"]
    )

    print(
        "Diğer makale sayısı:",
        first["other_article_count"]
    )

    print(
        "Mevcut TR Dizin konuları:"
    )

    print(
        first["current_trdizin_subjects"]
    )

    print("\nK-Means kümesinden gelen konu adayları:")

    # Burada sadece incelemek için ilk 10'u
    # yazdırıyoruz.
    # Bu TOP-10 tahmini değildir.
    for _, candidate in article_rows.head(10).iterrows():

        status = (
            "UYUŞUYOR"
            if candidate["exists_in_trdizin"]
            else "FARKLI"
        )

        print(
            f"  {candidate['candidate_subject']}"
            f" | destek="
            f"{candidate['support_count']}/"
            f"{candidate['other_article_count']}"
            f" | oran="
            f"{candidate['support_ratio']:.4f}"
            f" | {status}"
        )


print("\n" + "=" * 100)
print(
    "Dosya oluşturuldu:"
    " kmeans_leave_one_out_topics.csv"
)