import pandas as pd


# ============================================================
# DOSYALAR
# ============================================================

PROFILE_FILE = "results/final_coherent_cluster_profiles.csv"
CLUSTER_FILE = "results/final_article_clusters.csv"

OUTPUT_FILE = "results/hierarchy_confidence_threshold_analysis.csv"


# ============================================================
# VERİLERİ OKU
# ============================================================

profiles = pd.read_csv(
    PROFILE_FILE,
    encoding="utf-8-sig"
)

articles = pd.read_csv(
    CLUSTER_FILE,
    encoding="utf-8-sig"
)


print("=" * 110)
print("HİYERARŞİ CONFIDENCE THRESHOLD ANALİZİ")
print("=" * 110)

print("Cluster sayısı:", len(profiles))
print("Makale sayısı:", articles["article_id"].nunique())


# ============================================================
# 1. CONFIDENCE DAĞILIMLARI
# ============================================================

confidence_columns = {
    "MAIN": "main_confidence",
    "SUB": "sub_confidence_within_main",
    "LEAF": "leaf_confidence_within_sub"
}


print("\n" + "=" * 110)
print("CONFIDENCE DAĞILIMLARI")
print("=" * 110)


for level, column in confidence_columns.items():

    values = profiles[column].dropna()

    print(f"\n{level}")
    print("-" * 60)

    print(
        "Ortalama:",
        round(values.mean(), 4)
    )

    print(
        "Medyan:",
        round(values.median(), 4)
    )

    print(
        "Minimum:",
        round(values.min(), 4)
    )

    print(
        "Maksimum:",
        round(values.max(), 4)
    )

    for percentile in [
        5,
        10,
        25,
        50,
        75,
        90,
        95
    ]:

        value = values.quantile(
            percentile / 100
        )

        print(
            f"P{percentile}:",
            round(value, 4)
        )


# ============================================================
# 2. TEST EDİLECEK EŞİKLER
# ============================================================

thresholds = {
    "MAIN": [
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90
    ],

    "SUB": [
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90
    ],

    "LEAF": [
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.50,
        0.60
    ]
}


# ============================================================
# 3. MAKALELERİ CLUSTER PROFİLİYLE BİRLEŞTİR
# ============================================================

article_profiles = articles.merge(
    profiles,
    on="cluster_id",
    how="left"
)


print("\nBirleşmiş makale:", len(article_profiles))


# ============================================================
# 4. HER THRESHOLD İÇİN
#    CLUSTER + MAKALE COVERAGE
# ============================================================

results = []


for level, column in confidence_columns.items():

    print("\n" + "=" * 110)
    print(f"{level} THRESHOLD ANALİZİ")
    print("=" * 110)

    for threshold in thresholds[level]:

        # ----------------------------------------------------
        # Cluster bazında
        # ----------------------------------------------------

        cluster_found = (
            profiles[column]
            >=
            threshold
        )

        found_clusters = int(
            cluster_found.sum()
        )

        missing_clusters = (
            len(profiles)
            -
            found_clusters
        )

        cluster_coverage = (
            found_clusters
            /
            len(profiles)
        )


        # ----------------------------------------------------
        # Makale bazında
        # ----------------------------------------------------

        article_found = (
            article_profiles[column]
            >=
            threshold
        )

        found_articles = int(
            article_found.sum()
        )

        missing_articles = (
            len(article_profiles)
            -
            found_articles
        )

        article_coverage = (
            found_articles
            /
            len(article_profiles)
        )


        results.append(
            {
                "Level":
                    level,

                "Threshold":
                    threshold,

                "Found_Clusters":
                    found_clusters,

                "Missing_Clusters":
                    missing_clusters,

                "Cluster_Coverage":
                    round(
                        cluster_coverage,
                        4
                    ),

                "Found_Articles":
                    found_articles,

                "Missing_Articles":
                    missing_articles,

                "Article_Coverage":
                    round(
                        article_coverage,
                        4
                    )
            }
        )


        print(
            f"Threshold >= {threshold:.2f}"
            f" | Cluster: "
            f"{found_clusters}/{len(profiles)}"
            f" ({cluster_coverage:.2%})"
            f" | Makale: "
            f"{found_articles}/{len(article_profiles)}"
            f" ({article_coverage:.2%})"
        )


# ============================================================
# 5. SONUÇ DATAFRAME
# ============================================================

result_df = pd.DataFrame(
    results
)


# ============================================================
# 6. ÖZELLİKLE İNCELEYECEĞİMİZ ADAY EŞİKLER
# ============================================================

candidate_thresholds = {
    "MAIN": 0.75,
    "SUB": 0.75,
    "LEAF": 0.30
}


print("\n" + "=" * 110)
print("ADAY EŞİKLERLE MAKale DURUMU")
print("=" * 110)


for level, threshold in candidate_thresholds.items():

    column = confidence_columns[level]

    found = (
        article_profiles[column]
        >=
        threshold
    )

    print(
        f"{level} >= {threshold:.2f}: "
        f"Bulunan={int(found.sum())} | "
        f"Bulunamayan={int((~found).sum())} | "
        f"Coverage={found.mean():.2%}"
    )


# ============================================================
# 7. ÜÇ SEVİYENİN BİRLİKTE DURUMU
# ============================================================

main_ok = (
    article_profiles[
        "main_confidence"
    ]
    >=
    candidate_thresholds["MAIN"]
)

sub_ok = (
    article_profiles[
        "sub_confidence_within_main"
    ]
    >=
    candidate_thresholds["SUB"]
)

leaf_ok = (
    article_profiles[
        "leaf_confidence_within_sub"
    ]
    >=
    candidate_thresholds["LEAF"]
)


all_ok = (
    main_ok
    &
    sub_ok
    &
    leaf_ok
)


print("\n" + "=" * 110)
print("ÜÇ SEVİYE BİRLİKTE")
print("=" * 110)

print(
    "Ana + Alt + Leaf bulunan:",
    int(all_ok.sum())
)

print(
    "En az bir seviye bulunamayan:",
    int((~all_ok).sum())
)

print(
    "Tam coverage:",
    f"{all_ok.mean():.2%}"
)


# ============================================================
# 8. CSV
# ============================================================

result_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


print("\nDosya oluşturuldu:", OUTPUT_FILE)