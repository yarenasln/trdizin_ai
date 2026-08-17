import pandas as pd


INPUT_FILE = "final_hybrid_mismatches.csv"
OUTPUT_FILE = "manual_review_10_articles.csv"


df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8-sig"
)


# Her makaleden sadece bir kez seçmek için
article_level = (
    df
    .sort_values(
        [
            "article_id",
            "prediction_rank"
        ]
    )
    .groupby(
        "article_id",
        as_index=False
    )
    .first()
)


# Farklı hata tiplerinden örnek almak için
# basitçe ilk 10'u seçiyoruz.
# İstersen sonra daha dengeli örnekleme yaparız.
sample_ids = (
    article_level[
        "article_id"
    ]
    .head(10)
    .tolist()
)


sample = df[
    df[
        "article_id"
    ]
    .isin(
        sample_ids
    )
].copy()


sample = sample[
    [
        "article_id",
        "title",
        "trdizin_subjects",
        "predicted_subject",
        "prediction_rank",
        "kmeans_score",
        "semantic_score",
        "final_score",
        "support_count",
        "best_similarity",
        "selected_cluster_count"
    ]
]


sample.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


print("=" * 110)
print("MANUEL İNCELEME İÇİN 10 UYUŞMAZ MAKALE")
print("=" * 110)


for article_id, group in sample.groupby(
    "article_id"
):

    first = group.iloc[0]

    print("\n" + "-" * 110)

    print(
        "Article ID:",
        article_id
    )

    print(
        "Başlık:",
        first["title"]
    )

    print(
        "TR Dizin:",
        first["trdizin_subjects"]
    )

    print(
        "Bizim sistem:"
    )

    for _, row in group.sort_values(
        "prediction_rank"
    ).iterrows():

        print(
            f"  - {row['predicted_subject']}"
            f" | KMeans={row['kmeans_score']}"
            f" | Semantic={row['semantic_score']}"
            f" | Final={row['final_score']}"
        )


print(
    "\nDosya oluşturuldu:",
    OUTPUT_FILE
)