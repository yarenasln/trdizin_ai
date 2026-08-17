import pandas as pd
import numpy as np

INPUT_FILE = "kmeans_leave_one_out_topics.csv"

# Denenecek değerler
THRESHOLDS = np.arange(0.10, 0.81, 0.05)
MIN_SUPPORT_VALUES = [1, 2, 3, 4, 5]

df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")


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
# ETİKETLİ MAKALELER
# ============================================================

article_groups = []

for article_id, group in df.groupby("article_id"):

    real_subjects = parse_subjects(
        group.iloc[0]["current_trdizin_subjects"]
    )

    if not real_subjects:
        continue

    article_groups.append(
        (article_id, group, real_subjects)
    )


print("Değerlendirilen makale:", len(article_groups))


# ============================================================
# SUPPORT + THRESHOLD TARAMASI
# ============================================================

results = []

for min_support in MIN_SUPPORT_VALUES:

    for threshold in THRESHOLDS:

        tp = 0
        fp = 0
        fn = 0

        hit_count = 0
        exact_count = 0

        predicted_counts = []

        for article_id, group, real_subjects in article_groups:

            selected = group[
                (group["support_ratio"] >= threshold)
                &
                (group["support_count"] >= min_support)
            ]

            predicted_subjects = set(
                selected["candidate_subject"]
                .dropna()
                .astype(str)
            )

            predicted_counts.append(
                len(predicted_subjects)
            )

            # -------------------------------
            # TP / FP / FN
            # -------------------------------

            tp += len(
                predicted_subjects & real_subjects
            )

            fp += len(
                predicted_subjects - real_subjects
            )

            fn += len(
                real_subjects - predicted_subjects
            )

            # En az bir doğru konu
            if predicted_subjects & real_subjects:
                hit_count += 1

            # Bütün etiketler birebir aynı
            if predicted_subjects == real_subjects:
                exact_count += 1


        # ====================================================
        # METRİKLER
        # ====================================================

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        evaluated = len(article_groups)

        hit_rate = (
            hit_count / evaluated
            if evaluated else 0
        )

        exact_match = (
            exact_count / evaluated
            if evaluated else 0
        )

        avg_predicted = (
            np.mean(predicted_counts)
            if predicted_counts
            else 0
        )

        no_prediction_ratio = (
            np.mean(
                np.array(predicted_counts) == 0
            )
            if predicted_counts
            else 0
        )

        results.append({
            "Min_Support": min_support,
            "Threshold": round(float(threshold), 2),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
            "Hit_Rate": round(hit_rate, 4),
            "Exact_Match": round(exact_match, 4),
            "Avg_Predicted_Labels": round(avg_predicted, 2),
            "No_Prediction_Ratio": round(no_prediction_ratio, 4),
            "TP": tp,
            "FP": fp,
            "FN": fn
        })


results_df = pd.DataFrame(results)


# ============================================================
# HER SUPPORT İÇİN EN İYİ F1
# ============================================================

best_per_support = (
    results_df
    .sort_values(
        "F1",
        ascending=False
    )
    .groupby(
        "Min_Support",
        as_index=False
    )
    .first()
    .sort_values(
        "F1",
        ascending=False
    )
)


print("\n" + "=" * 110)
print("HER MIN_SUPPORT İÇİN EN İYİ SONUÇ")
print("=" * 110)

print(
    best_per_support.to_string(index=False)
)


# ============================================================
# GENEL EN İYİ
# ============================================================

best = results_df.loc[
    results_df["F1"].idxmax()
]

print("\n" + "=" * 110)
print("GENEL OLARAK EN İYİ SUPPORT + THRESHOLD")
print("=" * 110)

print(
    best.to_string()
)


# ============================================================
# EN İYİ 10 KOMBİNASYON
# ============================================================

print("\n" + "=" * 110)
print("EN İYİ 10 KOMBİNASYON")
print("=" * 110)

top10 = (
    results_df
    .sort_values(
        "F1",
        ascending=False
    )
    .head(10)
)

print(
    top10.to_string(index=False)
)


# ============================================================
# KAYDET
# ============================================================

results_df.to_csv(
    "kmeans_support_threshold_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nDosya oluşturuldu:"
    " kmeans_support_threshold_results.csv"
)