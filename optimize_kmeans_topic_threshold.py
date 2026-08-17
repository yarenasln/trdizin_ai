import pandas as pd
import numpy as np


# ============================================================
# AYARLAR
# ============================================================

INPUT_FILE = "kmeans_leave_one_out_topics.csv"

THRESHOLDS = np.arange(0.10, 0.81, 0.05)

MIN_SUPPORT = 2


# ============================================================
# VERİYİ OKU
# ============================================================

df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8-sig"
)

print("Toplam aday satırı:", len(df))


# ============================================================
# GERÇEK ETİKETLERİ PARÇALA
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
# SADECE GERÇEK ETİKETİ BULUNAN MAKALELER
# ============================================================

article_groups = []

for article_id, group in df.groupby("article_id"):

    real_subjects = parse_subjects(
        group.iloc[0]["current_trdizin_subjects"]
    )

    if not real_subjects:
        continue

    article_groups.append(
        (
            article_id,
            group,
            real_subjects
        )
    )


print(
    "Değerlendirilen etiketli makale:",
    len(article_groups)
)


# ============================================================
# THRESHOLD DENEYİ
# ============================================================

results = []


for threshold in THRESHOLDS:

    tp = 0
    fp = 0
    fn = 0

    predicted_label_counts = []

    exact_match_count = 0
    at_least_one_correct = 0


    for article_id, group, real_subjects in article_groups:

        selected = group[
            (group["support_ratio"] >= threshold)
            &
            (group["support_count"] >= MIN_SUPPORT)
        ]

        predicted_subjects = set(
            selected["candidate_subject"]
            .dropna()
            .astype(str)
        )

        predicted_label_counts.append(
            len(predicted_subjects)
        )

        # --------------------------------------------
        # MULTI-LABEL METRİKLER
        # --------------------------------------------

        tp += len(
            predicted_subjects
            & real_subjects
        )

        fp += len(
            predicted_subjects
            - real_subjects
        )

        fn += len(
            real_subjects
            - predicted_subjects
        )

        if predicted_subjects == real_subjects:
            exact_match_count += 1

        if predicted_subjects & real_subjects:
            at_least_one_correct += 1


    # ========================================================
    # METRİKLER
    # ========================================================

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

    exact_match = (
        exact_match_count / evaluated
        if evaluated > 0
        else 0
    )

    hit_rate = (
        at_least_one_correct / evaluated
        if evaluated > 0
        else 0
    )

    average_predicted_labels = (
        np.mean(predicted_label_counts)
        if predicted_label_counts
        else 0
    )

    no_prediction_ratio = (
        np.mean(
            np.array(predicted_label_counts) == 0
        )
        if predicted_label_counts
        else 0
    )


    results.append({

        "Threshold":
            round(float(threshold), 2),

        "Min_Support":
            MIN_SUPPORT,

        "Precision":
            round(precision, 4),

        "Recall":
            round(recall, 4),

        "F1":
            round(f1, 4),

        "Hit_Rate":
            round(hit_rate, 4),

        "Exact_Match":
            round(exact_match, 4),

        "Avg_Predicted_Labels":
            round(
                average_predicted_labels,
                2
            ),

        "No_Prediction_Ratio":
            round(
                no_prediction_ratio,
                4
            ),

        "TP": tp,
        "FP": fp,
        "FN": fn,

        "Evaluated_Articles":
            evaluated
    })


# ============================================================
# SONUÇ TABLOSU
# ============================================================

results_df = pd.DataFrame(
    results
)

print("\n")
print("=" * 120)
print("K-MEANS DİNAMİK KONU EŞİĞİ KARŞILAŞTIRMASI")
print("=" * 120)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# EN İYİ F1
# ============================================================

if not results_df.empty:

    best = results_df.loc[
        results_df["F1"].idxmax()
    ]

    print("\n")
    print("=" * 120)
    print("F1'E GÖRE EN İYİ EŞİK")
    print("=" * 120)

    print(
        best.to_string()
    )


# ============================================================
# DOSYAYA KAYDET
# ============================================================

results_df.to_csv(
    "kmeans_threshold_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nDosya oluşturuldu:"
    " kmeans_threshold_results.csv"
)