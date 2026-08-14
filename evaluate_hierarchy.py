import pandas as pd


# ============================================================
# 1. DOSYALAR
# ============================================================

PREDICTION_FILES = {
    "E5": "subject_predictions_E5.csv",
    "BGE-M3": "subject_predictions_BGE-M3.csv",
    "Qwen3": "subject_predictions_Qwen3.csv",
}

TOP_K_VALUES = [1, 3, 5]


# ============================================================
# 2. KONU YOLUNU PARÇALA
# ============================================================
#
# Örnek:
#
# Fen > Mühendislik > Bilgisayar Bilimleri, Yapay Zeka
#
# main = Fen
# sub  = Mühendislik
# leaf = Bilgisayar Bilimleri, Yapay Zeka
#

def parse_path(path):

    if pd.isna(path):
        return "", "", ""

    parts = [
        part.strip()
        for part in str(path).split(">")
        if part.strip()
    ]

    main = parts[0] if len(parts) >= 1 else ""
    sub = parts[1] if len(parts) >= 2 else ""
    leaf = " > ".join(parts[2:]) if len(parts) >= 3 else ""

    return main, sub, leaf


# ============================================================
# 3. GERÇEK ETİKETLERİ PARÇALA
# ============================================================

def get_true_paths(value):

    if pd.isna(value):
        return []

    return [
        path.strip()
        for path in str(value).split("||")
        if path.strip()
    ]


# ============================================================
# 4. HİYERARŞİK BAŞARI HESABI
# ============================================================

def evaluate_level(
    true_paths,
    predicted_paths,
    level
):

    true_values = set()
    predicted_values = set()

    # Gerçek değerler
    for path in true_paths:

        main, sub, leaf = parse_path(path)

        if level == "main":
            value = main

        elif level == "sub":
            value = sub

        else:
            value = leaf

        if value:
            true_values.add(value)


    # Tahmin edilen değerler
    for path in predicted_paths:

        main, sub, leaf = parse_path(path)

        if level == "main":
            value = main

        elif level == "sub":
            value = sub

        else:
            value = leaf

        if value:
            predicted_values.add(value)


    if not true_values:
        return 0, 0, 0, 0


    correct = len(
        true_values.intersection(
            predicted_values
        )
    )


    # Precision (kesinlik)
    precision = (
        correct / len(predicted_values)
        if predicted_values
        else 0
    )


    # Recall (duyarlılık)
    recall = (
        correct / len(true_values)
        if true_values
        else 0
    )


    # F1 Score (F1 skoru)
    if precision + recall > 0:

        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )

    else:
        f1 = 0


    # Hit Rate:
    # Bu seviyede en az bir doğru var mı?
    hit = 1 if correct > 0 else 0


    return (
        precision,
        recall,
        f1,
        hit
    )


# ============================================================
# 5. SONUÇLAR
# ============================================================

results = []


# ============================================================
# 6. HER MODELİ DEĞERLENDİR
# ============================================================

for model_name, file_name in PREDICTION_FILES.items():

    print("\n")
    print("=" * 80)
    print("MODEL:", model_name)
    print("=" * 80)

    df = pd.read_csv(
        file_name,
        encoding="utf-8-sig"
    )

    print(
        "Test makalesi:",
        len(df)
    )


    for k in TOP_K_VALUES:

        metrics = {
            "main": {
                "precision": [],
                "recall": [],
                "f1": [],
                "hit": []
            },

            "sub": {
                "precision": [],
                "recall": [],
                "f1": [],
                "hit": []
            },

            "leaf": {
                "precision": [],
                "recall": [],
                "f1": [],
                "hit": []
            }
        }


        for _, row in df.iterrows():

            true_paths = get_true_paths(
                row["true_subjects"]
            )


            # --------------------------------------------
            # İlk K tahmini al
            # --------------------------------------------

            predicted_paths = []

            for i in range(1, k + 1):

                column = f"prediction_{i}"

                if column not in df.columns:
                    continue

                value = row[column]

                if pd.notna(value):

                    value = str(value).strip()

                    if value:
                        predicted_paths.append(
                            value
                        )


            # --------------------------------------------
            # Üç seviyeyi ayrı ayrı değerlendir
            # --------------------------------------------

            for level in [
                "main",
                "sub",
                "leaf"
            ]:

                (
                    precision,
                    recall,
                    f1,
                    hit
                ) = evaluate_level(
                    true_paths,
                    predicted_paths,
                    level
                )

                metrics[level][
                    "precision"
                ].append(precision)

                metrics[level][
                    "recall"
                ].append(recall)

                metrics[level][
                    "f1"
                ].append(f1)

                metrics[level][
                    "hit"
                ].append(hit)


        # --------------------------------------------
        # ORTALAMA SONUÇLAR
        # --------------------------------------------

        for level in [
            "main",
            "sub",
            "leaf"
        ]:

            precision = sum(
                metrics[level]["precision"]
            ) / len(
                metrics[level]["precision"]
            )

            recall = sum(
                metrics[level]["recall"]
            ) / len(
                metrics[level]["recall"]
            )

            f1 = sum(
                metrics[level]["f1"]
            ) / len(
                metrics[level]["f1"]
            )

            hit = sum(
                metrics[level]["hit"]
            ) / len(
                metrics[level]["hit"]
            )


            results.append({

                "Model": model_name,

                "Top_K": k,

                "Level": level,

                "Precision": round(
                    precision,
                    4
                ),

                "Recall": round(
                    recall,
                    4
                ),

                "F1": round(
                    f1,
                    4
                ),

                "Hit_Rate": round(
                    hit,
                    4
                )
            })


# ============================================================
# 7. SONUÇ TABLOSU
# ============================================================

results_df = pd.DataFrame(
    results
)

print("\n\n")
print("=" * 110)

print(
    "HIERARCHICAL SUBJECT EVALUATION "
    "(HİYERARŞİK KONU DEĞERLENDİRMESİ)"
)

print("=" * 110)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 8. ÖZET TABLO - TOP 3
# ============================================================

top3 = results_df[
    results_df["Top_K"] == 3
].copy()

print("\n\n")
print("=" * 90)
print(
    "TOP-3 HİYERARŞİK ÖZET"
)
print("=" * 90)

print(
    top3[
        [
            "Model",
            "Level",
            "Precision",
            "Recall",
            "F1",
            "Hit_Rate"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 9. CSV KAYDET
# ============================================================

results_df.to_csv(
    "hierarchical_evaluation_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nSonuç dosyası oluşturuldu:"
    " hierarchical_evaluation_results.csv"
)