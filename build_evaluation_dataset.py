import pandas as pd


# ==================================================
# 1. DOSYALAR
# ==================================================

TEXT_FILE = "real_trdizin_texts.csv"
SUBJECT_FILE = "trdizin_subject_hierarchy.csv"
OUTPUT_FILE = "evaluation_dataset.csv"


# ==================================================
# 2. MAKALE METİNLERİNİ OKU
# ==================================================

texts = pd.read_csv(
    TEXT_FILE,
    encoding="utf-8-sig"
)

print("Metin satırı:", len(texts))
print("Farklı makale:", texts["article_id"].nunique())


# ==================================================
# 3. TR DİZİN GERÇEK ETİKETLERİNİ OKU
# ==================================================

subjects = pd.read_csv(
    SUBJECT_FILE,
    encoding="utf-8-sig"
)

print("Konu ilişkisi:", len(subjects))


# ==================================================
# 4. SADECE GERÇEK EN-ALT KONULAR
# ==================================================
# Fen veya Fen > Tıp gibi ara düğümleri
# leaf subject olarak değerlendirmiyoruz.

leaf_subjects = subjects[
    subjects["leaf_subject"].fillna("").str.strip() != ""
].copy()


# ==================================================
# 5. HER MAKALE İÇİN TÜM GERÇEK ETİKETLERİ TOPLA
# ==================================================

def unique_join(series):
    values = []

    for value in series.dropna():
        value = str(value).strip()

        if value and value not in values:
            values.append(value)

    return " || ".join(values)


article_labels = (
    leaf_subjects
    .groupby("article_id")
    .agg({
        "main_field": unique_join,
        "sub_field": unique_join,
        "leaf_subject": unique_join,
        "subject_fullname": unique_join
    })
    .reset_index()
)

article_labels = article_labels.rename(
    columns={
        "main_field": "true_main_fields",
        "sub_field": "true_sub_fields",
        "leaf_subject": "true_leaf_subjects",
        "subject_fullname": "true_subject_paths"
    }
)


# ==================================================
# 6. METİNLER + GERÇEK CEVAPLAR
# ==================================================

evaluation = texts.merge(
    article_labels,
    on="article_id",
    how="left"
)


# ==================================================
# 7. ETİKETİ OLMAYANLARI İŞARETLE
# ==================================================

evaluation["has_ground_truth"] = (
    evaluation["true_leaf_subjects"]
    .fillna("")
    .str.strip()
    .ne("")
)


# ==================================================
# 8. KONTROLLER
# ==================================================

print("\n" + "=" * 70)
print("DEĞERLENDİRME VERİ SETİ")
print("=" * 70)

print("Toplam satır:", len(evaluation))

print(
    "Farklı makale:",
    evaluation["article_id"].nunique()
)

print(
    "Gerçek konu etiketi bulunan satır:",
    evaluation["has_ground_truth"].sum()
)

print(
    "Etiketi bulunmayan satır:",
    (~evaluation["has_ground_truth"]).sum()
)


# ==================================================
# 9. MULTI-LABEL ÖRNEK
# ==================================================

multi_label = evaluation[
    evaluation["true_subject_paths"]
    .fillna("")
    .str.contains(r"\|\|", regex=True)
]

if not multi_label.empty:

    example = multi_label.iloc[0]

    print("\nMULTI-LABEL (ÇOK ETİKETLİ) ÖRNEK")
    print("=" * 70)

    print("Article ID:", example["article_id"])

    print(
        "Gerçek konu yolları:",
        example["true_subject_paths"]
    )


# ==================================================
# 10. KAYDET
# ==================================================

evaluation.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\nDosya oluşturuldu:", OUTPUT_FILE)