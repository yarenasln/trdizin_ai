import os
import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer


# ============================================================
# DOSYALAR
# ============================================================

INPUT_FILE = "results/final_cluster_ctfidf_summary.csv"

OUTPUT_FILE = (
    "results/contextual_subject_similarity.csv"
)

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


# ============================================================
# VERİYİ OKU
# ============================================================

df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8-sig"
)

print("=" * 110)
print("BAĞLAMLI TR DİZİN KONU TEMSİLİ ANALİZİ")
print("=" * 110)

print(
    "Toplam cluster:",
    df["cluster_id"].nunique()
)


# ============================================================
# GEREKLİ ALANLARI TEMİZLE
# ============================================================

columns = [
    "dominant_main_field",
    "dominant_sub_field",
    "top_subject",
    "ctfidf_keywords"
]

for column in columns:

    df[column] = (
        df[column]
        .fillna("")
        .astype(str)
        .str.strip()
    )


# ============================================================
# LEAF KONUYU ÇIKAR
# ============================================================

def extract_leaf(full_path):

    parts = [
        x.strip()
        for x in str(full_path).split(">")
        if x.strip()
    ]

    if not parts:
        return ""

    return parts[-1]


df["leaf_subject_name"] = (
    df["top_subject"]
    .apply(extract_leaf)
)


# ============================================================
# c-TF-IDF METNİ
# ============================================================

def build_keyword_text(value):

    parts = [
        x.strip()
        for x in str(value).split("||")
        if x.strip()
    ]

    # İlk 10 temsilci terim
    return ", ".join(
        parts[:10]
    )


df["ctfidf_text"] = (
    df["ctfidf_keywords"]
    .apply(build_keyword_text)
)


# ============================================================
# 3 FARKLI KONU TEMSİLİ OLUŞTUR
# ============================================================
#
# 1) Sadece leaf:
#    Alerji
#
# 2) Hiyerarşik:
#    Fen > Tıp > Alerji
#
# 3) Bağlamlı doğal cümle:
#    Bu konu Fen ana alanında, Tıp alt alanında
#    yer alan Alerji çalışmalarını temsil eder.
#
# Böylece hangi temsil biçiminin cluster terimleriyle
# daha iyi semantik eşleştiğini göreceğiz.
# ============================================================


def build_hierarchy_text(row):

    return (
        f"{row['dominant_main_field']} > "
        f"{row['dominant_sub_field']} > "
        f"{row['leaf_subject_name']}"
    )


def build_contextual_text(row):

    main = row[
        "dominant_main_field"
    ]

    sub = row[
        "dominant_sub_field"
    ]

    leaf = row[
        "leaf_subject_name"
    ]

    return (
        f"Bu konu {main} ana alanında, "
        f"{sub} alt alanında yer alan "
        f"{leaf} çalışmalarını temsil eder."
    )


df["hierarchy_text"] = (
    df.apply(
        build_hierarchy_text,
        axis=1
    )
)

df["contextual_text"] = (
    df.apply(
        build_contextual_text,
        axis=1
    )
)


# ============================================================
# BOŞ KAYITLARI ÇIKAR
# ============================================================

df = df[
    (df["leaf_subject_name"] != "")
    &
    (df["ctfidf_text"] != "")
].copy()


print(
    "Değerlendirilecek cluster:",
    len(df)
)


# ============================================================
# ÖRNEK GÖSTER
# ============================================================

print("\n" + "=" * 110)
print("ÖRNEK KONU TEMSİLİ")
print("=" * 110)

example = df.iloc[0]

print(
    "Leaf:",
    example["leaf_subject_name"]
)

print(
    "Hierarchy:",
    example["hierarchy_text"]
)

print(
    "Contextual:",
    example["contextual_text"]
)

print(
    "c-TF-IDF:",
    example["ctfidf_text"]
)


# ============================================================
# DEVICE
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "\nKullanılan cihaz:",
    device
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# MODEL
# ============================================================

print(
    "\nQwen3 yükleniyor..."
)

model = SentenceTransformer(
    MODEL_NAME,
    device=device
)


# ============================================================
# EMBEDDING FONKSİYONU
# ============================================================

def encode(texts):

    return model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )


# ============================================================
# EMBEDDINGLER
# ============================================================

print(
    "\nLeaf embedding üretiliyor..."
)

leaf_embeddings = encode(
    df["leaf_subject_name"].tolist()
)


print(
    "\nHierarchy embedding üretiliyor..."
)

hierarchy_embeddings = encode(
    df["hierarchy_text"].tolist()
)


print(
    "\nContextual embedding üretiliyor..."
)

contextual_embeddings = encode(
    df["contextual_text"].tolist()
)


print(
    "\nc-TF-IDF embedding üretiliyor..."
)

keyword_embeddings = encode(
    df["ctfidf_text"].tolist()
)


# ============================================================
# COSINE SIMILARITY
# ============================================================
#
# Embeddingler normalize edildiği için
# dot product = cosine similarity
# ============================================================

df["leaf_similarity"] = np.sum(
    leaf_embeddings
    *
    keyword_embeddings,
    axis=1
)


df["hierarchy_similarity"] = np.sum(
    hierarchy_embeddings
    *
    keyword_embeddings,
    axis=1
)


df["contextual_similarity"] = np.sum(
    contextual_embeddings
    *
    keyword_embeddings,
    axis=1
)


# ============================================================
# HANGİ TEMSİL DAHA İYİ?
# ============================================================

def best_representation(row):

    scores = {

        "LEAF":
            row["leaf_similarity"],

        "HIERARCHY":
            row["hierarchy_similarity"],

        "CONTEXTUAL":
            row["contextual_similarity"]
    }

    return max(
        scores,
        key=scores.get
    )


df["best_representation"] = (
    df.apply(
        best_representation,
        axis=1
    )
)


# ============================================================
# CONTEXTUAL İYİLEŞME
# ============================================================

df[
    "contextual_improvement_vs_leaf"
] = (

    df["contextual_similarity"]
    -
    df["leaf_similarity"]
)


# ============================================================
# GENEL SONUÇ
# ============================================================

print("\n" + "=" * 110)
print("GENEL KARŞILAŞTIRMA")
print("=" * 110)


for name, column in [

    (
        "LEAF",
        "leaf_similarity"
    ),

    (
        "HIERARCHY",
        "hierarchy_similarity"
    ),

    (
        "CONTEXTUAL",
        "contextual_similarity"
    )
]:

    print(
        f"\n{name}"
    )

    print(
        "Ortalama:",
        round(
            df[column].mean(),
            4
        )
    )

    print(
        "Medyan:",
        round(
            df[column].median(),
            4
        )
    )

    print(
        "Minimum:",
        round(
            df[column].min(),
            4
        )
    )

    print(
        "Maksimum:",
        round(
            df[column].max(),
            4
        )
    )


# ============================================================
# CONTEXTUAL GERÇEKTEN İYİLEŞTİRDİ Mİ?
# ============================================================

improved = (
    df[
        "contextual_improvement_vs_leaf"
    ]
    > 0
).sum()


worsened = (
    df[
        "contextual_improvement_vs_leaf"
    ]
    < 0
).sum()


same = (
    df[
        "contextual_improvement_vs_leaf"
    ]
    == 0
).sum()


print("\n" + "=" * 110)
print("CONTEXTUAL vs LEAF")
print("=" * 110)

print(
    "Contextual daha iyi:",
    improved
)

print(
    "Leaf daha iyi:",
    worsened
)

print(
    "Aynı:",
    same
)

print(
    "Ortalama değişim:",
    round(
        df[
            "contextual_improvement_vs_leaf"
        ].mean(),
        4
    )
)


# ============================================================
# EN İYİ TEMSİL DAĞILIMI
# ============================================================

print("\n" + "=" * 110)
print("HANGİ KONU TEMSİLİ DAHA İYİ?")
print("=" * 110)

print(
    df[
        "best_representation"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# EN ÇOK İYİLEŞENLER
# ============================================================

print("\n" + "=" * 140)
print("CONTEXTUAL İLE EN ÇOK İYİLEŞEN 15 CLUSTER")
print("=" * 140)

print(
    df.sort_values(
        "contextual_improvement_vs_leaf",
        ascending=False
    )
    [
        [
            "cluster_id",
            "leaf_subject_name",
            "leaf_similarity",
            "contextual_similarity",
            "contextual_improvement_vs_leaf"
        ]
    ]
    .head(15)
    .to_string(
        index=False
    )
)


# ============================================================
# EN ÇOK KÖTÜLEŞENLER
# ============================================================

print("\n" + "=" * 140)
print("CONTEXTUAL İLE EN ÇOK DÜŞEN 15 CLUSTER")
print("=" * 140)

print(
    df.sort_values(
        "contextual_improvement_vs_leaf",
        ascending=True
    )
    [
        [
            "cluster_id",
            "leaf_subject_name",
            "leaf_similarity",
            "contextual_similarity",
            "contextual_improvement_vs_leaf"
        ]
    ]
    .head(15)
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

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


print("\n" + "=" * 110)
print("TAMAMLANDI")
print("=" * 110)

print(
    "Dosya oluşturuldu:",
    OUTPUT_FILE
)