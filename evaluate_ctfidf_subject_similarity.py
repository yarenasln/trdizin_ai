import os
import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# DOSYALAR
# ============================================================

SUMMARY_FILE = (
    "results/final_cluster_ctfidf_summary.csv"
)

OUTPUT_FILE = (
    "results/ctfidf_subject_similarity.csv"
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


# ============================================================
# VERİYİ OKU
# ============================================================

df = pd.read_csv(
    SUMMARY_FILE,
    encoding="utf-8-sig"
)


print("=" * 110)
print("c-TF-IDF ↔ TR DİZİN KONU SEMANTIC SIMILARITY")
print("=" * 110)

print(
    "Cluster sayısı:",
    df["cluster_id"].nunique()
)


# ============================================================
# BOŞLARI TEMİZLE
# ============================================================

df["top_subject"] = (
    df["top_subject"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["ctfidf_keywords"] = (
    df["ctfidf_keywords"]
    .fillna("")
    .astype(str)
    .str.strip()
)


df = df[
    (
        df["top_subject"] != ""
    )
    &
    (
        df["ctfidf_keywords"] != ""
    )
].copy()


print(
    "Değerlendirilecek cluster:",
    len(df)
)


# ============================================================
# c-TF-IDF KEYWORD METNİNİ HAZIRLA
# ============================================================

def build_keyword_text(value):

    parts = [
        x.strip()
        for x in str(value).split("||")
        if x.strip()
    ]

    # İlk 10 terim yeterli
    parts = parts[:10]

    return ", ".join(parts)


df["ctfidf_text"] = (
    df["ctfidf_keywords"]
    .apply(
        build_keyword_text
    )
)


# ============================================================
# SADECE LEAF KONU ADINI DA ÇIKAR
# ============================================================

def extract_leaf_subject(full_path):

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
    .apply(
        extract_leaf_subject
    )
)


# ============================================================
# CİHAZ
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    "Kullanılan cihaz:",
    device
)


if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# MODELİ YÜKLE
# ============================================================

print(
    "\nQwen3 yükleniyor..."
)

model = SentenceTransformer(
    MODEL_NAME,
    device=device
)


# ============================================================
# EMBEDDINGLER
# ============================================================
#
# 1. Tam hiyerarşik TR Dizin konusu
#
# Fen > Mühendislik >
# Bilgisayar Bilimleri, Yapay Zeka
#
# 2. Sadece leaf konu
#
# Bilgisayar Bilimleri, Yapay Zeka
#
# 3. c-TF-IDF terimleri
#
# yapay zeka, öğrenme, derin öğrenme...
# ============================================================

full_subject_texts = (
    df["top_subject"]
    .tolist()
)

leaf_subject_texts = (
    df["leaf_subject_name"]
    .tolist()
)

keyword_texts = (
    df["ctfidf_text"]
    .tolist()
)


print(
    "Embedding üretiliyor..."
)


full_subject_embeddings = (
    model.encode(
        full_subject_texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
)


leaf_subject_embeddings = (
    model.encode(
        leaf_subject_texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
)


keyword_embeddings = (
    model.encode(
        keyword_texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
)


# ============================================================
# SATIR BAZLI COSINE SIMILARITY
# ============================================================

full_similarities = np.sum(
    full_subject_embeddings
    *
    keyword_embeddings,
    axis=1
)


leaf_similarities = np.sum(
    leaf_subject_embeddings
    *
    keyword_embeddings,
    axis=1
)


df[
    "full_subject_similarity"
] = full_similarities


df[
    "leaf_subject_similarity"
] = leaf_similarities


# ============================================================
# BASİT UYUM SEVİYESİ
# ============================================================

def similarity_level(value):

    if value >= 0.75:
        return "YUKSEK"

    elif value >= 0.60:
        return "ORTA"

    else:
        return "DUSUK"


df[
    "semantic_alignment_level"
] = (
    df[
        "leaf_subject_similarity"
    ]
    .apply(
        similarity_level
    )
)


# ============================================================
# GENEL İSTATİSTİKLER
# ============================================================

print("\n" + "=" * 110)
print("GENEL SONUÇ")
print("=" * 110)


print(
    "Ortalama full-path similarity:",
    round(
        df[
            "full_subject_similarity"
        ].mean(),
        4
    )
)


print(
    "Medyan full-path similarity:",
    round(
        df[
            "full_subject_similarity"
        ].median(),
        4
    )
)


print(
    "Ortalama leaf similarity:",
    round(
        df[
            "leaf_subject_similarity"
        ].mean(),
        4
    )
)


print(
    "Medyan leaf similarity:",
    round(
        df[
            "leaf_subject_similarity"
        ].median(),
        4
    )
)


# ============================================================
# DAĞILIM
# ============================================================

print("\n" + "=" * 110)
print("SEMANTIC UYUM SEVİYESİ")
print("=" * 110)


print(
    df[
        "semantic_alignment_level"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# PERCENTİLLER
# ============================================================

print("\nLeaf similarity percentilleri:")


for p in [
    10,
    25,
    50,
    75,
    90,
    95
]:

    print(
        f"P{p}:",
        round(
            df[
                "leaf_subject_similarity"
            ]
            .quantile(
                p / 100
            ),
            4
        )
    )


# ============================================================
# EN YÜKSEK UYUMLU CLUSTER'LAR
# ============================================================

print("\n" + "=" * 130)
print("EN YÜKSEK SEMANTIC UYUMLU 15 CLUSTER")
print("=" * 130)


print(
    df
    .sort_values(
        "leaf_subject_similarity",
        ascending=False
    )
    [
        [
            "cluster_id",
            "top_subject",
            "ctfidf_text",
            "leaf_subject_similarity"
        ]
    ]
    .head(15)
    .to_string(
        index=False
    )
)


# ============================================================
# EN DÜŞÜK UYUMLU CLUSTER'LAR
# ============================================================

print("\n" + "=" * 130)
print("EN DÜŞÜK SEMANTIC UYUMLU 15 CLUSTER")
print("=" * 130)


print(
    df
    .sort_values(
        "leaf_subject_similarity",
        ascending=True
    )
    [
        [
            "cluster_id",
            "top_subject",
            "ctfidf_text",
            "leaf_subject_similarity"
        ]
    ]
    .head(15)
    .to_string(
        index=False
    )
)


# ============================================================
# DOSYAYA KAYDET
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


print(
    "\nDosya oluşturuldu:",
    OUTPUT_FILE
)