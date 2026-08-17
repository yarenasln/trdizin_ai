import os
import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer


# ============================================================
# AYARLAR
# ============================================================

SUBJECT_FILE = "trdizin_subject_hierarchy.csv"

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
BATCH_SIZE = 2

OUTPUT_DIR = "embeddings"
OUTPUT_EMBEDDING = os.path.join(
    OUTPUT_DIR,
    "Qwen3_subject_embeddings.npy"
)

OUTPUT_METADATA = (
    "Qwen3_subject_metadata.csv"
)


# ============================================================
# KLASÖR
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# VERİYİ OKU
# ============================================================

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

subjects["subject_fullname"] = (
    subjects["subject_fullname"]
    .fillna("")
    .astype(str)
    .str.strip()
)


leaf_subjects = subjects[
    (subjects["leaf_subject"] != "")
    &
    (subjects["subject_fullname"] != "")
].copy()


# ============================================================
# BENZERSİZ KONU YOLLARI
# ============================================================

subject_metadata = (
    leaf_subjects[
        [
            "subject_fullname",
            "main_field",
            "sub_field",
            "leaf_subject"
        ]
    ]
    .drop_duplicates(
        subset=[
            "subject_fullname"
        ]
    )
    .reset_index(
        drop=True
    )
)


print("=" * 100)
print("TR DİZİN QWEN3 KONU EMBEDDING ÜRETİMİ")
print("=" * 100)

print(
    "Benzersiz en-alt konu:",
    len(subject_metadata)
)


# ============================================================
# EMBEDDING'E GİRECEK METİN
# ============================================================
#
# Sadece leaf_subject yerine tam yolu kullanıyoruz:
#
# Fen > Mühendislik > Bilgisayar Bilimleri, Yapay Zeka
#
# Böylece konu adı bağlamını da koruyoruz.
# ============================================================

subject_texts = (
    subject_metadata[
        "subject_fullname"
    ]
    .astype(str)
    .tolist()
)


# ============================================================
# CİHAZ
# ============================================================

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


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
# MODEL
# ============================================================

print(
    "Model yükleniyor:",
    MODEL_NAME
)


model = SentenceTransformer(
    MODEL_NAME,
    device=device
)


# ============================================================
# EMBEDDING
# ============================================================

print(
    "Konu embeddingleri oluşturuluyor..."
)


subject_embeddings = model.encode(

    subject_texts,

    batch_size=BATCH_SIZE,

    show_progress_bar=True,

    normalize_embeddings=True,

    convert_to_numpy=True
)


subject_embeddings = (
    subject_embeddings
    .astype(
        np.float32
    )
)


# ============================================================
# KONTROL
# ============================================================

print(
    "Embedding shape:",
    subject_embeddings.shape
)

print(
    "Embedding dimension:",
    subject_embeddings.shape[1]
)


# ============================================================
# ID EKLE
# ============================================================

subject_metadata.insert(
    0,
    "subject_embedding_id",
    range(
        len(subject_metadata)
    )
)


# ============================================================
# KAYDET
# ============================================================

np.save(
    OUTPUT_EMBEDDING,
    subject_embeddings
)


subject_metadata.to_csv(
    OUTPUT_METADATA,
    index=False,
    encoding="utf-8-sig"
)


print("\nDosyalar oluşturuldu:")

print(
    OUTPUT_EMBEDDING
)

print(
    OUTPUT_METADATA
)


# ============================================================
# İLK ÖRNEKLER
# ============================================================

print("\n" + "=" * 100)
print("İLK 10 KONU")
print("=" * 100)


print(
    subject_metadata[
        [
            "subject_embedding_id",
            "subject_fullname"
        ]
    ]
    .head(10)
    .to_string(
        index=False
    )
)