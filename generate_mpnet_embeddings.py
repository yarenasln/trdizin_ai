import os
import time
import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer


# ============================================================
# AYARLAR
# ============================================================

TEXT_FILE = "real_trdizin_texts.csv"

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-mpnet-base-v2"
)

OUTPUT_DIR = "embeddings"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "MPNet_multilingual_embeddings.npy"
)


# ============================================================
# VERİYİ OKU
# ============================================================

df = pd.read_csv(
    TEXT_FILE,
    encoding="utf-8-sig"
)


print("=" * 100)
print("MULTILINGUAL MPNet EMBEDDING")
print("=" * 100)

print(
    "Metin satırı:",
    len(df)
)


# ============================================================
# EMBEDDING TEXT KONTROL
# ============================================================

if "embedding_text" not in df.columns:

    raise ValueError(
        "embedding_text kolonu bulunamadı."
    )


texts = (
    df["embedding_text"]
    .fillna("")
    .astype(str)
    .tolist()
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
    "\nModel yükleniyor:"
)

print(
    MODEL_NAME
)


model = SentenceTransformer(
    MODEL_NAME,
    device=device
)


# ============================================================
# EMBEDDING
# ============================================================

start = time.time()


embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True
)


elapsed = (
    time.time()
    -
    start
)


# ============================================================
# KAYDET
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


np.save(
    OUTPUT_FILE,
    embeddings.astype(
        np.float32
    )
)


# ============================================================
# SONUÇ
# ============================================================

print("\n" + "=" * 100)
print("SONUÇ")
print("=" * 100)

print(
    "Embedding shape:",
    embeddings.shape
)

print(
    "Embedding dimension:",
    embeddings.shape[1]
)

print(
    "Süre:",
    round(
        elapsed,
        2
    ),
    "saniye"
)

print(
    "\nDosya oluşturuldu:",
    OUTPUT_FILE
)