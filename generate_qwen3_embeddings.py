import os
import time
import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer


CSV_PATH = "real_trdizin_texts.csv"

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

BATCH_SIZE = 2

OUTPUT_DIR = "embeddings"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "Qwen3_embeddings.npy"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# --------------------------------------------------
# VERİ
# --------------------------------------------------

df = pd.read_csv(
    CSV_PATH,
    encoding="utf-8-sig"
)

df["embedding_text"] = (
    df["embedding_text"]
    .fillna("")
    .astype(str)
)


print("=" * 100)
print("QWEN3 EMBEDDING ÜRETİMİ")
print("=" * 100)

print(
    "Metin satırı:",
    len(df)
)

print(
    "Benzersiz makale:",
    df["article_id"].nunique()
)


# --------------------------------------------------
# CİHAZ
# --------------------------------------------------

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


# --------------------------------------------------
# MODEL
# --------------------------------------------------

print(
    "\nModel yükleniyor:",
    MODEL_NAME
)


model = SentenceTransformer(
    MODEL_NAME,
    device=device
)


# --------------------------------------------------
# EMBEDDING
# --------------------------------------------------

texts = df[
    "embedding_text"
].tolist()


start = time.perf_counter()


embeddings = model.encode(

    texts,

    batch_size=BATCH_SIZE,

    show_progress_bar=True,

    normalize_embeddings=True,

    convert_to_numpy=True
)


elapsed = (
    time.perf_counter()
    -
    start
)


embeddings = embeddings.astype(
    np.float32
)


# --------------------------------------------------
# KONTROL
# --------------------------------------------------

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


# --------------------------------------------------
# KAYDET
# --------------------------------------------------

np.save(
    OUTPUT_FILE,
    embeddings
)


print(
    "\nDosya oluşturuldu:",
    OUTPUT_FILE
)