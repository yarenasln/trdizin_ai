import os
import gc
import time

import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. AYARLAR
# --------------------------------------------------

CSV_PATH = "real_trdizin_texts.csv"

MODELS = {
    "E5": "intfloat/multilingual-e5-large-instruct",
    "BGE-M3": "BAAI/bge-m3",
    "Qwen3": "Qwen/Qwen3-Embedding-0.6B",
}

BATCH_SIZE = 16

# Embeddingleri burada saklayacağız.
os.makedirs("embeddings", exist_ok=True)


# --------------------------------------------------
# 2. VERİYİ OKU
# --------------------------------------------------

df = pd.read_csv(
    CSV_PATH,
    encoding="utf-8-sig"
)

df["embedding_text"] = df["embedding_text"].fillna("")

print("Toplam metin:", len(df))


# --------------------------------------------------
# 3. TUR - ENG EŞLEŞEN MAKALELERİ BUL
# --------------------------------------------------

language_table = (
    df.groupby("article_id")["language"]
    .apply(set)
)

paired_article_ids = [
    article_id
    for article_id, languages in language_table.items()
    if "TUR" in languages and "ENG" in languages
]

print(
    "Hem TUR hem ENG metni bulunan makale:",
    len(paired_article_ids)
)


# --------------------------------------------------
# 4. CİHAZ
# --------------------------------------------------

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("Kullanılan cihaz:", device)


# --------------------------------------------------
# 5. SONUÇLAR
# --------------------------------------------------

results = []


# --------------------------------------------------
# 6. HER MODELİ AYNI VERİDE ÇALIŞTIR
# --------------------------------------------------

for short_name, model_name in MODELS.items():

    print("\n")
    print("=" * 70)
    print("MODEL:", short_name)
    print("Hugging Face:", model_name)
    print("=" * 70)

    # -------------------------------
    # Modeli yükle
    # -------------------------------

    load_start = time.perf_counter()

    model = SentenceTransformer(
        model_name,
        device=device
    )

    load_time = time.perf_counter() - load_start

    print(
        f"Model yükleme süresi: "
        f"{load_time:.2f} saniye"
    )


    # -------------------------------
    # Embedding üret
    # -------------------------------

    texts = df["embedding_text"].tolist()

    encode_start = time.perf_counter()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,

        # Cosine similarity için faydalı.
        normalize_embeddings=True,

        # NumPy array olarak dönsün.
        convert_to_numpy=True
    )

    encode_time = time.perf_counter() - encode_start


    # -------------------------------
    # Boyut bilgisi
    # -------------------------------

    embedding_dimension = embeddings.shape[1]

    print(
        "Embedding şekli:",
        embeddings.shape
    )

    print(
        f"Embedding üretme süresi: "
        f"{encode_time:.2f} saniye"
    )


    # -------------------------------
    # TUR - ENG BENZERLİĞİ
    # -------------------------------

    similarities = []

    for article_id in paired_article_ids:

        rows = df[df["article_id"] == article_id]

        tur_rows = rows[
            rows["language"] == "TUR"
        ]

        eng_rows = rows[
            rows["language"] == "ENG"
        ]

        if tur_rows.empty or eng_rows.empty:
            continue

        tur_index = tur_rows.index[0]
        eng_index = eng_rows.index[0]

        tur_vector = embeddings[tur_index]
        eng_vector = embeddings[eng_index]

        # Vektörleri normalize ettiğimiz için
        # dot product = cosine similarity.
        similarity = np.dot(
            tur_vector,
            eng_vector
        )

        similarities.append(float(similarity))


    if similarities:

        average_similarity = np.mean(similarities)
        median_similarity = np.median(similarities)
        minimum_similarity = np.min(similarities)
        maximum_similarity = np.max(similarities)

    else:

        average_similarity = 0
        median_similarity = 0
        minimum_similarity = 0
        maximum_similarity = 0


    print("\nTUR ↔ ENG BENZERLİK")

    print(
        "Karşılaştırılan çift:",
        len(similarities)
    )

    print(
        f"Ortalama cosine similarity: "
        f"{average_similarity:.4f}"
    )

    print(
        f"Medyan cosine similarity: "
        f"{median_similarity:.4f}"
    )

    print(
        f"En düşük: "
        f"{minimum_similarity:.4f}"
    )

    print(
        f"En yüksek: "
        f"{maximum_similarity:.4f}"
    )


    # -------------------------------
    # EMBEDDINGLERİ KAYDET
    # -------------------------------

    embedding_file = (
        f"embeddings/{short_name}_embeddings.npy"
    )

    np.save(
        embedding_file,
        embeddings
    )

    print(
        "\nEmbedding dosyası:",
        embedding_file
    )


    # -------------------------------
    # SONUÇ TABLOSUNA EKLE
    # -------------------------------

    results.append({
        "Model": short_name,
        "Dimension": embedding_dimension,
        "Load_Time_sec": round(load_time, 2),
        "Encode_Time_sec": round(encode_time, 2),
        "TR_EN_Pairs": len(similarities),
        "TR_EN_Mean": round(
            average_similarity,
            4
        ),
        "TR_EN_Median": round(
            median_similarity,
            4
        ),
        "TR_EN_Min": round(
            minimum_similarity,
            4
        ),
        "TR_EN_Max": round(
            maximum_similarity,
            4
        ),
    })


    # -------------------------------
    # MODELİ RAM / GPU'DAN TEMİZLE
    # -------------------------------

    del embeddings
    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------
# 7. KARŞILAŞTIRMA TABLOSU
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n\n")
print("=" * 90)
print("GERÇEK TR DİZİN EMBEDDING MODEL KARŞILAŞTIRMASI")
print("=" * 90)

print(
    results_df.to_string(
        index=False
    )
)


# --------------------------------------------------
# 8. SONUÇLARI CSV'YE KAYDET
# --------------------------------------------------

results_df.to_csv(
    "embedding_model_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nSonuç dosyası oluşturuldu:"
    " embedding_model_results.csv"
)