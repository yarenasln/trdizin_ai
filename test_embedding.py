import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Veriyi oku
df = pd.read_csv("data/test_articles.csv", encoding="utf-8")

# Embedding'e sadece title + abstract girecek
df["text"] = df["title"] + ". " + df["abstract"]

# Embedding modelini yükle
model_name = "intfloat/multilingual-e5-large-instruct"
model = SentenceTransformer(model_name)

# Üç makale seç
text_1 = df.loc[0, "text"]   # Akciğer - Türkçe
text_2 = df.loc[3, "text"]   # Akciğer - İngilizce
text_3 = df.loc[13, "text"]  # Faiz - Türkçe

# Metinleri vektörleştir
embeddings = model.encode([text_1, text_2, text_3])

# Benzerlikleri hesapla
similarity_same = cosine_similarity(
    [embeddings[0]],
    [embeddings[1]]
)[0][0]

similarity_different = cosine_similarity(
    [embeddings[0]],
    [embeddings[2]]
)[0][0]

print("Akciğer TR <-> Akciğer EN:")
print(similarity_same)

print("\nAkciğer TR <-> Faiz TR:")
print(similarity_different)