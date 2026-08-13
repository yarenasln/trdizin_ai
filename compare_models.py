import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score
from sklearn.preprocessing import LabelEncoder


# 1. Veriyi oku
df = pd.read_csv("data/test_articles.csv", encoding="utf-8")

# 2. Embedding'e girecek metni hazırla
# label burada KULLANILMIYOR
df["text"] = df["title"] + ". " + df["abstract"]


# 3. Gerçek etiketleri sadece değerlendirme için sayıya çevir
label_encoder = LabelEncoder()
true_labels = label_encoder.fit_transform(df["label"])


# 4. Karşılaştıracağımız embedding modelleri
models = {
    "E5": "intfloat/multilingual-e5-large-instruct",
    "BGE-M3": "BAAI/bge-m3",
    "Qwen3": "Qwen/Qwen3-Embedding-0.6B",
}


results = []


# 5. Her modeli aynı veri üzerinde test et
for model_short_name, model_name in models.items():

    print("\n" + "=" * 60)
    print(f"Model çalıştırılıyor: {model_short_name}")
    print("=" * 60)

    # Modeli yükle
    model = SentenceTransformer(model_name)

    # 18 makaleyi embedding'e çevir
    embeddings = model.encode(
        df["text"].tolist(),
        show_progress_bar=True
    )

    print("Embedding boyutu:", embeddings.shape)

    # 3 gerçek konu olduğu için şimdilik K = 3
    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    predicted_clusters = kmeans.fit_predict(embeddings)

    # Clustering sonuçlarını gerçek label'larla karşılaştır
    ari = adjusted_rand_score(true_labels, predicted_clusters)

    nmi = normalized_mutual_info_score(
        true_labels,
        predicted_clusters
    )

    v_measure = v_measure_score(
        true_labels,
        predicted_clusters
    )

    results.append({
        "Model": model_short_name,
        "ARI": ari,
        "NMI": nmi,
        "V-Measure": v_measure
    })

    # Hangi makalenin hangi kümeye düştüğünü göster
    temp_df = df[["title", "label"]].copy()
    temp_df["cluster"] = predicted_clusters

    print("\nKüme sonuçları:")
    print(temp_df.to_string(index=False))


# 6. Sonuçları tablo halinde göster
results_df = pd.DataFrame(results)

print("\n\nMODEL KARŞILAŞTIRMA SONUÇLARI")
print("=" * 60)

print(
    results_df
    .sort_values("V-Measure", ascending=False)
    .to_string(index=False)
)