import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# 1. AYARLAR
# ============================================================

TEXT_FILE = "real_trdizin_texts.csv"
SUBJECT_FILE = "trdizin_subject_hierarchy.csv"

EMBEDDING_FILES = {
    "E5": "embeddings/E5_embeddings.npy",
    "BGE-M3": "embeddings/BGE-M3_embeddings.npy",
    "Qwen3": "embeddings/Qwen3_embeddings.npy",
}

RANDOM_STATE = 42
TEST_SIZE = 0.20

TOP_K_VALUES = [1, 3, 5]


# ============================================================
# 2. YARDIMCI FONKSİYON
# ============================================================

def normalize_vector(vector):

    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm


# ============================================================
# 3. METİN DOSYASINI OKU
# ============================================================

texts_df = pd.read_csv(
    TEXT_FILE,
    encoding="utf-8-sig"
)

print("Toplam metin:", len(texts_df))

print(
    "Farklı makale:",
    texts_df["article_id"].nunique()
)


# ============================================================
# 4. GERÇEK TR DİZİN KONULARINI OKU
# ============================================================

subjects_df = pd.read_csv(
    SUBJECT_FILE,
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# Sadece gerçek en-alt konuları kullan
# ------------------------------------------------------------

subjects_df["leaf_subject"] = (
    subjects_df["leaf_subject"]
    .fillna("")
    .astype(str)
    .str.strip()
)

leaf_df = subjects_df[
    subjects_df["leaf_subject"] != ""
].copy()


# ------------------------------------------------------------
# Her makalenin gerçek konu yollarını set olarak tut
# ------------------------------------------------------------

article_true_labels = (
    leaf_df
    .groupby("article_id")["subject_fullname"]
    .apply(lambda x: set(x.dropna().astype(str)))
    .to_dict()
)

print(
    "Gerçek konu etiketi bulunan makale:",
    len(article_true_labels)
)


# ============================================================
# 5. EMBEDDING OLAN + ETİKETİ OLAN MAKALELER
# ============================================================

embedding_article_ids = set(
    texts_df["article_id"].unique()
)

label_article_ids = set(
    article_true_labels.keys()
)

usable_article_ids = sorted(
    embedding_article_ids.intersection(
        label_article_ids
    )
)

print(
    "Değerlendirmede kullanılabilir makale:",
    len(usable_article_ids)
)


# ============================================================
# 6. TRAIN / TEST AYIR
# ============================================================
#
# Aynı article_id'nin TUR ve ENG metni
# kesinlikle farklı gruplara düşmüyor.
#
# Train = referans konu merkezlerini oluşturur.
# Test  = model başarısını ölçer.
#

train_article_ids, test_article_ids = train_test_split(
    usable_article_ids,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

train_article_ids = set(train_article_ids)
test_article_ids = set(test_article_ids)

print("\nTRAIN / TEST")
print("=" * 60)

print(
    "Train (eğitim/referans) makale:",
    len(train_article_ids)
)

print(
    "Test makale:",
    len(test_article_ids)
)


# ============================================================
# 7. ARTICLE-LEVEL EMBEDDING OLUŞTUR
# ============================================================
#
# Aynı makalenin TUR + ENG embeddingleri varsa
# ortalamasını alıyoruz.
#
# Böylece bir makale değerlendirmede yalnızca
# bir kez temsil ediliyor.
#

def build_article_embeddings(
    row_embeddings,
    dataframe
):

    article_vectors = {}

    for article_id, group in dataframe.groupby(
        "article_id"
    ):

        indices = group.index.to_numpy()

        vectors = row_embeddings[indices]

        mean_vector = vectors.mean(
            axis=0
        )

        mean_vector = normalize_vector(
            mean_vector
        )

        article_vectors[article_id] = (
            mean_vector.astype(np.float32)
        )

    return article_vectors


# ============================================================
# 8. SUBJECT CENTROID OLUŞTUR
# ============================================================
#
# Centroid (küme/konu merkez vektörü):
#
# Aynı konuya ait TRAIN makalelerinin
# embedding ortalaması.
#

def build_subject_centroids(
    article_vectors
):

    subject_vectors = {}

    for article_id in train_article_ids:

        if article_id not in article_vectors:
            continue

        vector = article_vectors[
            article_id
        ]

        labels = article_true_labels.get(
            article_id,
            set()
        )

        for label in labels:

            if label not in subject_vectors:
                subject_vectors[label] = []

            subject_vectors[label].append(
                vector
            )


    centroids = {}
    support_counts = {}

    for label, vectors in subject_vectors.items():

        vectors = np.vstack(
            vectors
        )

        centroid = vectors.mean(
            axis=0
        )

        centroid = normalize_vector(
            centroid
        )

        centroids[label] = (
            centroid.astype(np.float32)
        )

        support_counts[label] = len(
            vectors
        )

    return centroids, support_counts


# ============================================================
# 9. TOP-K METRİKLER
# ============================================================

def calculate_metrics(
    true_labels,
    predicted_labels,
    k
):

    true_set = set(
        true_labels
    )

    pred_set = set(
        predicted_labels[:k]
    )

    correct = len(
        true_set.intersection(
            pred_set
        )
    )

    # Precision (kesinlik)
    precision = correct / k

    # Recall (duyarlılık / yakalama oranı)
    recall = (
        correct / len(true_set)
        if len(true_set) > 0
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


    # Hit = ilk K tahminden en az biri doğru mu?
    hit = 1 if correct > 0 else 0

    return (
        precision,
        recall,
        f1,
        hit
    )


# ============================================================
# 10. MODEL KARŞILAŞTIRMASI
# ============================================================

all_results = []


for model_name, embedding_file in EMBEDDING_FILES.items():

    print("\n\n")
    print("=" * 80)
    print(
        "MODEL:",
        model_name
    )
    print("=" * 80)


    # --------------------------------------------------------
    # Embeddingleri oku
    # --------------------------------------------------------

    embeddings = np.load(
        embedding_file
    )

    embeddings = embeddings.astype(
        np.float32
    )

    print(
        "Embedding shape:",
        embeddings.shape
    )


    # --------------------------------------------------------
    # Her satırı normalize et
    # --------------------------------------------------------

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1

    embeddings = (
        embeddings / norms
    )


    # --------------------------------------------------------
    # Makale bazlı embedding
    # --------------------------------------------------------

    article_vectors = (
        build_article_embeddings(
            embeddings,
            texts_df
        )
    )

    print(
        "Makale embedding sayısı:",
        len(article_vectors)
    )


    # --------------------------------------------------------
    # TRAIN verisinden subject centroid oluştur
    # --------------------------------------------------------

    centroids, support_counts = (
        build_subject_centroids(
            article_vectors
        )
    )

    print(
        "Train içinde temsil edilen konu:",
        len(centroids)
    )


    # --------------------------------------------------------
    # Centroid matrisi
    # --------------------------------------------------------

    centroid_labels = list(
        centroids.keys()
    )

    centroid_matrix = np.vstack([
        centroids[label]
        for label in centroid_labels
    ])


    # --------------------------------------------------------
    # Test sonuçları
    # --------------------------------------------------------

    metric_values = {
        k: {
            "precision": [],
            "recall": [],
            "f1": [],
            "hit": []
        }
        for k in TOP_K_VALUES
    }

    prediction_rows = []

    evaluated_articles = 0
    skipped_articles = 0


    for article_id in sorted(
        test_article_ids
    ):

        if article_id not in article_vectors:
            continue


        true_labels = article_true_labels.get(
            article_id,
            set()
        )


        # ----------------------------------------------------
        # Train içinde hiç görülmeyen gerçek etiketleri
        # değerlendirmeden çıkar.
        # ----------------------------------------------------

        known_true_labels = {
            label
            for label in true_labels
            if label in centroids
        }


        if not known_true_labels:

            skipped_articles += 1
            continue


        article_vector = article_vectors[
            article_id
        ]


        # ----------------------------------------------------
        # Cosine Similarity
        # (Kosinüs benzerliği)
        #
        # Vektörler normalize olduğu için:
        # dot product = cosine similarity
        # ----------------------------------------------------

        scores = centroid_matrix @ article_vector


        ranked_indices = np.argsort(
            scores
        )[::-1]


        top_indices = ranked_indices[:5]


        predicted_labels = [
            centroid_labels[i]
            for i in top_indices
        ]

        predicted_scores = [
            float(scores[i])
            for i in top_indices
        ]


        # ----------------------------------------------------
        # TOP-K METRİKLER
        # ----------------------------------------------------

        for k in TOP_K_VALUES:

            (
                precision,
                recall,
                f1,
                hit
            ) = calculate_metrics(
                known_true_labels,
                predicted_labels,
                k
            )

            metric_values[k][
                "precision"
            ].append(precision)

            metric_values[k][
                "recall"
            ].append(recall)

            metric_values[k][
                "f1"
            ].append(f1)

            metric_values[k][
                "hit"
            ].append(hit)


        evaluated_articles += 1


        # ----------------------------------------------------
        # Web arayüzünde de kullanabileceğimiz
        # tahminleri kaydet
        # ----------------------------------------------------

        prediction_rows.append({
            "article_id": article_id,

            "true_subjects":
                " || ".join(
                    sorted(
                        known_true_labels
                    )
                ),

            "prediction_1":
                predicted_labels[0],

            "score_1":
                round(
                    predicted_scores[0],
                    4
                ),

            "prediction_2":
                predicted_labels[1]
                if len(predicted_labels) > 1
                else "",

            "score_2":
                round(
                    predicted_scores[1],
                    4
                )
                if len(predicted_scores) > 1
                else "",

            "prediction_3":
                predicted_labels[2]
                if len(predicted_labels) > 2
                else "",

            "score_3":
                round(
                    predicted_scores[2],
                    4
                )
                if len(predicted_scores) > 2
                else "",

            "prediction_4":
                predicted_labels[3]
                if len(predicted_labels) > 3
                else "",

            "score_4":
                round(
                    predicted_scores[3],
                    4
                )
                if len(predicted_scores) > 3
                else "",

            "prediction_5":
                predicted_labels[4]
                if len(predicted_labels) > 4
                else "",

            "score_5":
                round(
                    predicted_scores[4],
                    4
                )
                if len(predicted_scores) > 4
                else "",
        })


    # --------------------------------------------------------
    # SONUÇLARI YAZDIR
    # --------------------------------------------------------

    print(
        "Değerlendirilen test makalesi:",
        evaluated_articles
    )

    print(
        "Train'de etiketi bulunmadığı için "
        "atlanılan:",
        skipped_articles
    )


    for k in TOP_K_VALUES:

        precision = np.mean(
            metric_values[k][
                "precision"
            ]
        )

        recall = np.mean(
            metric_values[k][
                "recall"
            ]
        )

        f1 = np.mean(
            metric_values[k][
                "f1"
            ]
        )

        hit_rate = np.mean(
            metric_values[k][
                "hit"
            ]
        )


        print("\n")
        print(
            f"TOP-{k}"
        )

        print(
            f"Precision (kesinlik): "
            f"{precision:.4f}"
        )

        print(
            f"Recall (duyarlılık): "
            f"{recall:.4f}"
        )

        print(
            f"F1 Score (F1 skoru): "
            f"{f1:.4f}"
        )

        print(
            f"Hit Rate "
            f"(en az bir doğru tahmin): "
            f"{hit_rate:.4f}"
        )


        all_results.append({

            "Model": model_name,

            "Top_K": k,

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
                hit_rate,
                4
            ),

            "Evaluated_Articles":
                evaluated_articles,

            "Available_Subjects":
                len(centroids)
        })


    # --------------------------------------------------------
    # MODEL TAHMİNLERİNİ CSV'YE KAYDET
    # --------------------------------------------------------

    prediction_df = pd.DataFrame(
        prediction_rows
    )

    prediction_file = (
        f"subject_predictions_{model_name}.csv"
    )

    prediction_df.to_csv(
        prediction_file,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "\nTahmin dosyası:",
        prediction_file
    )


# ============================================================
# 11. TÜM MODELLERİ KARŞILAŞTIR
# ============================================================

results_df = pd.DataFrame(
    all_results
)

print("\n\n")
print("=" * 100)

print(
    "GERÇEK TR DİZİN SUBJECT PREDICTION "
    "(KONU TAHMİNİ) KARŞILAŞTIRMASI"
)

print("=" * 100)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 12. CSV KAYDET
# ============================================================

results_df.to_csv(
    "subject_prediction_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nSonuç dosyası oluşturuldu:"
    " subject_prediction_results.csv"
)