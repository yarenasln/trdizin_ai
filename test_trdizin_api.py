import requests

url = "https://search.trdizin.gov.tr/api/defaultSearch/publication/"

params = {
    "q": "",
    "order": "publicationYear-DESC",
    "page": 1,
    "limit": 100
}

# TR Dizin API'ye istek gönder
response = requests.get(url, params=params, timeout=60)

# HTTP hatası varsa burada durdur
response.raise_for_status()

# JSON cevabını Python nesnesine çevir
data = response.json()

# Makaleleri al
articles = data["hits"]["hits"]

# --------------------------------------------------
# 1. VERİ KALİTESİ ANALİZİ
# --------------------------------------------------

total = len(articles)

subjects_filled = 0
subjects_empty = 0

tur_count = 0
eng_count = 0
both_languages = 0

abstract_empty = 0
keywords_empty = 0

usable_embedding_texts = 0

for article in articles:

    source = article["_source"]

    # Subjects kontrolü
    subjects = source.get("subjects")

    if subjects:
        subjects_filled += 1
    else:
        subjects_empty += 1

    # Bir yayının Türkçe / İngilizce metinleri
    abstracts = source.get("abstracts", [])

    languages = set()

    for item in abstracts:

        language = item.get("language")

        title = item.get("title") or ""
        abstract = item.get("abstract") or ""
        keywords = item.get("keywords") or []

        if language:
            languages.add(language)

        if language == "TUR":
            tur_count += 1

        elif language == "ENG":
            eng_count += 1

        if not abstract.strip():
            abstract_empty += 1

        if not keywords:
            keywords_empty += 1

        # Keywords API'den liste halinde geliyor.
        # Listeyi tek bir metne çeviriyoruz.
        keywords_text = " ".join(keywords)

        # EMBEDDING'E GİRECEK METİN:
        #
        # title + abstract + keywords
        #
        # SUBJECTS BURAYA GİRMİYOR!
        embedding_text = (
            title.strip()
            + ". "
            + abstract.strip()
            + ". Keywords: "
            + keywords_text.strip()
        ).strip()

        if title.strip() and embedding_text:
            usable_embedding_texts += 1

    # Yayında hem Türkçe hem İngilizce varsa
    if "TUR" in languages and "ENG" in languages:
        both_languages += 1


print("\nTR DİZİN VERİ KALİTESİ ANALİZİ")
print("=" * 60)

print("Toplam yayın:", total)

print("\nSUBJECTS")
print("Subjects dolu:", subjects_filled)
print("Subjects boş:", subjects_empty)

print("\nDİL")
print("Türkçe metin sayısı:", tur_count)
print("İngilizce metin sayısı:", eng_count)
print("Hem TUR hem ENG olan yayın:", both_languages)

print("\nİÇERİK")
print("Abstract boş:", abstract_empty)
print("Keywords boş:", keywords_empty)

print("\nEMBEDDING")
print("Embedding için kullanılabilir metin:", usable_embedding_texts)


# --------------------------------------------------
# 2. TR DİZİN KONU ALANLARINI ÇIKAR
# --------------------------------------------------

print("\n\nTR DİZİN KONU ALANLARI")
print("=" * 60)

aggregations = data.get("aggregations", {})

subject_facet = aggregations.get("facet-subject", {})

# API'nin döndürdüğü yapıya göre bucket'ları bul
if "buckets" in subject_facet:

    subject_buckets = subject_facet["buckets"]

elif "values" in subject_facet:

    subject_buckets = (
        subject_facet
        .get("values", {})
        .get("buckets", [])
    )

else:

    subject_buckets = []


print("API'nin döndürdüğü toplam konu alanı:", len(subject_buckets))

print()

for i, subject in enumerate(subject_buckets, start=1):

    subject_name = subject.get("key")
    publication_count = subject.get("doc_count")

    print(
        f"{i}. {subject_name} "
        f"-> {publication_count} yayın"
    )