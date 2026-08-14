import requests
import psycopg2
import json


# --------------------------------------------------
# 1. TR DİZİN API AYARLARI
# --------------------------------------------------

API_URL = "https://search.trdizin.gov.tr/api/defaultSearch/publication/"

params = {
    "q": "",
    "order": "publicationYear-DESC",
    "page": 1,
    "limit": 100
}


# --------------------------------------------------
# 2. POSTGRESQL BAĞLANTISI
# --------------------------------------------------

conn = psycopg2.connect(
    host="localhost",
    port=5433,
    database="trdizin_ai_db",
    user="postgres",
    password="postgres"
)

cursor = conn.cursor()


try:

    # --------------------------------------------------
    # 3. TR DİZİN'DEN VERİ ÇEK
    # --------------------------------------------------

    response = requests.get(
        API_URL,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    data = response.json()
    articles = data["hits"]["hits"]

    print("API'den gelen yayın:", len(articles))


    # --------------------------------------------------
    # 4. SAYACLAR
    # --------------------------------------------------

    article_count = 0
    text_count = 0
    subject_count = 0


    # --------------------------------------------------
    # 5. MAKALELERİ VERİTABANINA KAYDET
    # --------------------------------------------------

    for hit in articles:

        source = hit["_source"]

        external_id = str(source.get("id"))
        doi = source.get("doi")
        publication_year = source.get("publicationYear")
        publication_type = source.get("publicationType")


        # --------------------------------------------------
        # ARTICLE
        # --------------------------------------------------

        cursor.execute(
            """
            INSERT INTO article (
                external_id,
                doi,
                publication_year,
                publication_type
            )
            VALUES (%s, %s, %s, %s)

            ON CONFLICT (external_id)
            DO UPDATE SET
                doi = EXCLUDED.doi,
                publication_year = EXCLUDED.publication_year,
                publication_type = EXCLUDED.publication_type

            RETURNING id;
            """,
            (
                external_id,
                doi,
                publication_year,
                publication_type
            )
        )

        article_id = cursor.fetchone()[0]
        article_count += 1


        # --------------------------------------------------
        # ARTICLE_TEXT
        # --------------------------------------------------

        abstracts = source.get("abstracts") or []

        for item in abstracts:

            language = item.get("language")
            title = item.get("title")
            abstract = item.get("abstract")
            keywords = item.get("keywords") or []

            # Başlık veya dil yoksa bu kaydı kullanma.
            if not title or not language:
                continue

            # Keywords API'den liste olarak geliyor.
            # PostgreSQL'de TEXT alanında JSON biçiminde saklıyoruz.
            keywords_json = json.dumps(
                keywords,
                ensure_ascii=False
            )

            cursor.execute(
                """
                INSERT INTO article_text (
                    article_id,
                    language,
                    title,
                    abstract,
                    keywords
                )
                VALUES (%s, %s, %s, %s, %s)

                ON CONFLICT (article_id, language)
                DO UPDATE SET
                    title = EXCLUDED.title,
                    abstract = EXCLUDED.abstract,
                    keywords = EXCLUDED.keywords;
                """,
                (
                    article_id,
                    language,
                    title,
                    abstract,
                    keywords_json
                )
            )

            text_count += 1


        # --------------------------------------------------
        # ARTICLE_SUBJECT
        # --------------------------------------------------

        subjects = source.get("subjects") or []

        for subject in subjects:

            # TR Dizin subjects değerlerini object/dict
            # şeklinde döndürüyor.
            if not isinstance(subject, dict):
                continue

            subject_id = subject.get("id")
            subject_name = subject.get("name")
            subject_fullname = subject.get("fullName")
            root_name = subject.get("rootName")
            auto_label = subject.get("autoLabel")

            if not subject_name:
                continue

            cursor.execute(
                """
                INSERT INTO article_subject (
                    article_id,
                    subject_id,
                    subject_name,
                    subject_fullname,
                    root_name,
                    auto_label
                )
                VALUES (%s, %s, %s, %s, %s, %s)

                ON CONFLICT (article_id, subject_id)
                DO UPDATE SET
                    subject_name = EXCLUDED.subject_name,
                    subject_fullname = EXCLUDED.subject_fullname,
                    root_name = EXCLUDED.root_name,
                    auto_label = EXCLUDED.auto_label;
                """,
                (
                    article_id,
                    subject_id,
                    subject_name,
                    subject_fullname,
                    root_name,
                    auto_label
                )
            )

            subject_count += 1


    # --------------------------------------------------
    # 6. DEĞİŞİKLİKLERİ KAYDET
    # --------------------------------------------------

    conn.commit()

    print("\nKAYIT TAMAMLANDI")
    print("=" * 50)

    print("Article işlendi:", article_count)
    print("Article Text işlendi:", text_count)
    print("Subjects işlendi:", subject_count)


# --------------------------------------------------
# 7. HATA OLURSA
# --------------------------------------------------

except Exception as error:

    conn.rollback()

    print("\nHATA OLUŞTU!")
    print(error)


# --------------------------------------------------
# 8. BAĞLANTIYI KAPAT
# --------------------------------------------------

finally:

    cursor.close()
    conn.close()