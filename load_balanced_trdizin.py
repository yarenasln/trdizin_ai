import requests
import psycopg2
import json
import time


API_URL = "https://search.trdizin.gov.tr/api/defaultSearch/publication/"

PER_SUBJECT_LIMIT = 5


# --------------------------------------------------
# POSTGRESQL BAĞLANTISI
# --------------------------------------------------

conn = psycopg2.connect(
    host="localhost",
    port=5433,
    database="trdizin_ai_db",
    user="postgres",
    password="postgres"
)

cursor = conn.cursor()


def save_article(source):

    external_id = str(source.get("id"))
    doi = source.get("doi")
    publication_year = source.get("publicationYear")
    publication_type = source.get("publicationType")

    # ARTICLE
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


    # ARTICLE_TEXT
    abstracts = source.get("abstracts") or []

    for item in abstracts:

        language = item.get("language")
        title = item.get("title")
        abstract = item.get("abstract")
        keywords = item.get("keywords") or []

        if not title or not language:
            continue

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


    # ARTICLE_SUBJECT
    subjects = source.get("subjects") or []

    for subject in subjects:

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


try:

    # --------------------------------------------------
    # 1. SUBJECT LİSTESİNİ AL
    # --------------------------------------------------

    base_params = {
        "q": "",
        "order": "publicationYear-DESC",
        "page": 1,
        "limit": 1
    }

    response = requests.get(
        API_URL,
        params=base_params,
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    subject_facet = data["aggregations"]["facet-subject"]

    if "buckets" in subject_facet:
        subject_buckets = subject_facet["buckets"]

    elif "values" in subject_facet:
        subject_buckets = subject_facet["values"]["buckets"]

    else:
        subject_buckets = []


    print("Toplam subject:", len(subject_buckets))
    print("=" * 60)


    # --------------------------------------------------
    # 2. HER SUBJECT'TEN VERİ ÇEK
    # --------------------------------------------------

    total_requested = 0

    for index, subject in enumerate(subject_buckets, start=1):

        subject_fullname = subject["key"]

        print(
            f"\n[{index}/{len(subject_buckets)}] "
            f"{subject_fullname}"
        )

        params = {
            "q": "",
            "order": "publicationYear-DESC",
            "page": 1,
            "limit": PER_SUBJECT_LIMIT,
            "facet-subject": subject_fullname
        }

        response = requests.get(
            API_URL,
            params=params,
            timeout=60
        )

        response.raise_for_status()

        subject_data = response.json()

        hits = subject_data["hits"]["hits"]

        print("Gelen yayın:", len(hits))

        for hit in hits:

            source = hit["_source"]

            save_article(source)

            total_requested += 1

        # Her subject sonrası commit
        conn.commit()

        # API'ye aşırı hızlı yüklenmemek için
        time.sleep(0.2)


    print("\nDENGELİ VERİ ÇEKME TAMAMLANDI")
    print("=" * 60)

    print("İşlenen subject:", len(subject_buckets))
    print("İşlenen subject-yayın eşleşmesi:", total_requested)


except Exception as error:

    conn.rollback()

    print("\nHATA OLUŞTU!")
    print(error)


finally:

    cursor.close()
    conn.close()