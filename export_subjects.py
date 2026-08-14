import psycopg2
import csv

conn = psycopg2.connect(
    host="localhost",
    port=5433,
    database="trdizin_ai_db",
    user="postgres",
    password="postgres"
)

cursor = conn.cursor()

cursor.execute("""
    SELECT DISTINCT
        subject_id,
        subject_name,
        subject_fullname
    FROM article_subject
    ORDER BY subject_id;
""")

rows = cursor.fetchall()

with open(
    "bizim_subjectler_utf8.csv",
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "subject_id",
        "subject_name",
        "subject_fullname"
    ])

    writer.writerows(rows)

cursor.close()
conn.close()

print("Toplam subject:", len(rows))
print("Dosya oluşturuldu: bizim_subjectler_utf8.csv")