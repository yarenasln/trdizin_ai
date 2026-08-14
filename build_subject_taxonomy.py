import pandas as pd


# --------------------------------------------------
# 1. SUBJECT LİSTESİNİ OKU
# --------------------------------------------------

INPUT_FILE = "bizim_subjectler_utf8.csv"
OUTPUT_FILE = "subject_taxonomy.csv"

df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8-sig"
)

print("Toplam subject kaydı:", len(df))


# --------------------------------------------------
# 2. HİYERARŞİYİ PARÇALA
# --------------------------------------------------

def parse_path(fullname):

    if pd.isna(fullname):
        return "", "", "", 0

    parts = [
        x.strip()
        for x in str(fullname).split(">")
        if x.strip()
    ]

    level_1 = parts[0] if len(parts) >= 1 else ""
    level_2 = parts[1] if len(parts) >= 2 else ""
    level_3 = parts[2] if len(parts) >= 3 else ""

    return (
        level_1,
        level_2,
        level_3,
        len(parts)
    )


parsed = df["subject_fullname"].apply(parse_path)

df[
    [
        "main_field",
        "sub_field",
        "leaf_subject",
        "depth"
    ]
] = pd.DataFrame(
    parsed.tolist(),
    index=df.index
)


# --------------------------------------------------
# 3. KAYIT TİPİNİ BELİRLE
# --------------------------------------------------

def node_type(depth):

    if depth == 1:
        return "MAIN_FIELD"

    elif depth == 2:
        return "SUB_FIELD"

    elif depth >= 3:
        return "LEAF_SUBJECT"

    return "UNKNOWN"


df["node_type"] = df["depth"].apply(node_type)


# --------------------------------------------------
# 4. MULTI-LABEL İÇİN TAM YOLU KORU
# --------------------------------------------------

df["taxonomy_path"] = df["subject_fullname"]


# --------------------------------------------------
# 5. KONTROLLER
# --------------------------------------------------

print("\nKAYIT TİPLERİ")
print("=" * 60)

print(
    df["node_type"]
    .value_counts()
)


print("\nANA ALANLAR")
print("=" * 60)

print(
    df["main_field"]
    .dropna()
    .drop_duplicates()
    .to_string(index=False)
)


print("\nALT ALANLAR")
print("=" * 60)

sub_fields = (
    df.loc[
        df["sub_field"] != "",
        ["main_field", "sub_field"]
    ]
    .drop_duplicates()
    .sort_values(
        ["main_field", "sub_field"]
    )
)

print(
    sub_fields.to_string(index=False)
)


print("\nEN ALT KONU SAYISI")
print("=" * 60)

leaf_count = (
    df.loc[
        df["node_type"] == "LEAF_SUBJECT",
        "subject_id"
    ]
    .nunique()
)

print(leaf_count)


# --------------------------------------------------
# 6. TR DİZİN GÜNCEL MULTİDİSİPLİNER DALLARI
# --------------------------------------------------
# Bunları article_subject'e eklemiyoruz.
# Bunlar güncel taxonomy bilgisidir.

multidisciplinary = pd.DataFrame([
    {
        "subject_id": pd.NA,
        "subject_name": "Multidisipliner – Fen",
        "subject_fullname": "Fen > Multidisipliner – Fen",
        "main_field": "Fen",
        "sub_field": "Multidisipliner – Fen",
        "leaf_subject": "",
        "depth": 2,
        "node_type": "SUB_FIELD",
        "taxonomy_path": "Fen > Multidisipliner – Fen",
        "source": "TRDIZIN_CURRENT_TAXONOMY"
    },
    {
        "subject_id": pd.NA,
        "subject_name": "Multidisipliner – Sosyal",
        "subject_fullname": "Sosyal > Multidisipliner – Sosyal",
        "main_field": "Sosyal",
        "sub_field": "Multidisipliner – Sosyal",
        "leaf_subject": "",
        "depth": 2,
        "node_type": "SUB_FIELD",
        "taxonomy_path": "Sosyal > Multidisipliner – Sosyal",
        "source": "TRDIZIN_CURRENT_TAXONOMY"
    }
])


# --------------------------------------------------
# 7. KAYNAĞI İŞARETLE
# --------------------------------------------------

df["source"] = "ARTICLE_SUBJECT_DATA"

taxonomy_df = pd.concat(
    [
        df,
        multidisciplinary
    ],
    ignore_index=True
)


# --------------------------------------------------
# 8. CSV KAYDET
# --------------------------------------------------

taxonomy_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


print("\n" + "=" * 60)
print("TAXONOMY HAZIR")
print("=" * 60)

print("Dosya:", OUTPUT_FILE)
print("Toplam taxonomy kaydı:", len(taxonomy_df))

print(
    "\nNOT:"
    "\nARTICLE_SUBJECT_DATA = makalelerden gelen kayıt"
    "\nTRDIZIN_CURRENT_TAXONOMY = güncel TR Dizin ağacından gelen kayıt"
)