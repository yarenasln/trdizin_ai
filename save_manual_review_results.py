import pandas as pd


# ============================================================
# DOSYALAR
# ============================================================

INPUT_FILE = "manual_review_10_articles.csv"
OUTPUT_FILE = "manual_review_results.csv"


# ============================================================
# VERİYİ OKU
# ============================================================

df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8-sig"
)


# ============================================================
# MANUEL DEĞERLENDİRME SONUÇLARI
# ============================================================
#
# Sınıflar:
#
# TR_DIZIN_DAHA_UYGUN
# SISTEM_DE_UYGUN
# IKISI_DE_UYGUN
# SISTEM_YANLIS
# SISTEM_UYGUN_AMA_GENEL
#
# Buradaki yorumlar başlık + abstract içeriğine
# bakılarak oluşturuldu.
# ============================================================

manual_results = {

    57: {
        "manual_label":
            "SISTEM_DE_UYGUN",

        "manual_comment":
            (
                "Makale rezervuar jeolojisi, sedimantasyon, "
                "diyajenez ve gaz sahası üzerine. "
                "TR Dizin Mühendislik, Jeoloji ve Petrol etiketleri "
                "daha spesifik; sistemin Jeoloji tahmini de içerikle "
                "yüksek ölçüde uyumlu. Paleontoloji tahmini zayıf."
            )
    },

    58: {
        "manual_label":
            "TR_DIZIN_DAHA_UYGUN",

        "manual_comment":
            (
                "Çalışma Cu, Zn ve Ni içeren koordinasyon bileşikleri, "
                "kristal yapı ve spektroskopik karakterizasyon üzerine. "
                "TR Dizin'in Kimya, İnorganik ve Nükleer etiketi "
                "sistemin Organik/Uygulamalı Kimya tahminlerinden "
                "daha uygun."
            )
    },

    83: {
        "manual_label":
            "SISTEM_UYGUN_AMA_GENEL",

        "manual_comment":
            (
                "Makale nanokompozit katalizör kullanarak azo boya "
                "degradasyonu ve atık su arıtımı üzerine. "
                "Kimya Mühendisliği tahmini içerikle ilişkili ancak "
                "TR Dizin'in Nanobilim, Çevre Mühendisliği ve "
                "Uygulamalı Kimya etiketleri daha ayrıntılı."
            )
    },

    89: {
        "manual_label":
            "TR_DIZIN_DAHA_UYGUN",

        "manual_comment":
            (
                "Makale Au(I) kompleksleri, fosfin ligandları, "
                "koordinasyon yapıları ve kristal yapı üzerine. "
                "TR Dizin'in Kimya, İnorganik ve Nükleer etiketi "
                "Organik Kimya tahmininden daha uygun."
            )
    },

    90: {
        "manual_label":
            "IKISI_DE_UYGUN",

        "manual_comment":
            (
                "Makale metal ftalosiyaninlerin enzim inhibisyonu, "
                "antibakteriyel ve antikanser etkilerini inceliyor. "
                "TR Dizin'in Tıbbi Kimya etiketi uygun; sistemin "
                "Toksikoloji ve Onkoloji ek tahminleri de abstract "
                "içeriğinde doğrudan karşılık buluyor."
            )
    },

    91: {
        "manual_label":
            "TR_DIZIN_DAHA_UYGUN",

        "manual_comment":
            (
                "Çalışma Ni koordinasyon bileşikleri, kristal yapı, "
                "spektroskopik analiz ve teorik hesaplamalar üzerine. "
                "TR Dizin'in Kimya, İnorganik ve Nükleer etiketi "
                "daha uygun."
            )
    },

    93: {
        "manual_label":
            "SISTEM_YANLIS",

        "manual_comment":
            (
                "Makale vanadyumun çevresel yayılımı, toksisitesi, "
                "ekolojik etkileri ve çevresel kirletici rolü üzerine. "
                "TR Dizin'in Toksikoloji ve Çevre Bilimleri etiketleri "
                "uygun. Sistemin Malzeme Bilimleri, Tekstil tahmini "
                "içerikle uyumlu değil."
            )
    },

    118: {
        "manual_label":
            "SISTEM_UYGUN_AMA_GENEL",

        "manual_comment":
            (
                "Makale bir bitkide fotosistem II, fotosentez, "
                "elektron taşınımı ve rehidrasyon süreci üzerine. "
                "Sistemin Biyoloji tahmini doğru yönde ancak "
                "TR Dizin'in Bitki Bilimleri etiketi daha spesifik."
            )
    },

    120: {
        "manual_label":
            "TR_DIZIN_DAHA_UYGUN",

        "manual_comment":
            (
                "Makale Antik Çağ'da sosyal politika, sosyal düzenleme, "
                "refah ve işgücü politikalarının tarihsel gelişimi üzerine. "
                "Sistemin Felsefe ve Kamu Yönetimi tahminleri ilişkili "
                "olsa da TR Dizin'in Sosyal Çalışma ve Tarih etiketleri "
                "çalışmanın odağını daha iyi temsil ediyor."
            )
    },

    327: {
        "manual_label":
            "SISTEM_YANLIS",

        "manual_comment":
            (
                "Makalenin temel konusu sekonder adrenal yetmezlik "
                "olgusu ve ateşin nadir endokrin nedeni. "
                "TR Dizin'in Endokrinoloji, Enfeksiyon Hastalıkları "
                "ve Genel Dahili Tıp etiketleri uygun. "
                "Alerji ve Pediatri tahminleri içerikle uyumlu değil."
            )
    }
}


# ============================================================
# MAKALE BAZLI ÖZET TABLO
# ============================================================

article_rows = []


for article_id, group in df.groupby(
    "article_id"
):

    first = group.iloc[0]

    predicted_subjects = (
        group[
            "predicted_subject"
        ]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )


    predicted_subjects = [
        x
        for x in predicted_subjects
        if x
    ]


    result = manual_results.get(
        int(article_id),
        {
            "manual_label":
                "DEGERLENDIRILMEDI",

            "manual_comment":
                ""
        }
    )


    article_rows.append(
        {
            "article_id":
                article_id,

            "title":
                first["title"],

            "trdizin_subjects":
                first["trdizin_subjects"],

            "system_predictions":
                " || ".join(
                    predicted_subjects
                ),

            "manual_label":
                result[
                    "manual_label"
                ],

            "manual_comment":
                result[
                    "manual_comment"
                ]
        }
    )


result_df = pd.DataFrame(
    article_rows
)


# ============================================================
# KAYDET
# ============================================================

result_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# ÖZET
# ============================================================

print("=" * 110)
print("10 MAKALELİK MANUEL İNCELEME SONUÇLARI")
print("=" * 110)


print(
    result_df[
        "manual_label"
    ]
    .value_counts()
    .to_string()
)


print(
    "\nToplam makale:",
    len(result_df)
)


print("\n" + "=" * 110)
print("DETAYLI SONUÇ")
print("=" * 110)


print(
    result_df[
        [
            "article_id",
            "manual_label",
            "system_predictions",
            "trdizin_subjects"
        ]
    ]
    .to_string(
        index=False
    )
)


print(
    "\nDosya oluşturuldu:",
    OUTPUT_FILE
)