import pandas as pd

# Test verisini oku
df = pd.read_csv("data/test_articles.csv", encoding="utf-8")

# Embedding'e girecek metni hazırla:
# title + abstract
df["text"] = df["title"] + ". " + df["abstract"]

# Kaç makale olduğunu göster
print("Toplam makale:", len(df))

# İlk 3 makaleyi kontrol et
print(df[["title", "text", "label"]].head(3))