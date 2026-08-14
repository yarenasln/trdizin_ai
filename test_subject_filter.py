import requests

API_URL = "https://search.trdizin.gov.tr/api/defaultSearch/publication/"

params = {
    "q": "",
    "order": "publicationYear-DESC",
    "page": 1,
    "limit": 20,
    "facet-subject": "Fen > Ziraat > Bitki Bilimleri"
}

response = requests.get(
    API_URL,
    params=params,
    timeout=60
)

response.raise_for_status()

print("İstek URL:")
print(response.url)

data = response.json()
articles = data["hits"]["hits"]

print("\nGelen yayın sayısı:", len(articles))
print("=" * 60)

for hit in articles:
    source = hit["_source"]

    print("\nID:", source.get("id"))
    print("Subjects:", source.get("subjects"))

    abstracts = source.get("abstracts") or []

    if abstracts:
        print("Başlık:", abstracts[0].get("title"))

    print("-" * 60)