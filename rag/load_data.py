import json
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FILES = [
    "fiches_etablissements_rag.json",
    "fiches_programmes_rag.json",
    "orientation_rag.json",
    "referentiel_programme_rag.json",
]


def load_json(filename):
    path = DATA_DIR / filename

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


for filename in FILES:
    data = load_json(filename)

    print("=" * 60)
    print(f"Fichier : {filename}")
    print(f"Type Python : {type(data).__name__}")
    print(f"Nombre d'éléments : {len(data)}")

    if isinstance(data, list) and len(data) > 0:
        print("Clés du premier élément :", data[0].keys())