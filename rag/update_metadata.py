from pathlib import Path

import chromadb

from normalize_data import load_all_documents


BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "orientation_guinee"


# Charger les documents normalisés
documents = load_all_documents()

print(f"Documents chargés : {len(documents)}")


# Connexion à la base existante
client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print(
    f"Documents actuellement dans ChromaDB : "
    f"{collection.count()}"
)


# Mise à jour des métadonnées uniquement
ids = [document["id"] for document in documents]
metadatas = [document["metadata"] for document in documents]

print("\nMise à jour des métadonnées...")

collection.update(
    ids=ids,
    metadatas=metadatas
)

print("Mise à jour terminée.")

print(
    f"Documents dans ChromaDB : "
    f"{collection.count()}"
)