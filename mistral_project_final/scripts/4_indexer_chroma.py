"""
INDEXATION — Construit la base vectorielle Chroma à partir du corpus final.
"""
import json
import chromadb
from sentence_transformers import SentenceTransformer

CORPUS_PATH = "../data/processed/corpus_final.json"
CHROMA_PATH = "../data/chroma_db"
NOM_COLLECTION = "orientation_guinee_v2"
NOM_MODELE_EMBEDDING = "BAAI/bge-m3"


def main():
    print(f"Chargement du modèle {NOM_MODELE_EMBEDDING} (téléchargement au premier lancement)...")
    modele = SentenceTransformer(NOM_MODELE_EMBEDDING)

    with open(CORPUS_PATH, encoding="utf-8") as f:
        corpus = json.load(f)
    print(f"{len(corpus)} fiches chargées")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(NOM_COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(NOM_COLLECTION, metadata={"hnsw:space": "cosine"})

    textes = [f["text"] for f in corpus]
    ids = [f["id"] for f in corpus]
    metadatas = [dict(f["metadata"], type=f["type"]) for f in corpus]

    print("Génération des embeddings...")
    embeddings = modele.encode(textes, batch_size=16, show_progress_bar=True, normalize_embeddings=True)

    print("Insertion dans Chroma...")
    TAILLE_LOT = 200
    for i in range(0, len(corpus), TAILLE_LOT):
        collection.add(
            ids=ids[i:i + TAILLE_LOT],
            embeddings=embeddings[i:i + TAILLE_LOT].tolist(),
            documents=textes[i:i + TAILLE_LOT],
            metadatas=metadatas[i:i + TAILLE_LOT],
        )

    print(f"\nIndexation terminée : {collection.count()} fiches dans '{NOM_COLLECTION}'")


if __name__ == "__main__":
    main()
