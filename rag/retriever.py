from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
COLLECTION_NAME = "orientation_guinee"


print("Chargement du modèle...")
model = SentenceTransformer(MODEL_NAME)

print("Connexion à ChromaDB...")
client = chromadb.PersistentClient(path=str(CHROMA_DIR))

collection = client.get_collection(
    name=COLLECTION_NAME
)


def search_documents(question, top_k=5):

    query_embedding = model.encode(
        question,
        normalize_embeddings=True
    )

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    return results


def display_results(results):

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    print("\n" + "=" * 80)
    print("RÉSULTATS DE LA RECHERCHE")
    print("=" * 80)

    for i, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1
    ):

        print(f"\n--- Résultat {i} ---")

        print(f"Distance : {distance:.4f}")

        if metadata:
            print("Source :", metadata.get("source"))
            print("Catégorie :", metadata.get("categorie"))
            print("Établissement :", metadata.get("ies"))
            print("Ville :", metadata.get("ville"))
            print("Programme :", metadata.get("programme"))

        print("\nDocument :")
        print(document)

        print("-" * 80)


if __name__ == "__main__":

    question = input(
        "\nPosez votre question d'orientation : "
    )

    results = search_documents(
        question,
        top_k=5
    )

    display_results(results)