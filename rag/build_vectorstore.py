from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from normalize_data import load_all_documents


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CHROMA_DIR = BASE_DIR / "chroma_db"

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

COLLECTION_NAME = "orientation_guinee"


# ============================================================
# CHARGEMENT DES DOCUMENTS
# ============================================================

documents = load_all_documents()

print(f"Nombre de documents à indexer : {len(documents)}")


# ============================================================
# CHARGEMENT DU MODÈLE D'EMBEDDING
# ============================================================

print("\nChargement du modèle d'embedding...")

model = SentenceTransformer(MODEL_NAME)

print("Modèle chargé.")


# ============================================================
# PRÉPARATION DES DONNÉES
# ============================================================

ids = []
texts = []
metadatas = []

for document in documents:
    ids.append(document["id"])
    texts.append(document["text"])
    metadatas.append(document["metadata"])


# ============================================================
# CRÉATION DES EMBEDDINGS
# ============================================================

print("\nCréation des embeddings...")

embeddings = model.encode(
    texts,
    batch_size=16,
    show_progress_bar=True,
    normalize_embeddings=True
)

print(f"Nombre d'embeddings créés : {len(embeddings)}")
print(f"Dimension d'un embedding : {len(embeddings[0])}")


# ============================================================
# INITIALISATION DE CHROMADB
# ============================================================

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)


# ============================================================
# CRÉATION DE LA COLLECTION
# ============================================================

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)


# ============================================================
# INSERTION DES DOCUMENTS
# ============================================================

print("\nInsertion dans ChromaDB...")

collection.upsert(
    ids=ids,
    documents=texts,
    metadatas=metadatas,
    embeddings=embeddings.tolist()
)


print("\nIndexation terminée.")
print(f"Nombre de documents dans ChromaDB : {collection.count()}")
print(f"Base sauvegardée dans : {CHROMA_DIR}")