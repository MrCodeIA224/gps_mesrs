from pathlib import Path
import re

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
COLLECTION_NAME = "orientation_guinee"


# ============================================================
# CHARGEMENT DU MODÈLE ET DE CHROMADB
# ============================================================

print("Chargement du modèle d'embedding...")
model = SentenceTransformer(MODEL_NAME)

print("Connexion à ChromaDB...")
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_collection(name=COLLECTION_NAME)


# ============================================================
# EXTRACTION SIMPLE DU PROFIL
# ============================================================

def extract_profile(question):
    question_upper = question.upper()

    # Série
    series = ["SM", "SE-FA", "SE", "SS"]

    serie = None

    for s in series:
        if re.search(rf"\b{re.escape(s)}\b", question_upper):
            serie = s
            break

    # Moyenne : cherche par exemple 15/20 ou 14.5/20
    match = re.search(r"(\d+(?:[.,]\d+)?)\s*/\s*20", question)

    moyenne = None

    if match:
        moyenne = float(match.group(1).replace(",", "."))

    # Ville : première version
    villes_connues = [
        "Conakry",
        "Mamou",
        "Kindia",
        "Labé",
        "Kankan",
        "Faranah",
        "N'Zérékoré",
        "Boké",
    ]

    ville = None

    for v in villes_connues:
        if v.lower() in question.lower():
            ville = v
            break

    return {
        "serie": serie,
        "moyenne": moyenne,
        "ville": ville,
    }


# ============================================================
# RECHERCHE VECTORIELLE
# ============================================================

def semantic_search(question, top_k=20):

    query_embedding = model.encode(
        question,
        normalize_embeddings=True
    )

    return collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )


# ============================================================
# VÉRIFICATION DES CONTRAINTES
# ============================================================

def is_eligible(metadata, profile):

    # --------------------------------------------------------
    # Ville
    # --------------------------------------------------------

    if profile["ville"]:

        ville_document = str(
            metadata.get("ville", "")
        ).lower()

        if profile["ville"].lower() not in ville_document:
            return False

    # --------------------------------------------------------
    # Série du bac
    # --------------------------------------------------------

    if profile["serie"]:

        options = str(
            metadata.get("options_autorisees", "")
        ).upper()

        # Si l'information existe, on vérifie.
        if options and profile["serie"] not in options:
            return False

    # --------------------------------------------------------
    # Moyenne
    # --------------------------------------------------------

    if profile["moyenne"] is not None:

        seuil = metadata.get("seuil_bac")

        if seuil not in (None, ""):
            try:
                seuil = float(seuil)

                if profile["moyenne"] < seuil:
                    return False

            except (TypeError, ValueError):
                pass

    return True


# ============================================================
# RETRIEVAL HYBRIDE
# ============================================================

def hybrid_search(question, candidate_k=20, final_k=5):

    profile = extract_profile(question)

    results = semantic_search(
        question,
        top_k=candidate_k
    )

    final_results = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        if is_eligible(metadata, profile):

            final_results.append(
                {
                    "document": document,
                    "metadata": metadata,
                    "distance": distance,
                }
            )

        if len(final_results) >= final_k:
            break

    return profile, final_results


# ============================================================
# AFFICHAGE
# ============================================================

if __name__ == "__main__":

    question = input(
        "\nPosez votre question d'orientation : "
    )

    profile, results = hybrid_search(question)

    print("\nProfil détecté :")
    print(profile)

    print("\n" + "=" * 80)
    print("RÉSULTATS APRÈS FILTRAGE")
    print("=" * 80)

    if not results:
        print(
            "\nAucun programme correspondant "
            "aux contraintes n'a été trouvé."
        )

    for i, result in enumerate(results, start=1):

        metadata = result["metadata"]

        print(f"\n--- Résultat {i} ---")
        print(f"Distance : {result['distance']:.4f}")

        print(
            "Programme :",
            metadata.get("programme")
        )

        print(
            "Établissement :",
            metadata.get("ies")
        )

        print(
            "Ville :",
            metadata.get("ville")
        )

        print(
            "Séries autorisées :",
            metadata.get("options_autorisees")
        )

        print(
            "Seuil :",
            metadata.get("seuil_bac")
        )

        print("-" * 80)