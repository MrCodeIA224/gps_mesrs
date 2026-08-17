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
# INITIALISATION
# ============================================================

print("Chargement du modèle d'embedding...")
model = SentenceTransformer(
    MODEL_NAME,
    device="cpu"
)
print("Connexion à ChromaDB...")
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_collection(name=COLLECTION_NAME)

print(f"Documents disponibles : {collection.count()}")


# ============================================================
# 1. EXTRACTION DU PROFIL
# ============================================================

def extract_profile(question):
    question_upper = question.upper()

    # Important : SE-FA doit être testé avant SE.
    series = ["SE-FA", "SM", "SE", "SS"]

    serie = None

    for s in series:
        if re.search(rf"\b{re.escape(s)}\b", question_upper):
            serie = s
            break

    # Moyenne : 15/20, 14,5/20, etc.
    match = re.search(
        r"(\d+(?:[.,]\d+)?)\s*/\s*20",
        question
    )

    moyenne = None

    if match:
        moyenne = float(
            match.group(1).replace(",", ".")
        )

    villes = [
        "Conakry",
        "Mamou",
        "Kindia",
        "Labé",
        "Kankan",
        "Faranah",
        "Boké",
        "N'Zérékoré",
        "Nzérékoré",
    ]

    ville = None

    for v in villes:
        if v.lower() in question.lower():
            ville = v
            break

    return {
        "serie": serie,
        "moyenne": moyenne,
        "ville": ville,
    }


# ============================================================
# 2. DÉTECTION DE L'INTENTION
# ============================================================

def detect_intent(question):
    q = question.lower()

    verification_patterns = [
        "puis-je",
        "puis je",
        "est-ce que je peux",
        "est ce que je peux",
        "suis-je éligible",
        "suis je éligible",
        "ai-je droit",
        "est-ce que je suis admissible",
    ]

    recommendation_patterns = [
        "quelles formations",
        "quels programmes",
        "que puis-je choisir",
        "que puis je choisir",
        "quelles filières",
        "que me conseillez",
        "quelles sont mes options",
    ]

    for pattern in verification_patterns:
        if pattern in q:
            return "verification_eligibilite"

    for pattern in recommendation_patterns:
        if pattern in q:
            return "recommandation"

    return "information"


# ============================================================
# 3. RECHERCHE VECTORIELLE
# ============================================================

def semantic_search(question, top_k=20):

    query_embedding = model.encode(
        question,
        normalize_embeddings=True
    )

    return collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )


# ============================================================
# 4. TEST DES RÈGLES MÉTIER
# ============================================================

def evaluate_eligibility(metadata, profile):

    checks = {
        "serie": None,
        "moyenne": None,
        "ville": None,
    }

    reasons = []

    # --------------------------------------------------------
    # Série
    # --------------------------------------------------------

    if profile["serie"]:

        options = str(
            metadata.get("options_autorisees", "")
        ).upper()

        if options:
            allowed = [
                x.strip()
                for x in options.split(",")
            ]

            checks["serie"] = (
                profile["serie"] in allowed
            )

            if not checks["serie"]:
                reasons.append(
                    f"La série {profile['serie']} "
                    "n'est pas autorisée."
                )

    # --------------------------------------------------------
    # Moyenne
    # --------------------------------------------------------

    if profile["moyenne"] is not None:

        seuil = metadata.get("seuil_bac")

        if seuil not in (None, ""):

            try:
                seuil = float(seuil)

                checks["moyenne"] = (
                    profile["moyenne"] >= seuil
                )

                if not checks["moyenne"]:
                    reasons.append(
                        f"Moyenne {profile['moyenne']}/20 "
                        f"inférieure au seuil de {seuil}/20."
                    )

            except (TypeError, ValueError):
                pass

    # --------------------------------------------------------
    # Ville
    # --------------------------------------------------------

    if profile["ville"]:

        ville_document = str(
            metadata.get("ville", "")
        )

        checks["ville"] = (
            profile["ville"].lower()
            in ville_document.lower()
        )

        if not checks["ville"]:
            reasons.append(
                f"Le programme n'est pas situé à "
                f"{profile['ville']}."
            )

    # False sur une règle connue = non éligible.
    eligible = not any(
        value is False
        for value in checks.values()
    )

    return eligible, checks, reasons


# ============================================================
# 5. TRAITEMENT D'UNE VÉRIFICATION
# ============================================================

def handle_verification(question, profile):

    # Pour une vérification précise, on cherche les résultats
    # les plus proches de la formulation de l'utilisateur.
    search_results = semantic_search(
        question,
        top_k=10
    )

    documents = search_results["documents"][0]
    metadatas = search_results["metadatas"][0]
    distances = search_results["distances"][0]

    # Première version :
    # le résultat vectoriel le plus proche est considéré
    # comme le programme ciblé.
    document = documents[0]
    metadata = metadatas[0]
    distance = distances[0]

    eligible, checks, reasons = evaluate_eligibility(
        metadata,
        profile
    )

    return {
        "document": document,
        "metadata": metadata,
        "distance": distance,
        "eligible": eligible,
        "checks": checks,
        "reasons": reasons,
    }


# ============================================================
# 6. TRAITEMENT D'UNE RECOMMANDATION
# ============================================================

def handle_recommendation(
    question,
    profile,
    candidate_k=20,
    final_k=5
):

    search_results = semantic_search(
        question,
        top_k=candidate_k
    )

    documents = search_results["documents"][0]
    metadatas = search_results["metadatas"][0]
    distances = search_results["distances"][0]

    selected = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        eligible, checks, reasons = evaluate_eligibility(
            metadata,
            profile
        )

        if eligible:

            selected.append({
                "document": document,
                "metadata": metadata,
                "distance": distance,
                "checks": checks,
            })

        if len(selected) >= final_k:
            break

    return selected


# ============================================================
# 7. PIPELINE PRINCIPAL
# ============================================================

def process_question(question):

    profile = extract_profile(question)
    intent = detect_intent(question)

    if intent == "verification_eligibilite":

        result = handle_verification(
            question,
            profile
        )

    else:

        result = handle_recommendation(
            question,
            profile
        )

    return {
        "question": question,
        "intent": intent,
        "profile": profile,
        "result": result,
    }


# ============================================================
# 8. AFFICHAGE
# ============================================================

def display_response(response):

    print("\n" + "=" * 80)
    print("ANALYSE DE LA QUESTION")
    print("=" * 80)

    print("Intention :", response["intent"])
    print("Profil :", response["profile"])

    # --------------------------------------------------------
    # Vérification d'éligibilité
    # --------------------------------------------------------

    if response["intent"] == "verification_eligibilite":

        result = response["result"]
        metadata = result["metadata"]

        print("\nProgramme identifié :")
        print(
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

        print(
            f"Distance sémantique : "
            f"{result['distance']:.4f}"
        )

        print("\nVérification des contraintes :")
        print(result["checks"])

        if result["eligible"]:
            print("\nDÉCISION : ÉLIGIBLE")
        else:
            print("\nDÉCISION : NON ÉLIGIBLE")

            print("\nMotif(s) :")

            for reason in result["reasons"]:
                print("-", reason)

    # --------------------------------------------------------
    # Recommandation / information
    # --------------------------------------------------------

    else:

        results = response["result"]

        print("\nProgrammes trouvés :")

        if not results:
            print(
                "Aucun programme compatible trouvé."
            )
            return

        for i, result in enumerate(
            results,
            start=1
        ):

            metadata = result["metadata"]

            print(f"\n--- Résultat {i} ---")

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
                "Séries :",
                metadata.get("options_autorisees")
            )

            print(
                "Seuil :",
                metadata.get("seuil_bac")
            )

            print(
                f"Distance : "
                f"{result['distance']:.4f}"
            )


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

if __name__ == "__main__":

    question = input(
        "\nPosez votre question d'orientation : "
    )

    response = process_question(question)

    display_response(response)