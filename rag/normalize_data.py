import json
from pathlib import Path


# ============================================================
# CHEMINS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

FILES = [
    "fiches_etablissements_rag.json",
    "fiches_programmes_rag.json",
    "orientation_rag.json",
    "referentiel_programme_rag.json",
]


# ============================================================
# CHARGEMENT JSON
# ============================================================

def load_json(filename):
    path = DATA_DIR / filename

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# CONVERSION D'UNE VALEUR EN TEXTE
# ============================================================

def value_to_text(value):
    """
    Convertit proprement une valeur JSON en texte.
    """

    if value is None:
        return ""

    if isinstance(value, list):
        return ", ".join(str(v) for v in value)

    if isinstance(value, dict):
        return " ; ".join(
            f"{key}: {value_to_text(val)}"
            for key, val in value.items()
            if val is not None
        )

    return str(value)


# ============================================================
# NORMALISATION D'UN DOCUMENT
# ============================================================

def normalize_document(item, filename):
    """
    Transforme un élément JSON en document utilisable
    pour le futur système RAG.
    """

    document_id = item.get("id", "")
    titre = item.get("titre", "")
    source = item.get("source", "")
    contenu = item.get("contenu", {})
    metadata_original = item.get("metadata", {})

    # --------------------------------------------------------
    # Construction du texte à vectoriser
    # --------------------------------------------------------

    text_parts = []

    if titre:
        text_parts.append(f"Titre : {titre}")

    if isinstance(contenu, dict):

        for key, value in contenu.items():

            if value is not None and value != "":
                texte_value = value_to_text(value)

                text_parts.append(
                    f"{key.replace('_', ' ').capitalize()} : {texte_value}"
                )

    else:
        text_parts.append(str(contenu))

    text = "\n".join(text_parts)

    # --------------------------------------------------------
    # Construction des métadonnées
    # --------------------------------------------------------

    metadata = {
        "source": source,
        "fichier": filename,
    }

    if isinstance(metadata_original, dict):
        metadata.update(metadata_original)

    # Quelques informations importantes de contenu
    # sont également ajoutées aux métadonnées lorsqu'elles existent.

    if isinstance(contenu, dict):

        important_fields = [
            "ies",
            "ville",
            "faculte",
            "departement",
            "programme",
            "options_autorisees",
            "seuil_bac",
        ]

        for field in important_fields:

            if field in contenu and contenu[field] is not None:

                metadata[field] = value_to_text(contenu[field])

    return {
        "id": document_id,
        "text": text,
        "metadata": metadata,
    }


# ============================================================
# NORMALISATION DE TOUS LES FICHIERS
# ============================================================

def load_all_documents():

    documents = []

    for filename in FILES:

        data = load_json(filename)

        for item in data:

            document = normalize_document(item, filename)
            documents.append(document)

    return documents


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    documents = load_all_documents()

    print("=" * 70)
    print(f"Nombre total de documents normalisés : {len(documents)}")
    print("=" * 70)

    print("\nPREMIER DOCUMENT NORMALISÉ\n")

    print("ID :", documents[0]["id"])

    print("\nTEXTE :")
    print(documents[0]["text"])

    print("\nMÉTADONNÉES :")
    print(documents[0]["metadata"])