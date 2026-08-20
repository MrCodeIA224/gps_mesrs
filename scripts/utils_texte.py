"""Fonctions utilitaires partagées."""
import re
import json
import unicodedata


def normaliser(texte: str) -> str:
    if not texte:
        return ""
    texte = texte.lower()
    texte = unicodedata.normalize("NFKD", texte)
    texte = "".join(c for c in texte if not unicodedata.combining(c))
    texte = re.sub(r"[^a-z0-9\s]", " ", texte)
    texte = re.sub(r"\s+", " ", texte).strip()
    return texte


def extraire_json_de_texte(texte: str) -> dict:
    """Extrait un objet JSON depuis la réponse brute d'un LLM, quel que
    soit son habillage exact (avec ou sans balises de code, avec ou sans
    texte explicatif autour) -- une simple suppression de préfixe/suffixe
    s'est révélée fragile en changeant de fournisseur LLM : chaque modèle
    a ses propres habitudes de formatage, même avec une consigne stricte
    "réponds uniquement en JSON".

    Cherche le premier bloc {...} correctement équilibré dans le texte et
    le parse, plutôt que de supposer une structure de texte fixe autour."""
    texte = texte.strip()

    try:
        return json.loads(texte)
    except json.JSONDecodeError:
        pass

    debut = texte.find("{")
    if debut == -1:
        raise ValueError(f"Aucun objet JSON trouvé dans la réponse : {texte!r}")

    profondeur = 0
    for i in range(debut, len(texte)):
        if texte[i] == "{":
            profondeur += 1
        elif texte[i] == "}":
            profondeur -= 1
            if profondeur == 0:
                bloc = texte[debut:i + 1]
                return json.loads(bloc)

    raise ValueError(f"Bloc JSON non équilibré dans la réponse : {texte!r}")
