"""
SALUTATIONS — Détection en code (pas dans le prompt LLM), pour 2 raisons :
rapidité (pas d'appel API pour dire "bonjour") et fiabilité (pas de variation
d'un appel à l'autre).

Deux mécanismes combinés :
1. Correspondance exacte (rapide, pas de risque de faux positif)
2. Tolérance aux fautes de frappe (ex: "bonsir" -> "bonsoir"), pour les cas
   où l'étudiant fait une faute sur le mot de politesse lui-même -- ce que
   le correcteur de fautes du reste du pipeline (pretraitement.py) ne
   couvre pas, puisqu'il ne connaît que le vocabulaire du domaine
   (bac, universités...), pas les formules de politesse.
"""
import re
import difflib
from utils_texte import normaliser

MOTIFS_BONJOUR = re.compile(
    r"^\s*(bonjour|salut|hello|hi|coucou|bjr|slt)\s*[!.]*\s*$", re.IGNORECASE
)
MOTIFS_BONSOIR = re.compile(r"^\s*(bonsoir|bsr)\s*[!.]*\s*$", re.IGNORECASE)
MOTIFS_REMERCIEMENT = re.compile(
    r"^\s*(merci|merci beaucoup|thanks|thank you)\s*[!.]*\s*$", re.IGNORECASE
)
MOTIFS_AUREVOIR = re.compile(
    r"^\s*(au revoir|à bientôt|bye|à plus)\s*[!.]*\s*$", re.IGNORECASE
)

# Formes canoniques utilisées pour la tolérance aux fautes de frappe (mots
# courts, normalisés -- accents/casse retirés par normaliser()). Séparées
# des expressions régulières ci-dessus, qui restent la première vérification
# (rapide, zéro risque de faux positif sur une correspondance exacte).
_FORMES_BONJOUR = ["bonjour", "salut", "coucou"]
_FORMES_BONSOIR = ["bonsoir"]
_FORMES_REMERCIEMENT = ["merci", "merci beaucoup"]
_FORMES_AUREVOIR = ["au revoir"]

# Seuil volontairement strict (proche de 1.0) : on tolère une petite faute
# de frappe (1-2 lettres sur un mot court), mais pas un mot différent qui
# ressemblerait vaguement -- mieux vaut rater une faute de frappe extrême
# que de déclencher une salutation à tort sur une vraie question.
SEUIL_SIMILARITE_SALUTATION = 0.8


def _ressemble_a(message_normalise: str, formes: list) -> bool:
    if not message_normalise:
        return False
    return bool(difflib.get_close_matches(
        message_normalise, formes, n=1, cutoff=SEUIL_SIMILARITE_SALUTATION
    ))


# Vouvoiement partout, pour rester cohérent avec le prompt système (dont
# tous les exemples utilisent "vous") -- éviter qu'un raccourci codé en dur
# tutoie l'étudiant pendant qu'une réponse générée par le LLM le vouvoie.
REPONSE_BONJOUR = "Bonjour ! Je suis votre assistant d'orientation ParcourSup Guinée. Comment puis-je vous aider aujourd'hui ?"
REPONSE_BONSOIR = "Bonsoir ! Je suis votre assistant d'orientation ParcourSup Guinée. Comment puis-je vous aider ce soir ?"
REPONSE_REMERCIEMENT = "Avec plaisir ! N'hésitez pas si vous avez d'autres questions sur votre orientation."
REPONSE_AUREVOIR = "Au revoir, et bonne chance pour la suite de votre orientation !"


def reponse_fixe_si_politesse(message: str) -> str | None:
    """Retourne une réponse toute faite si le message est une simple
    politesse, sinon None (pour que le pipeline normal prenne le relais)."""
    # 1. Correspondance exacte d'abord (rapide, zéro ambiguïté)
    if MOTIFS_BONSOIR.match(message):
        return REPONSE_BONSOIR
    if MOTIFS_BONJOUR.match(message):
        return REPONSE_BONJOUR
    if MOTIFS_REMERCIEMENT.match(message):
        return REPONSE_REMERCIEMENT
    if MOTIFS_AUREVOIR.match(message):
        return REPONSE_AUREVOIR

    # 2. Tolérance aux fautes de frappe, uniquement si le message est court
    # (quelques mots max) -- pour ne jamais risquer de déclencher une
    # salutation à tort sur une vraie question longue contenant par hasard
    # un mot proche d'une formule de politesse.
    message_normalise = normaliser(message)
    if message_normalise and len(message_normalise.split()) <= 3:
        if _ressemble_a(message_normalise, _FORMES_BONSOIR):
            return REPONSE_BONSOIR
        if _ressemble_a(message_normalise, _FORMES_BONJOUR):
            return REPONSE_BONJOUR
        if _ressemble_a(message_normalise, _FORMES_REMERCIEMENT):
            return REPONSE_REMERCIEMENT
        if _ressemble_a(message_normalise, _FORMES_AUREVOIR):
            return REPONSE_AUREVOIR

    return None
