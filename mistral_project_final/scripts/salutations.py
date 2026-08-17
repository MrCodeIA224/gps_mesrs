"""
SALUTATIONS — Détection en code (pas dans le prompt LLM), pour 2 raisons :
rapidité (pas d'appel API pour dire "bonjour") et fiabilité (pas de variation
d'un appel à l'autre).
"""
import re

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
    if MOTIFS_BONSOIR.match(message):
        return REPONSE_BONSOIR
    if MOTIFS_BONJOUR.match(message):
        return REPONSE_BONJOUR
    if MOTIFS_REMERCIEMENT.match(message):
        return REPONSE_REMERCIEMENT
    if MOTIFS_AUREVOIR.match(message):
        return REPONSE_AUREVOIR
    return None
