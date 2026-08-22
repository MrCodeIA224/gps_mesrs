"""
fonction.py — Fonctions utilitaires du projet GPS-MESRS

Ce fichier centralise toute la logique métier du projet : chargement du
moteur de recherche, orchestration retrieval + mémoire + LLM (déléguée à
scripts/), export PDF, enregistrement des évaluations.

Le pipeline RAG réel vit dans scripts/ (retrieval.py, llm.py, memoire.py,
salutations.py, logs.py) -- ce fichier ne fait qu'orchestrer ces briques
pour l'interface définie dans main.py, exactement comme le faisait
auparavant scripts/app.py (supprimé : sa seule raison d'être était cette
orchestration, désormais ici, et main.py offre une interface plus complète
-- bouton flottant, export PDF, évaluation).

Convention de retour : repondre() renvoie toujours un tuple
(reponse: str, sources: list[dict]).
"""

import sys
from pathlib import Path

# scripts/ n'est pas un package Python (ses modules s'importent entre eux
# en absolu, ex. "from llm import appeler_llm_brut", en supposant que
# scripts/ est sur sys.path -- c'est le cas quand on lance un script
# directement depuis ce dossier). On ajoute donc scripts/ à sys.path ici
# pour que ces imports internes continuent de fonctionner tels quels
# quand ce module est importé depuis la racine du projet (main.py).
_SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from retrieval import MoteurRecherche
    from llm import appeler_llm
    from memoire import etat_initial, reformuler_avec_historique, mettre_a_jour_slots_depuis_entites, ajouter_echange
    from salutations import reponse_fixe_si_politesse
    from logs import enregistrer_echange
    _ERREUR_PIPELINE = None
except Exception as erreur:  # dépendances manquantes, clé API absente...
    _ERREUR_PIPELINE = erreur


# ---------------------------------------------------------------------------
# Chargement du moteur de recherche (mis en cache par Streamlit : coûteux --
# charge les modèles d'embedding/reranking et l'index Chroma en mémoire)
# ---------------------------------------------------------------------------

def _charger_moteur():
    import streamlit as st

    @st.cache_resource(show_spinner="Chargement du moteur de recherche...")
    def _construire():
        return MoteurRecherche()

    return _construire()


# ---------------------------------------------------------------------------
# État de mémoire du pipeline RAG (slots, historique technique, dernier
# menu proposé) -- distinct de st.session_state.historique dans main.py,
# qui ne sert qu'à l'affichage des bulles de chat.
# ---------------------------------------------------------------------------

def etat_memoire_initial() -> dict:
    if _ERREUR_PIPELINE is not None:
        return {}
    return etat_initial()


# ---------------------------------------------------------------------------
# Fonction principale utilisée par l'interface (main.py)
# ---------------------------------------------------------------------------

def repondre(question: str, etat_memoire: dict) -> str:
    """
    Prend la question de l'utilisateur et renvoie une réponse, en s'appuyant
    sur le pipeline RAG complet de scripts/ :
        1. Court-circuit salutation/politesse (pas d'appel LLM)
        2. Mémoire : reformulation de la question + résolution d'un menu proposé
        3. Retrieval : route fait/liste/hors-sujet/clarification
        4. Mémoire : mise à jour des slots à partir des entités extraites
        5. Génération de la réponse (ou court-circuit hors-sujet/clarification)
        6. Logs de l'échange
        7. Mémoire : masquage des données sensibles avant stockage

    Args:
        question: la question posée par l'utilisateur.
        etat_memoire: dictionnaire de session créé par etat_memoire_initial(),
            à conserver dans st.session_state d'un appel à l'autre.

    Returns:
        Le texte de la réponse à afficher.
    """
    if _ERREUR_PIPELINE is not None:
        return (
            "⚠️ Le moteur de recherche n'est pas encore configuré sur cette machine "
            f"({_ERREUR_PIPELINE}). Vérifiez l'installation des dépendances de "
            "scripts/requirements.txt, la présence d'une clé MISTRAL_API_KEY dans un "
            "fichier .env, et que l'index vectoriel a bien été construit "
            "(scripts/4_indexer_chroma.py)."
        )

    reponse_fixe = reponse_fixe_si_politesse(question)
    if reponse_fixe:
        return reponse_fixe

    moteur = _charger_moteur()

    question_traitee = reformuler_avec_historique(question, etat_memoire)

    resultat_recherche = moteur.rechercher(question_traitee)

    mettre_a_jour_slots_depuis_entites(etat_memoire, resultat_recherche["entites"])

    if resultat_recherche["intention"] == "hors_sujet":
        reponse = ("Je suis spécialisé dans l'orientation universitaire et les démarches "
                   "ParcourSup Guinée. Posez-moi une question sur ce sujet, je serai ravi de vous aider !")
    elif resultat_recherche["intention"] == "clarification":
        reponse = ("Pour vous répondre précisément, pouvez-vous préciser un peu votre demande ? "
                   "Par exemple : votre profil de bac, la ville ou l'université qui vous intéresse, "
                   "ou le domaine d'études recherché.")
    else:
        reponse = appeler_llm(
            question_traitee,
            resultat_recherche["resultats"],
            slots=etat_memoire["slots"],
            note=resultat_recherche.get("note"),
            historique=etat_memoire["historique"],
        )

    enregistrer_echange(
        question=question,
        reponse=reponse,
        intention_technique=resultat_recherche["intention"],
        intention_metier=resultat_recherche["entites"]["intention_metier"],
        nb_resultats=len(resultat_recherche["resultats"]),
        ville=etat_memoire["slots"].get("ville"),
    )

    ajouter_echange(etat_memoire, question, reponse)

    return reponse


# ---------------------------------------------------------------------------
# Questions d'exemple affichées dans l'interface (page d'accueil)
# ---------------------------------------------------------------------------

QUESTIONS_EXEMPLES = [
    "Quelles filières puis-je choisir avec un bac Sciences Expérimentales (SE) ?",
    "Quels sont les débouchés du programme Génie Informatique ?",
    "Comment créer mon compte sur ParcourSup Guinée ?",
    "Quelle est la procédure en cas d'erreur sur mon numéro INE ?",
]


# ---------------------------------------------------------------------------
# Export de la conversation en PDF
# ---------------------------------------------------------------------------

def generer_pdf_conversation(historique: list[dict]) -> bytes:
    """
    Génère un PDF récapitulatif de la conversation.

    Args:
        historique: liste de messages {"role": "user"/"assistant", "contenu": str}

    Returns:
        Le contenu du PDF en bytes, prêt pour st.download_button.
    """
    from fpdf import FPDF

    SUBSTITUTIONS = {
        "—": "-", "–": "-", "’": "'", "‘": "'",
        "“": '"', "”": '"', "…": "...", "•": "-",
    }

    def nettoyer(texte: str) -> str:
        for original, remplacement in SUBSTITUTIONS.items():
            texte = texte.replace(original, remplacement)
        return texte.encode("latin-1", "replace").decode("latin-1")

    pdf = FPDF()
    pdf.add_page()
    largeur = pdf.epw  # largeur utile de la page (hors marges)

    pdf.set_font("Helvetica", "B", 16)
    pdf.set_x(pdf.l_margin)
    pdf.cell(largeur, 10, nettoyer("GPS-MESRS — Récapitulatif de conversation"), new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.set_x(pdf.l_margin)
    pdf.cell(largeur, 8, nettoyer("Projet académique — Master 1 IA, DIT"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    pdf.set_text_color(0, 0, 0)

    for message in historique:
        auteur = "Vous" if message["role"] == "user" else "GPS-MESRS"
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(largeur, 7, nettoyer(f"{auteur} :"))
        pdf.set_font("Helvetica", "", 10)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(largeur, 6, nettoyer(message["contenu"]))
        pdf.ln(3)

    return bytes(pdf.output())


# ---------------------------------------------------------------------------
# Enregistrement des évaluations de satisfaction (pour la phase d'évaluation)
# ---------------------------------------------------------------------------

def enregistrer_evaluation(note: int, commentaire: str, nb_messages: int) -> None:
    """
    Enregistre une évaluation de satisfaction dans data/evaluations.json.

    Args:
        note: note donnée par l'utilisateur (1 à 5, via st.feedback "stars")
        commentaire: commentaire libre optionnel
        nb_messages: nombre de messages échangés dans la conversation évaluée
    """
    import json
    import os
    from datetime import datetime

    os.makedirs("data", exist_ok=True)
    chemin = "data/evaluations.json"

    evaluations = []
    if os.path.exists(chemin):
        try:
            with open(chemin, "r", encoding="utf-8") as f:
                evaluations = json.load(f)
        except (json.JSONDecodeError, IOError):
            evaluations = []

    evaluations.append({
        "date": datetime.now().isoformat(timespec="seconds"),
        "note_sur_5": note,
        "commentaire": commentaire,
        "nb_messages": nb_messages,
    })

    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(evaluations, f, ensure_ascii=False, indent=2)
