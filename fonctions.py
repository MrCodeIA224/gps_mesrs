"""
fonction.py — Fonctions utilitaires du projet GPS-MESRS

Ce fichier centralise toute la logique métier du projet :
chargement des données, embeddings, retrieval, appel au LLM.

Statut actuel : la fonction repondre() est un BOUCHON (mock).
Elle sera remplacée par le vrai pipeline RAG une fois que la partie
"Cœur RAG" (embeddings + retrieval + LLM) sera prête.

Convention de retour : repondre() renvoie toujours un tuple
(reponse: str, sources: list[dict]) pour que main.py n'ait rien
à changer le jour où on branche le vrai pipeline.
"""

import time
import random


# ---------------------------------------------------------------------------
# TODO (Personne B) : remplacer ce bloc par le chargement réel des données
# (JSON -> chunks -> embeddings -> base vectorielle), au démarrage de l'app.
# ---------------------------------------------------------------------------

def charger_base_connaissances():
    """
    TODO : charger et indexer les fichiers JSON (data/) dans la base
    vectorielle (ex. ChromaDB, FAISS). Appelée une seule fois au
    démarrage de l'application (voir main.py, mise en cache Streamlit).
    """
    pass


# ---------------------------------------------------------------------------
# Fonction principale utilisée par l'interface (main.py)
# ---------------------------------------------------------------------------

def repondre(question: str) -> tuple[str, list[dict]]:
    """
    Prend la question de l'utilisateur et renvoie une réponse.

    TODO (Personne B) : remplacer ce bouchon par le vrai pipeline :
        1. Embedding de la question
        2. Recherche des chunks pertinents (retrieval)
        3. Appel au LLM avec la question + les chunks trouvés
        4. Retour de la réponse générée + des sources utilisées

    Args:
        question: la question posée par l'utilisateur.

    Returns:
        Un tuple (reponse, sources) où :
        - reponse (str) : le texte de la réponse à afficher
        - sources (list[dict]) : liste des sources utilisées, chacune
          sous la forme {"titre": ..., "source": ...}
    """
    time.sleep(0.6)  # simule un temps de traitement réaliste

    reponses_bouchon = [
        "Ceci est une réponse simulée. Le vrai pipeline RAG n'est pas encore branché.",
        "[Réponse de test] Une fois le retrieval prêt, cette réponse viendra des guides officiels du MESRS.",
    ]

    sources_bouchon = [
        {"titre": "Exemple de programme — Économie", "source": "matrice_programmes"},
        {"titre": "Critères d'orientation — Bac SE", "source": "guide_orientation"},
    ]

    # sources_bouchon = [
    #     {"titre": "Exemple de programme — Économie", "source": "matrice_programmes"},
    #     {"titre": "Critères d'orientation — Bac SE", "source": "guide_orientation"},
    # ]

    return random.choice(reponses_bouchon), sources_bouchon


# ---------------------------------------------------------------------------
# Questions d'exemple affichées dans l'interface (sidebar)
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