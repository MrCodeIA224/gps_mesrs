"""
main.py — Point d'entrée de l'application GPS-MESRS

Structure :
  - Une page d'accueil unique (pas de sidebar) qui présente le projet.
  - Un bouton flottant (icône message), fixé au milieu à droite de l'écran.
  - Un clic sur ce bouton ouvre une fenêtre modale (st.dialog) contenant
    le chat — c'est là que la discussion avec GPS-MESRS a lieu.

Lancement :
    streamlit run main.py
"""

import streamlit as st
from fonctions import repondre, QUESTIONS_EXEMPLES, generer_pdf_conversation, enregistrer_evaluation


# ---------------------------------------------------------------------------
# Configuration de la page
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="GPS-MESRS",
    page_icon="./images/logo_mesrs.png",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# État de session
# ---------------------------------------------------------------------------

if "historique" not in st.session_state:
    st.session_state.historique = []

if "question_a_poser" not in st.session_state:
    st.session_state.question_a_poser = None

if "evaluation_visible" not in st.session_state:
    st.session_state.evaluation_visible = False


# ---------------------------------------------------------------------------
# Style du bouton flottant (icône message, fixé au milieu à droite)
#
# NOTE TECHNIQUE : on cible le bouton via la classe CSS générée par
# Streamlit à partir de son "key" (.st-key-fab_chat). C'est la méthode
# recommandée actuellement, mais elle dépend d'un détail d'implémentation
# interne de Streamlit — à revérifier si une mise à jour de Streamlit
# casse l'apparence du bouton.
# ---------------------------------------------------------------------------

st.html(
    """
    <style>
    .st-key-fab_chat {
        position: fixed;
        top: 50%;
        right: 50px;
        transform: translateY(-50%);
        z-index: 9999;
    }
    .st-key-fab_chat button {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        font-size: 1.6rem;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
        border: none;
    }
    .st-key-fab_chat button:hover {
        transform: scale(1.08);
        transition: transform 0.15s ease-in-out;
    }

    /* ---------------------------------------------------------------
       Positionnement du modal (st.dialog) à droite + scroll interne.
       NOTE TECHNIQUE : st.dialog n'offre pas de paramètre officiel de
       position. On cible ici la structure interne connue de Streamlit
       (data-testid="stDialog"), mais ce n'est pas une API publique
       garantie. Si l'alignement à droite ou le scroll ne s'appliquent
       pas visuellement chez vous, ouvrez l'inspecteur du navigateur
       (clic droit sur le modal → Inspecter), repérez la vraie classe
       ou le data-testid du conteneur, et ajustez les sélecteurs
       ci-dessous en conséquence.
       --------------------------------------------------------------- */
    div[data-testid="stDialog"] {
        justify-content: flex-end !important;
        padding-right: 50px;
    }
    div[data-testid="stDialog"] > div {
        max-height: 85vh;
        overflow-y: auto;
    }
    </style>
    """
)


# ---------------------------------------------------------------------------
# Fenêtre modale — c'est ici que la discussion a lieu
# ---------------------------------------------------------------------------

@st.dialog("Discuter avec GPS-MESRS")
def ouvrir_chat() -> None:
    # -----------------------------------------------------------------
    # Zone des messages — scroll interne natif (Streamlit)
    # -----------------------------------------------------------------
    zone_messages = st.container(height=420)
    with zone_messages:
        for message in st.session_state.historique:
            with st.chat_message(message["role"]):
                st.markdown(message["contenu"])                

    # -----------------------------------------------------------------
    # Saisie utilisateur
    # -----------------------------------------------------------------
    question_saisie = st.chat_input("Écrivez votre question ici...")

    if st.session_state.question_a_poser:
        question_saisie = st.session_state.question_a_poser
        st.session_state.question_a_poser = None

    if question_saisie:
        st.session_state.historique.append({"role": "user", "contenu": question_saisie})
        st.session_state.evaluation_visible = False

        with zone_messages:
            with st.chat_message("user"):
                st.markdown(question_saisie)
            with st.chat_message("assistant"):
                with st.spinner("Recherche dans les guides officiels..."):
                    reponse = repondre(question_saisie)
                st.markdown(reponse)                

        st.session_state.historique.append(
            {"role": "assistant", "contenu": reponse}
        )
        # Pas de st.rerun() ici : à l'intérieur d'un st.dialog, Streamlit ne
        # doit re-exécuter QUE le dialogue lui-même, pas toute l'appli — un
        # rerun manuel casse ce comportement et referme le modal.

    # -----------------------------------------------------------------
    # Petite barre d'icônes accolée à la saisie — effacer / exporter / noter
    # NOTE : Streamlit ne permet pas d'insérer des boutons directement DANS
    # st.chat_input, à côté de la flèche d'envoi (ce n'est pas exposé par
    # l'API publique). On les place donc juste en dessous, en icônes
    # compactes plutôt qu'en boutons pleine largeur, pour rester proche du
    # niveau de la saisie sans prendre toute la place.
    # -----------------------------------------------------------------
    col_effacer, col_exporter, col_terminer, _ = st.columns([1, 1, 1, 5])

    with col_effacer:
        if st.button("🗑️", help="Effacer tous les messages"):
            st.session_state.historique = []
            st.session_state.evaluation_visible = False

    with col_exporter:
        pdf_bytes = generer_pdf_conversation(st.session_state.historique) if st.session_state.historique else b""
        st.download_button(
            "📄",
            data=pdf_bytes,
            file_name="conversation_gps_mesrs.pdf",
            mime="application/pdf",
            disabled=not st.session_state.historique,
            help="Exporter la conversation en PDF",
        )

    with col_terminer:
        if st.button("⭐", help="Terminer et évaluer la conversation", disabled=not st.session_state.historique):
            st.session_state.evaluation_visible = True

    # -----------------------------------------------------------------
    # Système de notation — affiché après clic sur "Terminer"
    # -----------------------------------------------------------------
    if st.session_state.get("evaluation_visible"):
        st.divider()
        st.markdown("##### Votre degré de satisfaction par rapport aux réponses reçues :")
        note = st.feedback("stars", key="note_satisfaction")
        commentaire = st.text_area(
            "Un commentaire à ajouter ? (optionnel)",
            key="commentaire_satisfaction",
            placeholder="Ce qui a bien fonctionné, ce qui pourrait être amélioré...",
        )
        if st.button("Envoyer mon évaluation", type="primary"):
            if note is not None:
                enregistrer_evaluation(
                    note=note + 1,  # st.feedback "stars" renvoie un index 0-4
                    commentaire=commentaire,
                    nb_messages=len(st.session_state.historique),
                )
                st.session_state.evaluation_visible = False
                st.toast("Merci pour votre évaluation ! 🙏", icon="✅")
            else:
                st.warning("Merci de sélectionner une note avant d'envoyer.")


def ouvrir_chat_avec_question(question: str) -> None:
    st.session_state.question_a_poser = question
    ouvrir_chat()


# ---------------------------------------------------------------------------
# Bouton flottant (icône message)
# ---------------------------------------------------------------------------

if st.button("💬", key="fab_chat", help="Cliquez pour discuter avec GPS-MESRS"):
    ouvrir_chat()


# ---------------------------------------------------------------------------
# Page d'accueil
# ---------------------------------------------------------------------------

col_gauche, col_centre, col_droite = st.columns([1, 1, 1])
with col_centre:
    st.image("./images/logo_mesrs.png", width=150)

st.title("GPS-MESRS")
st.markdown(
    "##### L'assistant qui t'aide à trouver ta voie parmi les programmes "
    "universitaires officiels du Ministère de l'Enseignement Supérieur "
    "et de la Recherche Scientifique."
)
st.divider()

st.markdown(
    """
    ### Ce que je peux t'aider à trouver

    - 📚 Les **programmes et filières** disponibles selon ta série de bac
    - 💼 Les **débouchés** de chaque programme
    - 🏫 Les **établissements** qui proposent chaque formation
    - 📝 Les **démarches d'inscription** sur ParcourSup Guinée
    - Et bien plus encore ! Pose-moi ta question, je ferai de mon mieux pour te répondre.
    """
)

st.divider()

st.markdown("##### 💡 Quelques exemples pour commencer :")
cols = st.columns(2)
for i, question in enumerate(QUESTIONS_EXEMPLES):
    with cols[i % 2]:
        if st.button(question, use_container_width=True, key=f"exemple_{question}"):
            ouvrir_chat_avec_question(question)

st.divider()
st.caption(
    "⚠️ Projet académique — Master 1 IA, DIT. "
    "S'appuie sur des données officielles du MESRS, "
    "mais n'est pas affilié au Ministère."
)