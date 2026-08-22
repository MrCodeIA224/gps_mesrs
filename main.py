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
from fonctions import (
    repondre,
    QUESTIONS_EXEMPLES,
    generer_pdf_conversation,
    enregistrer_evaluation,
    etat_memoire_initial,
)


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

# État interne du pipeline RAG (slots, historique technique pour la
# reformulation, dernier menu proposé) -- distinct de "historique" ci-dessus,
# qui ne sert qu'à afficher les bulles de chat.
if "etat_memoire" not in st.session_state:
    st.session_state.etat_memoire = etat_memoire_initial()

if "question_a_poser" not in st.session_state:
    st.session_state.question_a_poser = None

if "evaluation_visible" not in st.session_state:
    st.session_state.evaluation_visible = False


# ---------------------------------------------------------------------------
# Style global de l'application
#
# NOTE TECHNIQUE : certains sélecteurs ciblent la structure interne de
# Streamlit (data-testid="stDialog"/"stChatMessage", classe générée à partir
# d'un "key" comme .st-key-fab_chat). Ce n'est pas une API publique garantie
# -- à revérifier si une mise à jour de Streamlit change l'apparence.
# ---------------------------------------------------------------------------

COULEUR_ACCENT = "#1B7A43"  # vert institutionnel, rappelle le drapeau guinéen

st.html(
    f"""
    <style>
    /* -----------------------------------------------------------
       Bouton flottant (icône message, fixé au milieu à droite)
       ----------------------------------------------------------- */
    .st-key-fab_chat {{
        position: fixed;
        top: 50%;
        right: 50px;
        transform: translateY(-50%);
        z-index: 9999;
    }}
    .st-key-fab_chat button {{
        width: 60px;
        height: 60px;
        border-radius: 50%;
        font-size: 1.6rem;
        border: none;
        background: {COULEUR_ACCENT};
        color: white;
        box-shadow: 0 6px 18px rgba(27, 122, 67, 0.35);
    }}
    .st-key-fab_chat button:hover {{
        transform: scale(1.08);
        box-shadow: 0 8px 22px rgba(27, 122, 67, 0.45);
        transition: transform 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
    }}

    /* -----------------------------------------------------------
       Modal de discussion : positionné à droite, coins arrondis,
       légère ombre pour bien le détacher de la page derrière.
       ----------------------------------------------------------- */
    div[data-testid="stDialog"] {{
        justify-content: flex-end !important;
        padding-right: 50px;
    }}
    div[data-testid="stDialog"] > div {{
        max-height: 85vh;
        overflow-y: auto;
        border-radius: 18px;
        box-shadow: 0 16px 48px rgba(0, 0, 0, 0.22);
    }}

    /* -----------------------------------------------------------
       Bulles de chat : un peu plus respirantes, coins arrondis.
       ----------------------------------------------------------- */
    div[data-testid="stChatMessage"] {{
        border-radius: 14px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.4rem;
    }}

    /* -----------------------------------------------------------
       Boutons "carte" de la page d'accueil (exemples de questions),
       alignés à gauche plutôt que centrés, pour ressembler à des
       suggestions plutôt qu'à des boutons d'action génériques.
       ----------------------------------------------------------- */
    div[data-testid="stButton"] button p {{
        text-align: left;
    }}
    </style>
    """
)


# ---------------------------------------------------------------------------
# Fenêtre modale — c'est ici que la discussion a lieu
# ---------------------------------------------------------------------------

@st.dialog("🎓 Discuter avec GPS-MESRS", width="large")
def ouvrir_chat() -> None:
    st.caption("Réponses basées sur les guides officiels du MESRS — Guinée.")

    # -----------------------------------------------------------------
    # Zone des messages — scroll interne natif (Streamlit)
    # -----------------------------------------------------------------
    zone_messages = st.container(height=420)
    with zone_messages:
        if not st.session_state.historique:
            st.info("Posez votre première question ci-dessous pour démarrer la conversation. 👇")
        for message in st.session_state.historique:
            avatar = "🧑‍🎓" if message["role"] == "user" else "🎓"
            with st.chat_message(message["role"], avatar=avatar):
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
            with st.chat_message("user", avatar="🧑‍🎓"):
                st.markdown(question_saisie)
            with st.chat_message("assistant", avatar="🎓"):
                with st.spinner("Recherche dans les guides officiels..."):
                    reponse = repondre(question_saisie, st.session_state.etat_memoire)
                st.markdown(reponse)

        st.session_state.historique.append({"role": "assistant", "contenu": reponse})
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
        st.button("🗑️", help="Effacer tous les messages", on_click=_effacer_conversation)

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
                # Ferme le modal : à l'intérieur d'un st.dialog, un st.rerun()
                # explicite est justement la façon de le fermer (contrairement
                # à l'envoi d'un message, où on l'évite pour rester ouvert).
                st.rerun()
            else:
                st.warning("Merci de sélectionner une note avant d'envoyer.")


def _effacer_conversation() -> None:
    """Callback (on_click) plutôt que logique inline : un callback s'exécute
    et met à jour session_state AVANT que le script ne se redessine, donc
    zone_messages lit déjà le nouvel historique (vide) dès ce même clic.
    Une logique inline placée après le rendu de zone_messages, elle,
    n'aurait d'effet visible qu'au clic suivant (l'affichage de cette
    exécution-ci étant déjà construit avec l'ancien historique)."""
    st.session_state.historique = []
    st.session_state.etat_memoire = etat_memoire_initial()
    st.session_state.evaluation_visible = False


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

col_logo, col_titre = st.columns([1, 3], vertical_alignment="center")
with col_logo:
    st.image("./images/logo_mesrs.png", width=110)
with col_titre:
    st.title("GPS-MESRS")
    st.caption("Guide vers les Programmes et Spécialités du MESRS")

st.markdown(
    "##### L'assistant qui t'aide à trouver ta voie parmi les programmes "
    "universitaires officiels du Ministère de l'Enseignement Supérieur "
    "et de la Recherche Scientifique."
)

st.divider()

st.markdown("### Ce que je peux t'aider à trouver")
col_a, col_b = st.columns(2)
with col_a:
    with st.container(border=True):
        st.markdown("📚 **Programmes et filières**")
        st.caption("Disponibles selon ta série de bac")
    with st.container(border=True):
        st.markdown("🏫 **Établissements**")
        st.caption("Qui proposent chaque formation")
with col_b:
    with st.container(border=True):
        st.markdown("💼 **Débouchés**")
        st.caption("De chaque programme")
    with st.container(border=True):
        st.markdown("📝 **Démarches d'inscription**")
        st.caption("Sur ParcourSup Guinée")

st.divider()

st.markdown("##### 💡 Quelques exemples pour commencer :")
cols = st.columns(2)
for i, question in enumerate(QUESTIONS_EXEMPLES):
    with cols[i % 2]:
        if st.button(f"💬 {question}", use_container_width=True, key=f"exemple_{question}"):
            ouvrir_chat_avec_question(question)

st.divider()
st.caption(
    "⚠️ Projet académique — Master 1 IA, DIT. "
    "S'appuie sur des données officielles du MESRS, "
    "mais n'est pas affilié au Ministère."
)
