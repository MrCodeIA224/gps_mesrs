"""
APP.PY — Interface Streamlit + orchestration complète.
Lancer avec : streamlit run app.py

Ordre d'orchestration à chaque message :
  1. Salutation ? -> réponse fixe immédiate, pas d'appel LLM (rapidité/fiabilité)
  2. Mémoire : reformulation de la question + résolution d'un menu proposé
  3. Extraction des slots structurés (ville, profil bac...) pour usage futur
  4. Retrieval : route "fait" (hybride+reranking) ou "liste" (filtrage structuré)
  5. LLM : génération de la réponse avec le prompt système à 34 sections
  6. Mémoire : masquage des données sensibles avant stockage, détection d'un
     nouveau menu proposé dans la réponse
"""
import streamlit as st
from retrieval import MoteurRecherche
from llm import appeler_llm
from memoire import etat_initial, reformuler_avec_historique, extraire_slots, ajouter_echange
from salutations import reponse_fixe_si_politesse
from logs import detecter_intention_metier, enregistrer_echange

st.set_page_config(page_title="Assistant orientation ParcourSup Guinée", page_icon="🎓")
st.title("🎓 Assistant orientation — ParcourSup Guinée")


@st.cache_resource
def charger_moteur():
    return MoteurRecherche()


moteur = charger_moteur()

if "etat" not in st.session_state:
    st.session_state.etat = etat_initial()

for echange in st.session_state.etat["historique"]:
    with st.chat_message("user"):
        st.write(echange["question"])
    with st.chat_message("assistant"):
        st.write(echange["reponse"])

message = st.chat_input("Ta question...")

if message:
    with st.chat_message("user"):
        st.write(message)

    with st.chat_message("assistant"):
        # 1. Court-circuit salutation/politesse (code, pas de retrieval ni LLM)
        reponse_fixe = reponse_fixe_si_politesse(message)
        if reponse_fixe:
            st.write(reponse_fixe)
            st.session_state.etat["historique"].append({"question": message, "reponse": reponse_fixe})
        else:
            with st.spinner("Recherche en cours..."):
                etat = st.session_state.etat

                # 2. Mémoire : reformulation + résolution de menu
                question_traitee = reformuler_avec_historique(message, etat)

                # 3. Extraction des slots structurés (mise à jour en continu)
                extraire_slots(question_traitee, etat)

                # 4. Retrieval (route fait/liste/hors-sujet)
                resultat_recherche = moteur.rechercher(question_traitee)

                # 5. Génération -- court-circuits sans appel LLM de génération
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
                        slots=etat["slots"],
                        note=resultat_recherche.get("note"),
                        historique=etat["historique"],
                    )

            st.write(reponse)

            if resultat_recherche["resultats"]:
                with st.expander("Sources utilisées (debug)"):
                    st.json(resultat_recherche)

            # Observabilité : intention métier + log, sans jamais influencer
            # la recherche ou la génération (déjà décidées ci-dessus)
            intention_metier = detecter_intention_metier(message)
            enregistrer_echange(
                question=message,
                reponse=reponse,
                intention_technique=resultat_recherche["intention"],
                intention_metier=intention_metier,
                nb_resultats=len(resultat_recherche["resultats"]),
                ville=etat["slots"].get("ville"),
            )

            # 6. Mise à jour mémoire (masquage sensible inclus)
            ajouter_echange(etat, message, reponse)
