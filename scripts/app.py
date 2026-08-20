"""
APP.PY — Interface Streamlit + orchestration complète.
Lancer avec : streamlit run app.py

Ordre d'orchestration à chaque message :
  1. Salutation ? -> réponse fixe immédiate, pas d'appel LLM (rapidité/fiabilité)
  2. Mémoire : reformulation de la question + résolution d'un menu proposé
  3. Retrieval : route fait/liste/hors-sujet/clarification -- ce SEUL appel
     extrait aussi ville/profil/moyenne/projet/intention_metier (consolidé :
     auparavant 3 appels séparés et redondants faisaient chacun leur propre
     extraction similaire -- réduit ici à 1 seul, passant le total d'appels
     par question de ~6-7 à ~4-5)
  4. Mémoire : mise à jour des slots à partir des entités déjà extraites
     (aucun appel API supplémentaire)
  5. LLM : génération de la réponse avec le prompt système complet
  6. Logs : utilise l'intention métier déjà extraite à l'étape 3
  7. Mémoire : masquage des données sensibles avant stockage, détection d'un
     nouveau menu proposé dans la réponse
"""
import streamlit as st
from retrieval import MoteurRecherche
from llm import appeler_llm
from memoire import etat_initial, reformuler_avec_historique, mettre_a_jour_slots_depuis_entites, ajouter_echange
from salutations import reponse_fixe_si_politesse
from logs import enregistrer_echange

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

                # 3. Retrieval (route fait/liste/hors-sujet/clarification) --
                # extrait aussi les entités (ville, profil, moyenne, projet,
                # intention métier) en un seul appel, réutilisées ci-dessous
                resultat_recherche = moteur.rechercher(question_traitee)

                # 4. Mémoire : mise à jour des slots, sans appel API
                # supplémentaire (utilise les entités déjà extraites à l'étape 3)
                mettre_a_jour_slots_depuis_entites(etat, resultat_recherche["entites"])

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

            # 6. Logs : intention métier déjà extraite à l'étape 3, pas de
            # nouvel appel API dédié
            enregistrer_echange(
                question=message,
                reponse=reponse,
                intention_technique=resultat_recherche["intention"],
                intention_metier=resultat_recherche["entites"]["intention_metier"],
                nb_resultats=len(resultat_recherche["resultats"]),
                ville=etat["slots"].get("ville"),
            )

            # 7. Mise à jour mémoire (masquage sensible inclus)
            ajouter_echange(etat, message, reponse)
