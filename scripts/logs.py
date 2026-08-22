"""
LOGS — Journalisation structurée et anonymisée de chaque échange, pour
permettre l'analyse d'usage réel du chatbot une fois en service (quelles
intentions reviennent le plus souvent, taux de questions sans réponse,
répartition géographique des demandes...).

Conçu comme la brique de données brutes qui alimentera, plus tard si le
temps le permet, un dashboard de suivi -- mais utile en soi dès
maintenant, y compris sans dashboard, pour toute analyse manuelle
(ouvrir le fichier de logs et compter).

RÈGLE DE SÉCURITÉ (héritée de securite.py, appliquée systématiquement ici) :
aucune donnée sensible (mot de passe, code, INE brut) ne doit jamais être
écrite dans un fichier de log, même masquée par erreur d'une étape en
amont -- on remasque explicitement avant écriture, par défense en profondeur.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from securite import masquer_donnees_sensibles
from llm import appeler_llm_brut

# Chemin absolu (indépendant du répertoire de travail courant) : ce module
# est désormais aussi importé depuis main.py à la racine du projet, pas
# seulement exécuté depuis scripts/.
LOGS_PATH = str(Path(__file__).resolve().parent.parent / "data" / "logs_conversations.jsonl")

INTENTIONS_METIER = [
    "procedure", "recherche_programme", "recherche_universite", "paiement",
    "inscription", "resultat", "bourse", "reorientation",
    "probleme_technique", "recuperation_compte", "autre",
]


def detecter_intention_metier(question: str) -> str:
    """Classifie la question dans une intention métier, à des fins
    d'observabilité uniquement -- ne modifie jamais le comportement de
    recherche ou de génération (voir discussion : ce champ sert les logs,
    pas le routage technique, déjà assuré par ailleurs)."""
    prompt = f"""Classe cette question d'étudiant dans UNE SEULE de ces catégories :
{', '.join(INTENTIONS_METIER)}

Question : "{question}"

Réponds uniquement avec le nom de la catégorie, rien d'autre."""
    try:
        reponse = appeler_llm_brut(prompt).strip().lower()
        return reponse if reponse in INTENTIONS_METIER else "autre"
    except Exception:
        return "autre"


def enregistrer_echange(question: str, reponse: str, intention_technique: str,
                          intention_metier: str, nb_resultats: int,
                          ville: str | None = None) -> None:
    """Écrit une ligne de log JSONL, avec la question et la réponse
    systématiquement remasquées avant écriture (défense en profondeur,
    indépendante du masquage déjà fait ailleurs dans le pipeline)."""
    entree = {
        "horodatage": datetime.now(timezone.utc).isoformat(),
        "question": masquer_donnees_sensibles(question),
        "reponse_extrait": masquer_donnees_sensibles(reponse[:200]),
        "intention_technique": intention_technique,   # fait / liste / hors_sujet / clarification
        "intention_metier": intention_metier,           # paiement / bourse / recherche_programme...
        "nb_resultats": nb_resultats,
        "ville": ville,
    }

    os.makedirs(os.path.dirname(LOGS_PATH), exist_ok=True)
    with open(LOGS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entree, ensure_ascii=False) + "\n")


def resume_logs() -> dict:
    """Analyse rapide des logs déjà enregistrés -- utilisable dès
    maintenant sans dashboard, pour un premier aperçu de l'usage."""
    if not os.path.exists(LOGS_PATH):
        return {"nb_echanges": 0}

    from collections import Counter
    intentions_metier = Counter()
    intentions_techniques = Counter()
    villes = Counter()
    nb_sans_resultat = 0
    nb_total = 0

    with open(LOGS_PATH, encoding="utf-8") as f:
        for ligne in f:
            entree = json.loads(ligne)
            nb_total += 1
            intentions_metier[entree["intention_metier"]] += 1
            intentions_techniques[entree["intention_technique"]] += 1
            if entree.get("ville"):
                villes[entree["ville"]] += 1
            if entree["nb_resultats"] == 0:
                nb_sans_resultat += 1

    return {
        "nb_echanges": nb_total,
        "intentions_metier": dict(intentions_metier.most_common()),
        "intentions_techniques": dict(intentions_techniques.most_common()),
        "villes_demandees": dict(villes.most_common()),
        "taux_sans_resultat": round(100 * nb_sans_resultat / nb_total, 1) if nb_total else 0,
    }


if __name__ == "__main__":
    print(json.dumps(resume_logs(), ensure_ascii=False, indent=2))
