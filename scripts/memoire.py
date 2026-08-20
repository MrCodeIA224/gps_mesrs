"""
MEMOIRE — Deux mécanismes distincts, comme exigé par le prompt système :

1. Mémoire conversationnelle (fenêtre glissante) : reformule une question de
   suivi ambiguë à partir des derniers échanges. Existait déjà dans le
   premier projet.

2. Mémoire structurée ("slots") : retient durablement, pour toute la durée
   de la session, des informations données à n'importe quel moment
   (ville, profil du bac, projet professionnel...) -- nécessaire pour :
   - l'escalade centre d'appel par ville (sections 9-10 du prompt)
   - la recommandation de programme (sections 6 et 22 du prompt)

3. Suivi du dernier menu de clarification proposé (section 7-8 du prompt) :
   permet d'interpréter une réponse courte de l'étudiant ("2", "le
   problème 3") en la reliant à l'option correspondante.
"""
from llm import appeler_llm_brut
from securite import masquer_donnees_sensibles

TAILLE_FENETRE = 3
SLOTS_CONNUS = ["ville", "profil_bac", "moyenne_bac", "projet_professionnel"]


def etat_initial() -> dict:
    """Structure de session à créer une fois, au début d'une conversation."""
    return {
        "historique": [],           # liste de {question, reponse}
        "slots": {cle: None for cle in SLOTS_CONNUS},
        "dernier_menu": None,       # liste des options du dernier menu proposé, ou None
    }


def reformuler_avec_historique(question: str, etat: dict) -> str:
    """Reformule la question de suivi si elle dépend du contexte récent."""
    historique = etat["historique"]
    if not historique:
        question_a_traiter = question
    else:
        derniers = historique[-TAILLE_FENETRE:]
        historique_texte = "\n".join(
            f"Étudiant : {e['question']}\nAssistant : {e['reponse']}" for e in derniers
        )
        prompt = f"""Voici l'historique récent d'une conversation :

{historique_texte}

Nouvelle question : "{question}"

Si cette question fait référence à quelque chose mentionné dans l'historique,
reformule-la pour qu'elle soit compréhensible seule. Sinon renvoie-la telle quelle.
Réponds uniquement avec la question, sans commentaire."""
        question_a_traiter = appeler_llm_brut(prompt).strip()

    # Résolution d'une réponse courte à un menu proposé juste avant
    # (ex: l'étudiant répond "2" après un menu numéroté -> on relie
    # explicitement au libellé de l'option correspondante)
    if etat["dernier_menu"] and question.strip().isdigit():
        index = int(question.strip()) - 1
        if 0 <= index < len(etat["dernier_menu"]):
            option_choisie = etat["dernier_menu"][index]
            question_a_traiter = f"{option_choisie}"

    return question_a_traiter


def extraire_slots(question: str, etat: dict) -> None:
    """[CONSERVÉE POUR RÉFÉRENCE / COMPATIBILITÉ -- plus appelée par app.py]

    Faisait un appel LLM dédié pour extraire ville/profil_bac/moyenne_bac/
    projet_professionnel. Redondant avec l'extraction déjà faite par
    classifier_intention() dans retrieval.py -- consolidé en un seul appel
    pour réduire le nombre de requêtes API par question (voir
    mettre_a_jour_slots_depuis_entites, qui fait la même mise à jour SANS
    appel API, à partir du résultat déjà obtenu)."""
    prompt = f"""Analyse ce message d'étudiant et extrais, si présentes, les informations
suivantes. Réponds UNIQUEMENT en JSON, sans commentaire ni balise de code.

Message : "{question}"

Format attendu :
{{"ville": null ou "nom de ville", "profil_bac": null ou "SE/SM/SS/SE-FA/SS-FA",
"moyenne_bac": null ou nombre, "projet_professionnel": null ou "texte court"}}

Ne mets une valeur que si elle est explicitement mentionnée dans le message."""

    try:
        reponse = appeler_llm_brut(prompt).strip()
        reponse = reponse.strip("`").replace("json\n", "").strip()
        import json
        extraits = json.loads(reponse)
        for cle in SLOTS_CONNUS:
            valeur = extraits.get(cle)
            if valeur is not None:
                etat["slots"][cle] = valeur
    except Exception:
        pass  # extraction best-effort : en cas d'échec, on ne bloque pas la conversation


def mettre_a_jour_slots_depuis_entites(etat: dict, entites: dict) -> None:
    """Met à jour les slots structurés SANS appel API, à partir des
    entités déjà extraites par MoteurRecherche.rechercher() (champ
    "entites" de son résultat) -- remplace extraire_slots(), qui faisait
    un appel LLM redondant avec celui déjà fait pour la classification
    fait/liste. Même logique de fusion : on ne met à jour un slot que si
    une nouvelle valeur est trouvée, pour ne jamais effacer une info
    connue d'un échange précédent."""
    for cle in SLOTS_CONNUS:
        valeur = entites.get(cle)
        if valeur is not None:
            etat["slots"][cle] = valeur


def detecter_menu_propose(reponse_assistant: str) -> list | None:
    """Détecte si la réponse de l'assistant contient un menu numéroté, pour
    pouvoir interpréter la prochaine réponse courte de l'étudiant.
    Retourne la liste des libellés d'options, ou None si pas de menu."""
    import re
    lignes = reponse_assistant.split("\n")
    options = []
    for ligne in lignes:
        m = re.match(r"^\s*\d+[\.\)]\s+(.+)", ligne.strip())
        if m:
            options.append(m.group(1).strip())
    return options if len(options) >= 2 else None


def ajouter_echange(etat: dict, question: str, reponse: str) -> None:
    """Ajoute l'échange à l'historique, en masquant toute donnée sensible
    AVANT stockage (exigence de sécurité du prompt système, section 11-13 :
    la consigne du LLM ne protège que le texte affiché, pas ce qui est
    conservé en mémoire/logs)."""
    question_masquee = masquer_donnees_sensibles(question)
    etat["historique"].append({"question": question_masquee, "reponse": reponse})
    etat["dernier_menu"] = detecter_menu_propose(reponse)
