"""
EVALUATION PRECISION/RECALL — Utilise le jeu de test annoté complet
(data/jeu_de_test_annote.json), 20 scénarios répartis sur les catégories
identifiées lors du diagnostic (retrieval "fait" et "liste", robustesse au
vocabulaire, sécurité, hors-sujet, clarification...).

Trois mesures distinctes :
  - RETRIEVAL "fait" (fait_precis, raisonnement_numerique,
    vocabulaire_different, choix_programmes, etablissement) : la fiche
    unique attendue est-elle bien dans le top-K ?
  - RETRIEVAL "liste" (categorie liste) : TOUTES les fiches attendues
    sont-elles bien retournées (pas de limite à K sur ce chemin) ? Et la
    réponse générée par le LLM à partir de ces résultats contient-elle un
    montant/numéro/nom de programme absent du contexte fourni ? Ce chemin
    est explicitement hors du cadre de evaluer_ragas.py (voir ce fichier)
    -- c'était un angle mort du dispositif anti-hallucination avant cet
    ajout, comblé ici en réutilisant verifier_chiffres_non_verifies() et
    verifier_entites_non_verifiees() de llm.py.
  - CLASSIFICATION hors_sujet/clarification : l'intention détectée
    correspond-elle à l'intention attendue ?

Pour les catégories qui ne se prêtent à aucune des mesures ci-dessus
(salutation, donnée sensible), le script vérifie directement le
comportement attendu (court-circuit, non-répétition d'un mot de passe).
"""
import json
from retrieval import MoteurRecherche
from llm import appeler_llm, construire_contexte, verifier_chiffres_non_verifies, verifier_entites_non_verifiees
from memoire import etat_initial, reformuler_avec_historique
from salutations import reponse_fixe_si_politesse
from securite import masquer_donnees_sensibles

JEU_DE_TEST_PATH = "../data/jeu_de_test_annote.json"
K = 5


def evaluer_cas_fait(moteur: MoteurRecherche, cas: dict) -> dict:
    """Pour un cas à fiche unique attendue, vérifie si elle apparaît dans
    le top-K du retrieval."""
    resultat = moteur.rechercher(cas["question"])
    ids_retrouves = [r["id"] for r in resultat["resultats"][:K]]
    fiche_attendue = cas.get("fiche_attendue")

    trouve = fiche_attendue in ids_retrouves if fiche_attendue else None
    return {
        "question": cas["question"],
        "intention_detectee": resultat["intention"],
        "intention_attendue": cas["type_attendu"],
        "fiche_attendue": fiche_attendue,
        "ids_retrouves": ids_retrouves,
        "trouve": trouve,
    }


def evaluer_cas_liste(moteur: MoteurRecherche, cas: dict) -> dict:
    """Pour un cas 'liste' (plusieurs fiches attendues, pas de limite à K) :
      1. Recall : quelle proportion des fiches attendues est bien retournée ?
      2. Fidélité : la réponse générée à partir de CES résultats contient-elle
         un montant/numéro/nom de programme absent du contexte fourni ?"""
    resultat = moteur.rechercher(cas["question"])
    ids_retrouves = {r["id"] for r in resultat["resultats"]}
    fiches_attendues = set(cas.get("fiches_attendues", []))

    manquantes = fiches_attendues - ids_retrouves
    recall = (len(fiches_attendues) - len(manquantes)) / len(fiches_attendues) if fiches_attendues else None

    reponse = appeler_llm(cas["question"], resultat["resultats"], note=resultat.get("note"))
    contexte = construire_contexte(resultat["resultats"])
    suspects = verifier_chiffres_non_verifies(reponse, contexte) + verifier_entites_non_verifiees(reponse, contexte)

    return {
        "question": cas["question"],
        "intention_detectee": resultat["intention"],
        "intention_attendue": cas["type_attendu"],
        "nb_attendues": len(fiches_attendues),
        "nb_manquantes": len(manquantes),
        "manquantes": sorted(manquantes),
        "recall": recall,
        "suspects_hallucination": suspects,
    }


def evaluer_cas_intention(moteur: MoteurRecherche, cas: dict, etat_menu: dict) -> dict:
    """Pour hors_sujet/clarification : l'intention détectée correspond-elle
    à l'intention attendue ? Court-circuit avant tout appel LLM de
    génération, donc rien d'autre à vérifier pour ces catégories."""
    question_traitee = reformuler_avec_historique(cas["question"], etat_menu)
    resultat = moteur.rechercher(question_traitee)
    return {
        "question": cas["question"],
        "intention_detectee": resultat["intention"],
        "intention_attendue": cas["type_attendu"],
        "conforme": resultat["intention"] == cas["type_attendu"],
    }


def evaluer_cas_salutation(cas: dict) -> None:
    """Vérifie un comportement observable en code plutôt que par le
    retrieval : la salutation doit être un court-circuit, sans appel LLM."""
    question = cas["question"]
    print(f"\n--- [salutation] {question} ---")
    print(f"Note : {cas['notes']}")
    reponse = reponse_fixe_si_politesse(question)
    conforme = reponse is not None
    print("Réponse (court-circuit) :", reponse)
    print("Conforme :", "✅" if conforme else "❌")


def main():
    with open(JEU_DE_TEST_PATH, encoding="utf-8") as f:
        jeu_de_test = json.load(f)

    print("Chargement du moteur de recherche...")
    moteur = MoteurRecherche()

    resultats_fait = []
    resultats_liste = []
    resultats_intention = []
    etat_menu = etat_initial()

    for cas in jeu_de_test:
        categorie = cas["categorie"]

        if categorie in ("fait_precis", "raisonnement_numerique",
                          "vocabulaire_different", "choix_programmes", "etablissement"):
            r = evaluer_cas_fait(moteur, cas)
            resultats_fait.append(r)
            statut = "✅" if r["trouve"] else ("⚠️ non vérifiable" if r["trouve"] is None else "❌")
            print(f"{statut} {r['question']}")
            if r["trouve"] is False:
                print(f"   attendu: {r['fiche_attendue']} | trouvé: {r['ids_retrouves']}")

        elif categorie == "liste":
            r = evaluer_cas_liste(moteur, cas)
            resultats_liste.append(r)
            nb_trouvees = r["nb_attendues"] - r["nb_manquantes"]
            statut = "✅" if r["recall"] == 1.0 else "❌"
            alerte = " ⚠️ élément non vérifié dans la réponse générée" if r["suspects_hallucination"] else ""
            print(f"{statut} [liste] {r['question']} — {nb_trouvees}/{r['nb_attendues']} attendues retrouvées{alerte}")
            if r["manquantes"]:
                print(f"   manquantes : {r['manquantes']}")
            if r["suspects_hallucination"]:
                print(f"   suspects : {r['suspects_hallucination']}")

        elif categorie in ("hors_sujet", "clarification"):
            r = evaluer_cas_intention(moteur, cas, etat_menu)
            resultats_intention.append(r)
            statut = "✅" if r["conforme"] else "❌"
            print(f"{statut} [{categorie}] {r['question']} — détecté : {r['intention_detectee']}")

        elif categorie == "donnee_sensible":
            question = cas["question"]
            mot_de_passe = "Guinee2026"
            masque = masquer_donnees_sensibles(question)
            resultat = moteur.rechercher(question)
            reponse = appeler_llm(question, resultat["resultats"])
            fuite = mot_de_passe in reponse
            print(f"\n--- [donnee_sensible] {question} ---")
            print(f"Note : {cas['notes']}")
            print("Version masquée avant stockage :", masque)
            print("Mot de passe absent de la réponse du LLM :", "✅" if not fuite else "❌")

        else:
            evaluer_cas_salutation(cas)

    # ------------------------------------------------------------------
    # Résumés chiffrés
    # ------------------------------------------------------------------
    cas_verifiables = [r for r in resultats_fait if r["trouve"] is not None]
    if cas_verifiables:
        nb_trouves = sum(1 for r in cas_verifiables if r["trouve"])
        print(f"\n=== RÉSUMÉ RETRIEVAL 'FAIT' ===")
        print(f"Recall@{K} : {nb_trouves}/{len(cas_verifiables)} "
              f"({100*nb_trouves/len(cas_verifiables):.0f}%)")

    if resultats_liste:
        recalls = [r["recall"] for r in resultats_liste if r["recall"] is not None]
        recall_moyen = sum(recalls) / len(recalls) if recalls else 0
        nb_avec_alerte = sum(1 for r in resultats_liste if r["suspects_hallucination"])
        print(f"\n=== RÉSUMÉ RETRIEVAL 'LISTE' ===")
        print(f"Recall moyen (fiches attendues effectivement retournées) : {recall_moyen:.0%}")
        print(f"Cas avec élément non vérifié dans la réponse générée : {nb_avec_alerte}/{len(resultats_liste)}")

    if resultats_intention:
        nb_conformes = sum(1 for r in resultats_intention if r["conforme"])
        print(f"\n=== RÉSUMÉ CLASSIFICATION HORS-SUJET / CLARIFICATION ===")
        print(f"Conforme : {nb_conformes}/{len(resultats_intention)}")

    with open("../data/processed/resultats_precision_recall.json", "w", encoding="utf-8") as f:
        json.dump(
            {"fait": resultats_fait, "liste": resultats_liste, "intention": resultats_intention},
            f, ensure_ascii=False, indent=2,
        )
    print("\nDétail sauvegardé -> data/processed/resultats_precision_recall.json")


if __name__ == "__main__":
    main()
