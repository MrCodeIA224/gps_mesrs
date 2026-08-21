"""
EVALUATION PRECISION/RECALL — Utilise le jeu de test annoté complet
(data/jeu_de_test_annote.json), qui couvre les 20 scénarios identifiés
lors du diagnostic, pas seulement quelques questions de façade.

Contrairement à evaluer_ragas.py (qui mesure la qualité de la réponse
générée), ce script mesure la qualité du RETRIEVAL seul : le bon chunk
attendu est-il bien retrouvé, pour les questions où une fiche précise est
identifiable à l'avance ?

Pour les catégories qui ne se prêtent pas à une métrique numérique
(salutation, menu, liste, donnée sensible...), le script se contente
d'exécuter le pipeline et d'afficher le résultat pour une relecture
manuelle guidée par le champ "notes" de chaque cas.
"""
import json
from retrieval import MoteurRecherche
from llm import appeler_llm
from memoire import etat_initial, reformuler_avec_historique, ajouter_echange
from salutations import reponse_fixe_si_politesse
from securite import masquer_donnees_sensibles

JEU_DE_TEST_PATH = "../data/jeu_de_test_annote.json"
K = 5


def evaluer_cas_fait(moteur: MoteurRecherche, cas: dict) -> dict:
    """Pour un cas 'fait_precis', vérifie si la fiche attendue apparaît
    dans le top-K du retrieval."""
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


def executer_cas_special(moteur: MoteurRecherche, cas: dict, etat_menu: dict) -> None:
    """Pour les autres catégories, exécute simplement le pipeline complet
    et affiche le résultat pour relecture manuelle."""
    question = cas["question"]
    categorie = cas["categorie"]

    print(f"\n--- [{categorie}] {question} ---")
    print(f"Note : {cas['notes']}")

    if categorie == "salutation":
        reponse = reponse_fixe_si_politesse(question)
        print("Réponse (court-circuit) :", reponse)
        return

    if categorie == "donnee_sensible":
        masque = masquer_donnees_sensibles(question)
        print("Version masquée avant stockage :", masque)
        # on continue quand même vers le LLM pour vérifier qu'il ne répète pas le mdp
        resultat = moteur.rechercher(question)
        reponse = appeler_llm(question, resultat["resultats"])
        print("Réponse du LLM :", reponse[:200], "...")
        print("Le mot de passe apparaît-il dans la réponse ? :", "Guinee2026" in reponse)
        return

    question_traitee = reformuler_avec_historique(question, etat_menu)
    resultat = moteur.rechercher(question_traitee)

    if resultat["intention"] in ("hors_sujet", "clarification"):
        print(f"Intention détectée : {resultat['intention']} (court-circuit, pas d'appel LLM)")
        return

    reponse = appeler_llm(question_traitee, resultat["resultats"], note=resultat.get("note"))
    print("Intention détectée :", resultat["intention"])
    print("Nombre de résultats :", len(resultat["resultats"]))
    print("Réponse :", reponse[:300], "...")
    ajouter_echange(etat_menu, question, reponse)


def main():
    with open(JEU_DE_TEST_PATH, encoding="utf-8") as f:
        jeu_de_test = json.load(f)

    print("Chargement du moteur de recherche...")
    moteur = MoteurRecherche()

    resultats_fait = []
    etat_menu = etat_initial()

    for cas in jeu_de_test:
        if cas["categorie"] in ("fait_precis", "raisonnement_numerique",
                                  "vocabulaire_different", "choix_programmes", "etablissement"):
            r = evaluer_cas_fait(moteur, cas)
            resultats_fait.append(r)
            statut = "✅" if r["trouve"] else ("⚠️ non vérifiable" if r["trouve"] is None else "❌")
            print(f"{statut} {r['question']}")
            if r["trouve"] is False:
                print(f"   attendu: {r['fiche_attendue']} | trouvé: {r['ids_retrouves']}")
        else:
            executer_cas_special(moteur, cas, etat_menu)

    # Résumé chiffré, uniquement sur les cas où une fiche précise était identifiable
    cas_verifiables = [r for r in resultats_fait if r["trouve"] is not None]
    if cas_verifiables:
        nb_trouves = sum(1 for r in cas_verifiables if r["trouve"])
        print(f"\n=== RÉSUMÉ ===")
        print(f"Recall@{K} sur les cas vérifiables : {nb_trouves}/{len(cas_verifiables)} "
              f"({100*nb_trouves/len(cas_verifiables):.0f}%)")

    with open("../data/processed/resultats_precision_recall.json", "w", encoding="utf-8") as f:
        json.dump(resultats_fait, f, ensure_ascii=False, indent=2)
    print("\nDétail sauvegardé -> data/processed/resultats_precision_recall.json")


if __name__ == "__main__":
    main()
