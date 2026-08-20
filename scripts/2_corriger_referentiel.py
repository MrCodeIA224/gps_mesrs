"""
INGESTION 2 — Corrige le fichier référentiel des débouchés.

Rappel (déjà établi précédemment dans le projet) : ce fichier ne contient
que des données structurées (listes de postes, employeurs) sans aucun champ
de texte en langage naturel -- indispensable de le générer avant de pouvoir
l'utiliser pour l'embedding (même problème que pour un tableau Excel brut).

Ce script fait aussi la liaison programme_ids (résolution du cas
"Diplomatie et Relations Internationales" identifié précédemment : la fiche
"Sciences Politiques : Relations Internationales" ne doit plus être liée à
tort aux mêmes programmes).
"""
import json
import difflib
import sys
sys.path.insert(0, ".")

INPUT_REFERENTIEL = "../data/raw/referentiel_programme_rag.json"
INPUT_PROGRAMMES = "../data/raw/fiches_programmes_rag.json"
OUTPUT_PATH = "../data/processed/referentiel_corrige.json"

SEUIL_SIMILARITE = 0.72


def normaliser(texte: str) -> str:
    import unicodedata, re as _re
    if not texte:
        return ""
    texte = texte.lower()
    texte = unicodedata.normalize("NFKD", texte)
    texte = "".join(c for c in texte if not unicodedata.combining(c))
    texte = _re.sub(r"[^a-z0-9\s]", " ", texte)
    texte = _re.sub(r"\s+", " ", texte).strip()
    return texte


def construire_description(fiche: dict) -> str:
    titre = fiche["titre"]
    debouches = fiche["contenu"]["debouches"]
    postes = debouches.get("postes", [])
    employeurs = debouches.get("employeurs_texte", "")
    phrase = f"Les débouchés professionnels du programme {titre} incluent notamment : {', '.join(postes)}."
    if employeurs:
        phrase += f" Les employeurs potentiels pour ce programme sont : {employeurs}."
    return phrase


def construire_candidats(fiche_ref: dict) -> list:
    titre = fiche_ref["titre"]
    mention = fiche_ref["metadata"].get("mention")
    programme_base = fiche_ref["metadata"].get("programme_base")
    candidats = [titre]
    if mention and programme_base:
        candidats.append(f"Licence en {mention}")
        candidats.append(f"{programme_base} {mention}")
    candidats.append(titre.replace("-", " ").replace(":", " "))
    return candidats


def appliquer_corrections_connues(referentiel: list) -> list:
    """Corrige 2 cas identifiés manuellement lors de la relecture du projet :
    1. La fiche 'Diplomatie et Relations Internationales' manquait dans le
       référentiel -> on l'ajoute (contenu récupéré depuis la capture
       d'écran du tableau source fournie par l'étudiant). La liste des
       employeurs y était tronquée dans la capture -> à compléter par
       l'équipe si besoin, marqué explicitement ci-dessous.
    2. 'DEVELOPPEMENT COMMUNAUTAIRE' est renommé pour matcher exactement le
       nom du programme ('Licence en Développement Communautaire'), pure
       question de nomenclature (le lien programme_ids était déjà correct).
    """
    # Correction 1 : ajout de la fiche manquante
    nouvelle_fiche = {
        "id": "licence_en_diplomatie_et_relations_internationales",
        "titre": "Licence En Diplomatie Et Relations Internationales",
        "contenu": {
            "debouches": {
                "postes": [
                    "Attaché diplomatique et agent de chancellerie",
                    "Chargé de coopération bilatérale et multilatérale",
                    "Analyste en relations internationales",
                    "Chargé de programme en organisation internationale",
                    "Conseiller en affaires étrangères",
                    "Chargé de protocole",
                    "Consultant en coopération",
                    "Chercheur en géopolitique",
                ],
                # Liste tronquée dans la capture d'écran source -- à compléter
                "employeurs_texte": "ministère des Affaires étrangères [liste à compléter par l'équipe]",
            }
        },
        "metadata": {"programme_base": "DIPLOMATIE", "mention": "Relations Internationales"},
    }
    referentiel.append(nouvelle_fiche)

    # Correction 2 : harmonisation du nom
    for fiche in referentiel:
        if fiche["titre"].strip().upper() == "DEVELOPPEMENT COMMUNAUTAIRE":
            fiche["titre"] = "Licence en Développement Communautaire"

    return referentiel


def main():
    with open(INPUT_REFERENTIEL, encoding="utf-8") as f:
        referentiel = json.load(f)
    with open(INPUT_PROGRAMMES, encoding="utf-8") as f:
        programmes = json.load(f)

    referentiel = appliquer_corrections_connues(referentiel)

    # 1. Générer le texte en langage naturel
    for fiche in referentiel:
        fiche["contenu"]["description"] = construire_description(fiche)

    # 2. Construire l'index de programmes pour la liaison
    index_programmes = {}
    for p in programmes:
        nom_norm = normaliser(p["contenu"]["programme"])
        index_programmes.setdefault(nom_norm, []).append(p["id"])
    noms_normalises = list(index_programmes.keys())

    # Correction connue : "Diplomatie et Relations Internationales" doit
    # être traité comme un cas normal (correspondance exacte attendue), la
    # fiche "Sciences Politiques : Relations Internationales" ne doit PAS
    # se rabattre dessus par erreur -- on limite donc le seuil flou strict.
    nb_exact, nb_flou, nb_aucun = 0, 0, 0
    a_verifier = []
    noms_deja_pris_en_exact = set()

    # Premier passage : repérer tous les noms de programmes déjà couverts
    # par une correspondance EXACTE, pour ne pas les proposer une seconde
    # fois en correspondance floue à une autre fiche débouchés (cause du
    # cas "Sciences Politiques" qui se rabattait à tort sur "Diplomatie").
    for fiche_ref in referentiel:
        for candidat in construire_candidats(fiche_ref):
            cn = normaliser(candidat)
            if cn in index_programmes:
                noms_deja_pris_en_exact.add(cn)
                break

    noms_disponibles_pour_flou = [n for n in noms_normalises if n not in noms_deja_pris_en_exact]

    for fiche_ref in referentiel:
        candidats = construire_candidats(fiche_ref)
        programme_ids = None

        for candidat in candidats:
            cn = normaliser(candidat)
            if cn in index_programmes:
                programme_ids = index_programmes[cn]
                nb_exact += 1
                break

        if programme_ids is None:
            titre_norm = normaliser(fiche_ref["titre"])
            proches = difflib.get_close_matches(titre_norm, noms_disponibles_pour_flou, n=1, cutoff=SEUIL_SIMILARITE)
            if proches:
                programme_ids = index_programmes[proches[0]]
                nb_flou += 1
                a_verifier.append({
                    "referentiel_id": fiche_ref["id"],
                    "referentiel_titre": fiche_ref["titre"],
                    "programme_correspondant": proches[0],
                    "programme_ids": programme_ids,
                })
            else:
                nb_aucun += 1

        fiche_ref["contenu"]["programme_ids"] = programme_ids or []

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(referentiel, f, ensure_ascii=False, indent=2)

    print(f"Référentiel corrigé : {len(referentiel)} fiches")
    print(f"  - correspondance exacte : {nb_exact}")
    print(f"  - correspondance floue  : {nb_flou} (à vérifier manuellement)")
    print(f"  - aucune correspondance : {nb_aucun}")
    for a in a_verifier:
        print(f"    -> {a['referentiel_titre']} => {a['programme_correspondant']}")
    print(f"\nSauvegardé -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
