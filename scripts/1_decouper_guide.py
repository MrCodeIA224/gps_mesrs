"""
INGESTION 1 — Découpe le guide d'orientation (Markdown) en fiches JSON.

Le guide a été rédigé avec une structure volontairement homogène et déjà
"chunking-ready" (voir les échanges de conception) :
  - Sections numérotées ("1. CRÉATION DU COMPTE") séparées par des lignes de "="
  - Sous-blocs en MAJUSCULES à l'intérieur de chaque section, chacun autonome
    (titre descriptif + phrase de reformulation + étapes numérotées)

Règle de découpage : un sous-bloc en MAJUSCULES = un chunk. S'il n'y a pas de
sous-bloc (texte directement sous le titre de section), le contenu direct sous
le titre de section forme lui-même un chunk. Chaque chunk reçoit le titre de sa section
parente en préfixe, pour rester compréhensible isolément (règle établie dès le
début du projet : pas de dépendance entre chunks).
"""
import json
import re

INPUT_PATH = "../data/raw/guide_orientation_final.md"
OUTPUT_PATH = "../data/processed/guide_orientation_fiches.json"

SEPARATEUR = "=" * 59


def nettoyer_id(titre: str) -> str:
    id_ = titre.lower()
    id_ = re.sub(r"[^a-z0-9àâäéèêëïîôöùûüç\s-]", "", id_)
    id_ = re.sub(r"\s+", "_", id_.strip())
    return id_[:80]


def est_titre_section(ligne: str) -> bool:
    """Ex: '1. CRÉATION DU COMPTE' ou 'QU'EST-CE QUE LE SALON...'"""
    l = ligne.strip()
    if not l:
        return False
    return bool(re.match(r"^(\d+\.\s+)?[A-ZÀÂÄÉÈÊËÏÎÔÖÙÛÜÇ]", l)) and l.upper() == l and len(l) > 4


def est_sous_titre(ligne: str, dans_liste_numerotee: bool) -> bool:
    """Un sous-titre est une ligne entièrement en majuscules, qui n'est pas
    une étape numérotée (1., 2. ...) ni une ligne de séparation.
    On ignore le contenu entre parenthèses ou guillemets français pour ce
    test, car certains titres incluent des exemples en minuscules
    (ex: 'ERREUR DE CONNEXION (« INE incorrect »...)')."""
    l = ligne.strip()
    if not l or l == SEPARATEUR or l.startswith("="):
        return False
    if re.match(r"^\d+\.\s", l):  # étape numérotée -> pas un titre
        return False
    l_sans_parentheses = re.sub(r"\([^)]*\)", "", l)
    l_sans_guillemets = re.sub(r"«[^»]*»", "", l_sans_parentheses)
    l_test = l_sans_guillemets.strip()
    if l_test and l_test.upper() == l_test and any(c.isalpha() for c in l_test) and len(l) > 4:
        return True
    return False


def parser_guide(texte: str) -> list:
    lignes = texte.split("\n")
    fiches = []

    section_courante = None
    sous_titre_courant = None
    buffer = []
    compteur = {}

    def sauvegarder():
        contenu = "\n".join(buffer).strip()
        if not contenu or not section_courante:
            return
        titre_complet = sous_titre_courant or section_courante
        prefixe = f"[{section_courante}] " if sous_titre_courant and sous_titre_courant != section_courante else ""
        texte_final = f"{prefixe}{titre_complet}\n{contenu}" if prefixe else f"{titre_complet}\n{contenu}"

        id_base = nettoyer_id(titre_complet)
        compteur[id_base] = compteur.get(id_base, 0) + 1
        id_final = id_base if compteur[id_base] == 1 else f"{id_base}_{compteur[id_base]}"

        fiches.append({
            "id": id_final,
            "titre": titre_complet,
            "section_parente": section_courante,
            "texte": texte_final,
        })

    i = 0
    while i < len(lignes):
        ligne = lignes[i]
        l = ligne.strip()

        if l == SEPARATEUR:
            # La ligne suivante (non vide) est un titre de section
            j = i + 1
            while j < len(lignes) and not lignes[j].strip():
                j += 1
            if j < len(lignes) and est_titre_section(lignes[j]):
                sauvegarder()
                section_courante = lignes[j].strip()
                sous_titre_courant = None
                buffer = []
                i = j + 1
                # sauter la ligne de fermeture "===" suivante
                while i < len(lignes) and (not lignes[i].strip() or lignes[i].strip() == SEPARATEUR):
                    i += 1
                continue
            i += 1
            continue

        if est_sous_titre(l, False):
            sauvegarder()
            sous_titre_courant = l
            buffer = []
            i += 1
            continue

        buffer.append(ligne)
        i += 1

    sauvegarder()
    return fiches


def main():
    with open(INPUT_PATH, encoding="utf-8") as f:
        texte = f.read()

    fiches = parser_guide(texte)

    resultat = []
    for f in fiches:
        resultat.append({
            "id": f["id"],
            "type": "guide_orientation",
            "text": f["texte"],
            "source": "guide_orientation_final.md",
            "metadata": {"section": f["section_parente"]},
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        json.dump(resultat, out, ensure_ascii=False, indent=2)

    print(f"{len(resultat)} fiches générées depuis le guide -> {OUTPUT_PATH}")
    tailles = [len(f["text"].split()) for f in resultat]
    print(f"Taille moyenne : {sum(tailles)/len(tailles):.0f} mots | max : {max(tailles)} mots")
    print(f"Fiches > 400 mots : {sum(1 for t in tailles if t > 400)}")

    print("\nAperçu des 5 premières fiches :")
    for f in resultat[:5]:
        print(f"  [{f['id']}] ({len(f['text'].split())} mots)")


if __name__ == "__main__":
    main()
