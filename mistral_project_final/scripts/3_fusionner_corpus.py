"""
INGESTION 3 — Fusionne les 4 sources en un seul corpus, schéma homogène
{id, type, text, source, metadata}.
"""
import json

GUIDE = "../data/processed/guide_orientation_fiches.json"
PROGRAMMES = "../data/raw/fiches_programmes_rag.json"
REFERENTIEL = "../data/processed/referentiel_corrige.json"
ETABLISSEMENTS = "../data/raw/fiches_etablissements_rag.json"
OUTPUT_PATH = "../data/processed/corpus_final.json"


def charger(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def nettoyer_meta(meta: dict) -> dict:
    return {k: v for k, v in meta.items() if v is not None}


def convertir_guide(fiche):
    return {
        "id": f"guide_{fiche['id']}",
        "type": "guide_orientation",
        "text": fiche["text"],
        "source": "guide_orientation_final.md",
        "metadata": nettoyer_meta({"section": fiche["metadata"].get("section")}),
    }


def convertir_programme(fiche):
    c = fiche["contenu"]
    return {
        "id": fiche["id"],
        "type": "programme",
        "text": c["description"],
        "source": "fiches_programmes_rag.json",
        "metadata": nettoyer_meta({
            "ies": c.get("ies"), "ville": c.get("ville"), "faculte": c.get("faculte"),
            "departement": c.get("departement"), "programme": c.get("programme"),
            "options_autorisees": ",".join(c.get("options_autorisees", [])),
            "seuil_bac": c.get("seuil_bac"),
        }),
    }


def convertir_referentiel(fiche):
    c = fiche["contenu"]
    return {
        "id": f"debouches_{fiche['id']}",
        "type": "debouches",
        "text": c["description"],
        "source": "referentiel_programme_rag.json",
        "metadata": nettoyer_meta({
            "programme_base": fiche.get("metadata", {}).get("programme_base"),
            "mention": fiche.get("metadata", {}).get("mention"),
            "programme_ids": ",".join(c.get("programme_ids", [])),
        }),
    }


def convertir_etablissement(fiche):
    c = fiche["contenu"]
    return {
        "id": fiche["id"],
        "type": "etablissement",
        "text": c["presentation"],
        "source": "fiches_etablissements_rag.json",
        "metadata": nettoyer_meta({"ies": c.get("ies"), "ville": c.get("ville")}),
    }


def main():
    guide = charger(GUIDE)
    programmes = charger(PROGRAMMES)
    referentiel = charger(REFERENTIEL)
    etablissements = charger(ETABLISSEMENTS)

    corpus = []
    corpus += [convertir_guide(f) for f in guide]
    corpus += [convertir_programme(f) for f in programmes]
    corpus += [convertir_referentiel(f) for f in referentiel]
    corpus += [convertir_etablissement(f) for f in etablissements]

    ids = [f["id"] for f in corpus]
    doublons = set(i for i in ids if ids.count(i) > 1)
    if doublons:
        raise ValueError(f"IDs en double : {doublons}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print("Fusion terminée :")
    print(f"  - guide d'orientation : {len(guide)}")
    print(f"  - programmes          : {len(programmes)}")
    print(f"  - débouchés           : {len(referentiel)}")
    print(f"  - établissements      : {len(etablissements)}")
    print(f"  TOTAL                 : {len(corpus)} fiches")
    print(f"  IDs uniques           : {len(set(ids))}/{len(ids)}")
    print(f"\nSauvegardé -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
