"""
PRETRAITEMENT — Normalisation et correction légère AVANT le classifieur
d'intention et la recherche, pour absorber :
  - abréviations courantes (SM, sc maths, info...)
  - fautes de frappe sur des mots-clés critiques du domaine (bac -> bas)
  - variantes de noms d'universités (univ gamal -> Gamal Abdel Nasser)

Combine volontairement 3 mécanismes (pas un simple dictionnaire figé) :
  1. Dictionnaire d'abréviations manuel, pour les cas très fréquents et
     bien identifiés (rapide, 100% fiable, pas d'ambiguïté)
  2. Vocabulaire du domaine extrait AUTOMATIQUEMENT du corpus (noms d'IES,
     sigles, mots-clés de programmes) -- pas écrit à la main
  3. Correction floue (fuzzy matching) contre ce vocabulaire combiné, pour
     les fautes de frappe non prévues par le dictionnaire

Cette normalisation ne remplace pas le classifieur LLM ni la recherche
hybride -- elle les protège en amont, pour qu'ils reçoivent une question
déjà nettoyée plutôt que de compter uniquement sur leur propre robustesse.
"""
import re
import difflib
from spellchecker import SpellChecker
from utils_texte import normaliser

# Dictionnaire français complet (bibliothèque, pas une liste manuelle) --
# garde-fou principal contre la corruption de mots français courants
# (le bug "maintenant" -> "maintenance" trouvé en test réel : un mot
# français valide ne doit normalement jamais être "corrigé").
_DICTIONNAIRE_FR = SpellChecker(language="fr")

# ----------------------------------------------------------------------
# 1. Dictionnaire d'abréviations et synonymes courants (cas fréquents,
#    difficilement déductibles du corpus seul car ce sont des raccourcis
#    côté étudiant, pas du vocabulaire qui apparaît dans les documents)
# ----------------------------------------------------------------------
DICTIONNAIRE_ABREVIATIONS = {
    "sm": "sciences mathématiques",
    "se": "sciences expérimentales",
    "ss": "sciences sociales",
    "sc maths": "sciences mathématiques",
    "sc math": "sciences mathématiques",
    "sc exp": "sciences expérimentales",
    "sc soc": "sciences sociales",
    "maths": "mathématiques",
    "eco": "économie",
    # Les entrées "math", "info", "gestion", "medecine", "archi" ont été
    # retirées après vérification systématique : le mot tapé par l'étudiant
    # matchait déjà tout seul dans le corpus, sans transformation --
    # ajouter une transformation inutile ne fait qu'introduire un risque
    # (comme "droit") pour aucun bénéfice. Principe retenu : ne transformer
    # un mot de domaine QUE si le mot original ne matche vraiment rien seul.
    #
    # Décalages singulier/pluriel ou forme du mot, trouvés en test
    # systématique sur une large liste de domaines (contrairement à "droit",
    # ici le mot existe bien dans le corpus mais sous une forme légèrement
    # différente que la recherche par sous-chaîne ne capte pas) :
    "tourisme": "touristique",
    "agriculture": "agricole",
    "langues": "langue",
    # "droit" -> retiré volontairement (contrairement aux autres entrées de
    # ce dictionnaire) : les vrais programmes du corpus s'appellent "Licence
    # En Droit", jamais "Sciences Juridiques". Cette transformation
    # remplaçait à tort "droit" par un terme absent du corpus, empêchant
    # TOUTE question sur le droit de trouver un résultat -- bug réel
    # confirmé en test (seul "droit" échouait systématiquement, tous les
    # autres domaines fonctionnaient normalement).
    "univ": "université",
    "univ gamal": "université gamal abdel nasser de conakry",
    "gamal": "gamal abdel nasser",
    "uganc": "université gamal abdel nasser de conakry",
}

# Correspondance profil en toutes lettres -> code stocké dans les métadonnées
# (indispensable : les métadonnées du corpus stockent des codes comme "SM",
# "SE-FA", jamais le nom complet -- sans cette table, le filtre par profil
# échouerait silencieusement même après une bonne extraction du LLM)
PROFILS_BAC_VERS_CODE = {
    "sciences mathématiques": "SM",
    "sciences expérimentales": "SE",
    "sciences sociales": "SS",
    "sciences expérimentales franco-arabe": "SE-FA",
    "sciences sociales franco-arabe": "SS-FA",
    "sm": "SM", "se": "SE", "ss": "SS", "se-fa": "SE-FA", "ss-fa": "SS-FA",
}
# Les clés ci-dessus sont normalisées une fois ici (accents retirés), pour
# correspondre exactement à ce que produit normaliser() au moment du lookup
# -- sinon "mathématiques" (avec accent) ne matcherait jamais "mathematiques".
_PROFILS_BAC_NORMALISE = {normaliser(k): v for k, v in PROFILS_BAC_VERS_CODE.items()}


def code_profil_bac(profil_libre: str) -> str | None:
    """Convertit un profil exprimé librement (par le LLM ou l'étudiant) en
    code exact tel que stocké dans les métadonnées ('SM', 'SE-FA'...).
    Retourne None si aucune correspondance fiable n'est trouvée."""
    if not profil_libre:
        return None
    return _PROFILS_BAC_NORMALISE.get(normaliser(profil_libre))

# Mots-clés critiques du domaine, sur lesquels une faute de frappe change
# complètement le sens si non corrigée (le cas "bac" -> "bas" notamment).
# Utilisés comme cible de la correction floue, en plus du vocabulaire
# extrait automatiquement du corpus.
MOTS_CLES_CRITIQUES = [
    "bac", "baccalauréat", "ine", "orientation", "inscription",
    "biométrie", "biométrique", "bourse", "moyenne", "profil",
    "programme", "université", "faculté", "seuil",
]

SEUIL_SIMILARITE_FLOU = 0.85

# Confusions CONNUES et dangereuses dans ce domaine précis : ce sont de
# vrais mots français (donc le dictionnaire seul les laisserait passer),
# mais dans le contexte de l'orientation universitaire, il s'agit presque
# toujours d'une faute de frappe voulant dire autre chose. Corrigées
# explicitement, MÊME si le mot est un mot français valide -- cas d'usage
# d'origine qui a motivé la création de cette couche de correction.
CONFUSIONS_CONNUES = {
    "bas": "bac",
    # "Konakry" (avec K) est une variante d'orthographe courante de
    # "Conakry" -- trop différente de la forme officielle pour être
    # rattrapée par la seule correction floue (similarité 0.71, sous le
    # seuil de 0.85), alors que c'est la ville la plus demandée du corpus.
    "konakry": "conakry",
    "konakri": "conakry",
}

# Filet de sécurité supplémentaire, en plus du dictionnaire français --
# utile pour des mots très courants qui existeraient malgré tout dans le
# dictionnaire mais qu'on veut protéger sans même faire l'appel au
# dictionnaire (performance, et robustesse en cas d'absence du mot dans
# le dictionnaire pour une raison quelconque).
MOTS_PROTEGES = {
    "faire", "avoir", "etre", "aller", "vouloir", "pouvoir", "devoir", "dire",
    "suis", "peux", "veux", "dois", "vais", "sais", "fais", "est", "sont",
    "ont", "peut", "veut", "doit", "sait", "cette", "avec", "pour", "dans",
    "sur", "sans", "mais", "donc", "alors", "comme", "quel", "quelle",
    "quels", "quelles", "comment", "pourquoi", "combien", "quand", "puis",
    "moi", "toi", "nous", "vous", "leur", "leurs",
}


def construire_vocabulaire_domaine(corpus_metadatas: list) -> set:
    """Extrait automatiquement un vocabulaire de référence depuis le vrai
    corpus (noms d'IES, sigles, mots des noms de programmes) -- pas écrit
    à la main, pour ne pas dépendre uniquement d'un dictionnaire manuel."""
    vocabulaire = set()
    for meta in corpus_metadatas:
        for champ in ("ies", "ville", "programme"):
            valeur = meta.get(champ)
            if valeur:
                for mot in normaliser(valeur).split():
                    if len(mot) > 3:  # ignore les mots très courts (bruit)
                        vocabulaire.add(mot)
    vocabulaire.update(MOTS_CLES_CRITIQUES)
    return vocabulaire


def remplacer_abreviations(question: str) -> str:
    """Remplace les abréviations connues par leur forme complète.
    Fonctionne sur le texte normalisé, en tenant compte des expressions
    à plusieurs mots (ex: 'sc maths') avant les mots isolés.

    IMPORTANT : la substitution se fait en UN SEUL passage (un seul motif
    combiné), pas une boucle avec un re.sub() par abréviation -- sinon le
    texte déjà substitué est rescanné par les abréviations suivantes,
    causant des doublons. Bug réel détecté en test :
    'univ gamal' -> 'université gamal abdel nasser de conakry', puis ce
    résultat était rescanné et 'gamal' (à l'intérieur) était substitué une
    seconde fois par 'gamal abdel nasser', donnant
    '...gamal abdel nasser abdel nasser de conakry'."""
    q_norm = normaliser(question)
    motifs_tries = sorted(DICTIONNAIRE_ABREVIATIONS, key=len, reverse=True)
    motif_combine = re.compile(r"\b(" + "|".join(re.escape(m) for m in motifs_tries) + r")\b")
    return motif_combine.sub(lambda m: DICTIONNAIRE_ABREVIATIONS[m.group(1)], q_norm)


def corriger_fautes_de_frappe(question_normalisee: str, vocabulaire: set) -> str:
    """Corrige chaque mot qui ressemble fortement à un terme du vocabulaire
    de référence, sans y correspondre exactement -- cas des fautes de
    frappe non couvertes par le dictionnaire d'abréviations.

    Ordre des vérifications, du plus prioritaire au plus général :
    1. Confusion connue et dangereuse (ex: bas -> bac) -- corrigée même si
       le mot est un vrai mot français.
    2. Mot déjà présent dans le vocabulaire du domaine -- laissé tel quel.
    3. Mot protégé (liste rapide) ou mot français reconnu par le
       dictionnaire -- laissé tel quel (évite la corruption de mots
       courants comme "maintenant" -> "maintenance").
    4. Sinon, tentative de correction floue contre le vocabulaire du corpus.
    """
    mots = question_normalisee.split()
    mots_corriges = []
    for mot in mots:
        if mot in CONFUSIONS_CONNUES:
            mots_corriges.append(CONFUSIONS_CONNUES[mot])
            continue
        if mot in vocabulaire or mot in MOTS_PROTEGES or len(mot) <= 4:
            mots_corriges.append(mot)
            continue
        if mot in _DICTIONNAIRE_FR:
            mots_corriges.append(mot)
            continue
        proches = difflib.get_close_matches(mot, vocabulaire, n=1, cutoff=SEUIL_SIMILARITE_FLOU)
        mots_corriges.append(proches[0] if proches else mot)
    return " ".join(mots_corriges)


def normaliser_et_corriger(question: str, vocabulaire: set) -> str:
    """Pipeline complet : abréviations puis correction floue."""
    etape1 = remplacer_abreviations(question)
    etape2 = corriger_fautes_de_frappe(etape1, vocabulaire)
    return etape2


if __name__ == "__main__":
    # Test avec un vocabulaire minimal simulé (sans dépendre du corpus réel
    # pour ce test isolé)
    vocab_test = {"bac", "gamal", "abdel", "nasser", "conakry", "universite",
                  "informatique", "sciences", "mathematiques"} | set(MOTS_CLES_CRITIQUES)

    tests = [
        "j'ai fait bac SM je peux faire quoi",
        "j'ai eu 10/20 de moyenne opac est-ce que je peux faire l'architecture",  # "opac" faute pour rien de connu, doit rester tel quel
        "j'ai eu 10 de moyenne au bas est-ce que je peux faire larchitecture",     # "bas" -> "bac"
        "sc maths quels programmes",
        "univ gamal quels programmes",
        "info a conakry",
    ]
    for t in tests:
        print(f"{t!r}\n  -> {normaliser_et_corriger(t, vocab_test)!r}\n")
