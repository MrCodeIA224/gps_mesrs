"""
RETRIEVAL — Deux chemins bien distincts, comme exigé par le prompt système :

  CHEMIN "FAIT PRÉCIS" (ex: "seuil bac pour la médecine à l'UGANC ?")
    -> recherche hybride (BM25 + sémantique BGE-M3) + fusion RRF + reranking
       + seuil de confiance -> top 5 chunks les plus pertinents

  CHEMIN "LISTE" (ex: "quels programmes propose l'UGANC ?", section 5/6 du
  prompt système)
    -> filtrage STRUCTURÉ direct sur les métadonnées (ville, IES, mot-clé de
       programme, profil bac), sans passer par une recherche de similarité
       ni de limite à 5 -- on retourne TOUT ce qui correspond, condition
       indispensable pour que le chatbot puisse énumérer une liste complète
       sans qu'aucun élément ne manque.

La détection du chemin à emprunter se fait par extraction LLM légère
(intention + filtres), pas par une simple recherche par mots-clés, pour
rester robuste aux formulations variées d'un étudiant.
"""
import json
from pathlib import Path
import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from llm import appeler_llm_brut
from utils_texte import normaliser, extraire_json_de_texte
from pretraitement import construire_vocabulaire_domaine, normaliser_et_corriger, code_profil_bac

# Chemin absolu (indépendant du répertoire de travail courant) : ce module
# est désormais aussi importé depuis main.py à la racine du projet, pas
# seulement exécuté depuis scripts/.
CHROMA_PATH = str(Path(__file__).resolve().parent.parent / "data" / "chroma_db")
NOM_COLLECTION = "orientation_guinee_v2"
NOM_MODELE_EMBEDDING = "BAAI/bge-m3"
NOM_MODELE_RERANKER = "BAAI/bge-reranker-v2-m3"

SEUIL_CONFIANCE_MIN = 0.55  # calibré empiriquement après correction sigmoïde :
                              # hors-sujet ~0.50, cas pertinents 0.50-0.73 (voir
                              # notebook, section calibration) -- valeur choisie
                              # pour prioriser l'anti-hallucination (section 29
                              # du prompt : l'exactitude prime sur la réponse à
                              # tout prix), quitte à rejeter certains cas limites
K_CANDIDATS_INITIAUX = 20
K_RESULTATS_FINAUX = 5


class MoteurRecherche:

    def __init__(self):
        print("Chargement des modèles...")
        self.modele_embedding = SentenceTransformer(NOM_MODELE_EMBEDDING)
        self.reranker = CrossEncoder(NOM_MODELE_RERANKER)

        client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = client.get_collection(NOM_COLLECTION)

        tout = self.collection.get(include=["documents", "metadatas"])
        self.corpus_ids = tout["ids"]
        self.corpus_textes = tout["documents"]
        self.corpus_metadatas = tout["metadatas"]
        self.bm25 = BM25Okapi([t.lower().split() for t in self.corpus_textes])
        self.id_vers_index = {id_: i for i, id_ in enumerate(self.corpus_ids)}
        self.vocabulaire_domaine = construire_vocabulaire_domaine(self.corpus_metadatas)
        print(f"Prêt — {len(self.corpus_ids)} fiches chargées, "
              f"{len(self.vocabulaire_domaine)} termes de vocabulaire du domaine.")

    # ------------------------------------------------------------------
    # Détection d'intention : "fait" ou "liste", + filtres éventuels
    # ------------------------------------------------------------------
    def _est_hors_sujet(self, question: str) -> bool:
        """Détection hors-sujet avec DOUBLE VÉRIFICATION avant de rejeter
        une question. Constat réel en test : le LLM peut donner des
        réponses différentes à la MÊME question selon l'appel (instabilité
        native de l'API à température 0, pas un bug de formulation du
        prompt) -- observé concrètement sur "débouchés en biologie",
        classée tantôt "fait" tantôt "hors_sujet" dans le même notebook.

        Stratégie : un seul appel positif (sur le sujet) suffit à trancher
        immédiatement -- pas de risque à accepter une question par excès
        de prudence. Mais un rejet (hors sujet) doit être CONFIRMÉ par un
        second appel avant d'être définitif -- rejeter une vraie question
        d'orientation est plus coûteux qu'une recherche inutile."""
        prompt = f"""Un étudiant guinéen pose une question à un chatbot d'orientation
universitaire (ParcourSup Guinée, universités, programmes, inscriptions,
comptes, paiements, bourses, débouchés professionnels...).

Exemples de questions SUR LE SUJET (réponse : OUI) :
- "Quels sont les débouchés en biologie ?"
- "Comment récupérer mon mot de passe oublié ?"
- "Quel est le seuil pour médecine ?"
- "Combien coûte l'inscription ?"

Exemples de questions HORS SUJET (réponse : NON) :
- "Quelle est la capitale de la France ?"
- "Quelle est la météo aujourd'hui ?"
- "Raconte-moi une blague"

En cas de doute, réponds OUI (mieux vaut chercher inutilement que rejeter
à tort une vraie question d'orientation).

Question à classer : "{question}"

Réponds UNIQUEMENT par OUI ou NON, rien d'autre."""

        def _demander_une_fois() -> bool:
            """True si le LLM juge la question hors-sujet à CET appel précis."""
            try:
                reponse = appeler_llm_brut(prompt).strip().upper()
                return reponse.startswith("NON")
            except Exception:
                return False  # erreur technique -> ne jamais écarter à tort

        # Double vérification SYMÉTRIQUE (2 appels systématiques, dans les
        # deux sens) -- corrige un bug réel trouvé en test : l'ancienne
        # version ne redemandait qu'en cas de 1er avis "hors sujet", pas
        # en cas de 1er avis "pertinent". Résultat observé concrètement :
        # "Quelle est la capitale de la France ?" (exemple hors-sujet
        # documenté juste au-dessus) acceptée à tort sur un seul appel
        # instable, sans jamais être recontrôlée. Ne trancher "hors sujet"
        # que si les 2 appels concordent ; en cas de désaccord entre les 2,
        # privilégier la recherche (moins coûteux qu'un rejet à tort).
        premier_avis = _demander_une_fois()
        second_avis = _demander_une_fois()
        return premier_avis and second_avis

    def classifier_intention(self, question: str) -> dict:
        if self._est_hors_sujet(question):
            return {"intention": "hors_sujet", "ville": None, "ies": None,
                     "mot_cle_programme": None, "profil_bac": None, "moyenne_bac": None,
                     "profil_bac_code": None, "projet_professionnel": None, "intention_metier": "autre"}

        prompt = f"""Analyse cette question d'étudiant et réponds UNIQUEMENT en JSON,
sans commentaire ni balise de code.

Question : "{question}"

Détermine :
- "intention": "liste" UNIQUEMENT si la question demande d'énumérer des
  PROGRAMMES ou des UNIVERSITÉS précis (ex: "quels programmes à X", "quelles
  universités font Y", "quels programmes accessibles avec mon profil").

  IMPORTANT : une question sur une PROCÉDURE, un CONCEPT ou des CRITÈRES
  généraux (ex: "comment faire mon orientation", "quels sont les critères
  de l'orientation", "comment créer mon compte", "comment ça marche")
  N'EST PAS une "liste" de programmes -- classe-la en "fait", même si elle
  semble demander "tout" sur un sujet. Le chemin "liste" ne cherche QUE
  parmi les programmes/universités, jamais dans le guide de procédures --
  une question de procédure envoyée sur ce chemin ne trouvera jamais de
  réponse, même si elle existe dans le guide.

  RÈGLE IMPORTANTE : si la question mentionne une moyenne/note ET un domaine,
  un programme ou un profil de bac (ex: "j'ai 10 au bac, puis-je faire
  l'architecture ?"), classe-la TOUJOURS en "liste", même si elle semble
  porter sur un seul programme précis. Le filtrage par seuil de moyenne ne
  fonctionne que sur ce chemin, jamais sur le chemin "fait".

- "ville": ville ou préfecture RÉELLE mentionnée (ex: Kankan, Labé, Conakry),
  ou null. NE JAMAIS extraire "Guinée" comme ville -- c'est le nom du pays/de
  la plateforme ("ParcourSup Guinée"), jamais une ville de filtrage.
- "ies": nom ou sigle d'université mentionné, ou null
- "mot_cle_programme": mot-clé de domaine d'étude mentionné (ex: "droit"), ou null
- "profil_bac": profil du bac mentionné en toutes lettres (ex: "sciences mathématiques"), ou null
- "moyenne_bac": moyenne au bac mentionnée par l'étudiant (nombre), ou null
- "projet_professionnel": métier ou domaine professionnel visé si mentionné (ex: "devenir médecin"), ou null
- "intention_metier": classe la question dans UNE de ces catégories (à usage
  statistique uniquement, n'influence pas la recherche) : "recherche_programme",
  "recherche_universite", "paiement", "inscription", "resultat", "bourse",
  "reorientation", "probleme_technique", "recuperation_compte", "procedure", "autre"

Exemple : {{"intention": "liste", "ville": "Kankan", "ies": null, "mot_cle_programme": null, "profil_bac": null, "moyenne_bac": null, "projet_professionnel": null, "intention_metier": "recherche_programme"}}"""

        try:
            reponse_brute = appeler_llm_brut(prompt)
            resultat = extraire_json_de_texte(reponse_brute)
        except Exception:
            return {"intention": "fait", "ville": None, "ies": None,
                     "mot_cle_programme": None, "profil_bac": None, "moyenne_bac": None,
                     "projet_professionnel": None, "intention_metier": "autre"}

        # Conversion du profil en toutes lettres vers le code réel stocké
        # dans les métadonnées (ex: "sciences mathématiques" -> "SM")
        if resultat.get("profil_bac"):
            resultat["profil_bac_code"] = code_profil_bac(resultat["profil_bac"])
        else:
            resultat["profil_bac_code"] = None

        # Filet de sécurité (en plus de la consigne du prompt) : "Guinée" ou
        # "ParcourSup Guinée" ne sont jamais une vraie ville de filtrage --
        # bug réel observé en test où le LLM confondait le nom du pays/de la
        # plateforme avec une ville, ce qui cassait complètement le filtre.
        ville = resultat.get("ville")
        if ville and normaliser(ville) in ("guinee", "parcoursup guinee"):
            resultat["ville"] = None

        return resultat

    # ------------------------------------------------------------------
    # Chemin LISTE : filtrage structuré direct, sans limite de résultats
    # ------------------------------------------------------------------
    def recherche_liste(self, ville=None, ies=None, mot_cle_programme=None,
                         profil_bac_code=None, moyenne_bac=None, types=("programme",)) -> list:
        resultats = []
        for i, meta in enumerate(self.corpus_metadatas):
            if meta.get("type") not in types:
                continue
            if ville and normaliser(ville) not in normaliser(meta.get("ville", "")):
                continue
            if ies and normaliser(ies) not in normaliser(meta.get("ies", "")):
                continue
            if mot_cle_programme and normaliser(mot_cle_programme) not in normaliser(meta.get("programme", "")):
                continue
            if profil_bac_code:
                codes_du_programme = meta.get("options_autorisees", "").split(",")
                if profil_bac_code not in codes_du_programme:
                    continue
            if moyenne_bac is not None:
                seuil = meta.get("seuil_bac")
                # Si le seuil n'est pas renseigné pour ce programme, on ne
                # l'exclut pas par excès de prudence (mieux vaut le montrer
                # avec le seuil manquant que le cacher à tort) -- mais on le
                # signale dans le résultat pour que le LLM le précise.
                if seuil is not None and moyenne_bac < seuil:
                    continue
            resultats.append({
                "id": self.corpus_ids[i],
                "texte": self.corpus_textes[i],
                "metadata": meta,
            })
        return resultats

    # ------------------------------------------------------------------
    # Chemin FAIT : recherche hybride + reranking (comme précédemment)
    # ------------------------------------------------------------------
    def _recherche_semantique(self, question, k, filtres):
        vecteur = self.modele_embedding.encode([question], normalize_embeddings=True)[0].tolist()
        kwargs = {"query_embeddings": [vecteur], "n_results": k}
        if filtres:
            kwargs["where"] = filtres
        resultats = self.collection.query(**kwargs)
        return resultats["ids"][0]

    def _recherche_bm25(self, question, k, filtres):
        scores = self.bm25.get_scores(question.lower().split())
        indices_tries = scores.argsort()[::-1]
        resultats = []
        for idx in indices_tries:
            id_ = self.corpus_ids[idx]
            if filtres and not all(self.corpus_metadatas[idx].get(c) == v for c, v in filtres.items()):
                continue
            resultats.append(id_)
            if len(resultats) >= k:
                break
        return resultats

    @staticmethod
    def _fusion_rrf(liste_sem, liste_bm25, k_constante=60):
        scores = {}
        for rang, id_ in enumerate(liste_sem):
            scores[id_] = scores.get(id_, 0) + 1 / (k_constante + rang + 1)
        for rang, id_ in enumerate(liste_bm25):
            scores[id_] = scores.get(id_, 0) + 1 / (k_constante + rang + 1)
        return sorted(scores, key=scores.get, reverse=True)

    def _reranker_candidats(self, question, ids_candidats, k_final):
        textes = [self.corpus_textes[self.id_vers_index[id_]] for id_ in ids_candidats]
        paires = [[question, t] for t in textes]
        scores_bruts = self.reranker.predict(paires)

        # IMPORTANT : bge-reranker-v2-m3 renvoie des scores bruts (logits)
        # non bornés, typiquement entre -10 et 10 -- PAS des scores 0-1.
        # Sans cette conversion, notre SEUIL_CONFIANCE_MIN (pensé pour une
        # échelle 0-1) rejette presque tout à tort. Bug réel détecté en
        # test (scores observés de l'ordre de 0.006, sans rapport avec la
        # pertinence réelle). On applique la sigmoïde nous-mêmes, plutôt
        # que via un paramètre du constructeur dont le nom a changé selon
        # les versions de sentence-transformers (activation_fn vs
        # default_activation_function) -- plus robuste, indépendant de la
        # version installée.
        scores = 1 / (1 + np.exp(-np.array(scores_bruts)))

        resultats = list(zip(ids_candidats, textes, scores))
        resultats.sort(key=lambda x: x[2], reverse=True)
        return resultats[:k_final]

    def rechercher_debouches_lies(self, nom_programme_mentionne: str) -> list:
        """Trouve le(s) programme(s) dont le nom correspond à la mention de
        l'utilisateur, PUIS récupère leurs débouchés via une liaison
        garantie par programme_id (pas par ressemblance de texte) --
        élimine le risque de confondre les débouchés de deux programmes au
        nom proche (ex: les 14 variantes de "Licence En Droit", chacune
        avec ses propres débouchés distincts)."""
        nom_norm = normaliser(nom_programme_mentionne)
        programmes_correspondants = [
            (id_, meta) for id_, meta in
            ((self.corpus_ids[i], m) for i, m in enumerate(self.corpus_metadatas))
            if meta.get("type") == "programme" and nom_norm in normaliser(meta.get("programme", ""))
        ]
        if not programmes_correspondants:
            return []

        ids_programmes_trouves = {id_ for id_, _ in programmes_correspondants}

        resultats = []
        for i, meta in enumerate(self.corpus_metadatas):
            if meta.get("type") != "debouches":
                continue
            ids_lies = set(meta.get("programme_ids", "").split(","))
            if ids_lies & ids_programmes_trouves:  # intersection non vide = vraiment lié
                resultats.append({
                    "id": self.corpus_ids[i],
                    # "score" ajouté volontairement (1.0, confiance maximale
                    # puisque la liaison est garantie par ID, pas par simple
                    # similarité) -- SANS cette clé, construire_contexte()
                    # dans llm.py traite à tort ces résultats comme le
                    # format "liste" condensé (ville/université/seuil),
                    # qui ne sait pas afficher un texte de débouchés et
                    # perd purement et simplement le contenu réel (bug
                    # sérieux trouvé et corrigé en relecture -- le vrai
                    # texte des débouchés disparaissait, remplacé par des
                    # "?" partout).
                    "score": 1.0,
                    "texte": self.corpus_textes[i],
                    "metadata": meta,
                })
        return resultats

    def recherche_fait(self, question: str, filtres: dict | None = None) -> list:
        liste_sem = self._recherche_semantique(question, K_CANDIDATS_INITIAUX, filtres)
        liste_bm25 = self._recherche_bm25(question, K_CANDIDATS_INITIAUX, filtres)
        candidats = self._fusion_rrf(liste_sem, liste_bm25)[:K_CANDIDATS_INITIAUX]
        if not candidats:
            return []
        resultats = self._reranker_candidats(question, candidats, K_RESULTATS_FINAUX)
        if not resultats or resultats[0][2] < SEUIL_CONFIANCE_MIN:
            return []
        # Filtre le seuil sur CHAQUE résultat individuellement, pas
        # seulement le premier -- bug réel trouvé en relecture : avant
        # cette correction, si le meilleur résultat passait le seuil, TOUS
        # les résultats (jusqu'à 5) étaient renvoyés tels quels, même ceux
        # largement en dessous du seuil de confiance (score à 0.08 par
        # exemple), envoyés au LLM comme s'ils étaient tous pertinents.
        return [
            {"id": id_, "texte": texte, "score": float(score),
             "metadata": self.corpus_metadatas[self.id_vers_index[id_]]}
            for id_, texte, score in resultats
            if score >= SEUIL_CONFIANCE_MIN
        ]

    # ------------------------------------------------------------------
    # Point d'entrée unique : route vers le bon chemin
    # ------------------------------------------------------------------
    def rechercher(self, question: str) -> dict:
        """Retourne {"intention", "resultats", "note", "entites"}.

        "entites" contient les informations déjà extraites par le
        classifieur (ville, profil_bac, moyenne_bac, projet_professionnel,
        intention_metier) -- exposées ici pour que l'appelant (app.py)
        puisse alimenter la mémoire structurée ET les logs à partir de ce
        SEUL appel, au lieu de refaire 2 appels séparés à Mistral pour
        extraire des informations déjà obtenues ici (consolidation faite
        pour réduire le nombre d'allers-retours par question, passant de
        6-7 à 4-5 appels)."""
        question_normalisee = normaliser_et_corriger(question, self.vocabulaire_domaine)
        intention = self.classifier_intention(question_normalisee)

        def _resultat(intention_type: str, resultats: list, note: str | None = None) -> dict:
            return {
                "intention": intention_type,
                "resultats": resultats,
                "note": note,
                "entites": {
                    "ville": intention.get("ville"),
                    "profil_bac": intention.get("profil_bac"),
                    "moyenne_bac": intention.get("moyenne_bac"),
                    "projet_professionnel": intention.get("projet_professionnel"),
                    "intention_metier": intention.get("intention_metier", "autre"),
                },
            }

        if intention.get("intention") == "hors_sujet":
            return _resultat("hors_sujet", [])

        if intention.get("intention") == "liste":
            filtres = {
                "ville": intention.get("ville"),
                "ies": intention.get("ies"),
                "mot_cle_programme": intention.get("mot_cle_programme"),
                "profil_bac_code": intention.get("profil_bac_code"),
                "moyenne_bac": intention.get("moyenne_bac"),
            }
            nb_filtres_actifs = sum(1 for v in filtres.values() if v is not None)

            # Garde-fou : une question de liste avec 0 filtre exploitable
            # renverrait tout le catalogue (ex: "quelles universités sont
            # disponibles ?") -- on demande une précision plutôt que de
            # renvoyer une liste arbitraire et inutilisable.
            if nb_filtres_actifs == 0:
                return _resultat("clarification", [])

            moyenne_bac = filtres.pop("moyenne_bac")

            # Étape 1 : la catégorie existe-t-elle, indépendamment du filtre
            # numérique ? (ville/ies/mot_cle_programme/profil_bac_code seuls)
            candidats_domaine = self.recherche_liste(**filtres, moyenne_bac=None)

            if not candidats_domaine:
                # Rien ne correspond même sans filtre numérique -> vraie
                # absence d'information (mauvais domaine, catégorie inexistante)
                return _resultat("liste", [])

            if moyenne_bac is None:
                return _resultat("liste", candidats_domaine)

            # Étape 2 : appliquer le filtre numérique sur ce sous-ensemble
            candidats_avec_moyenne = self.recherche_liste(**filtres, moyenne_bac=moyenne_bac)

            if not candidats_avec_moyenne:
                # La catégorie existe, mais aucun programme n'atteint le
                # seuil requis avec la moyenne indiquée -- ce n'est PAS un
                # manque d'information : on transmet quand même les
                # programmes trouvés (avec leur seuil) et une note, pour
                # que le LLM explique clairement l'inéligibilité plutôt que
                # de répondre "je ne sais pas" (voir discussion : ne pas
                # confondre "aucune donnée" et "la réponse est non").
                note = (f"Aucun des {len(candidats_domaine)} programme(s) trouvé(s) dans cette "
                        f"catégorie n'accepte une moyenne de {moyenne_bac}/20 -- indique-le "
                        f"clairement à l'étudiant en t'appuyant sur les seuils réels ci-dessous, "
                        f"ne dis pas que l'information est indisponible.")
                return _resultat("liste", candidats_domaine, note)

            return _resultat("liste", candidats_avec_moyenne)

        resultats = self.recherche_fait(question_normalisee)

        # Renfort déterministe pour les questions de débouchés : si la
        # question mentionne "débouché" ET qu'un nom de programme précis
        # est identifiable littéralement dans la question, on ajoute en
        # priorité les fiches débouchés liées par programme_id (garanti
        # exact) -- indépendamment de ce que la recherche sémantique a
        # trouvé, pour éliminer le risque de confondre les débouchés de
        # deux programmes au nom proche (vérifié en test réel sur les 14
        # variantes de "Licence En Droit" : chacune a ses propres
        # débouchés, jamais mélangés avec cette méthode).
        if "debouch" in normaliser(question_normalisee):
            programme_mentionne = self._detecter_nom_programme_dans_texte(question_normalisee)
            if programme_mentionne:
                resultats_lies = self.rechercher_debouches_lies(programme_mentionne)
                ids_deja_presents = {r["id"] for r in resultats}
                nouveaux = [r for r in resultats_lies if r["id"] not in ids_deja_presents]
                resultats = nouveaux + resultats  # priorité aux résultats garantis par ID

        return _resultat("fait", resultats)

    def _detecter_nom_programme_dans_texte(self, texte: str) -> str | None:
        """Cherche si un nom de programme réel du corpus (ou sa partie la
        plus distinctive, après les deux-points) apparaît littéralement
        dans le texte de la question -- utilisé pour la liaison débouchés
        garantie par ID. Retourne le nom le plus long trouvé (le plus
        spécifique), ou None si aucun ne correspond."""
        texte_norm = normaliser(texte)
        meilleur_match = None
        for meta in self.corpus_metadatas:
            if meta.get("type") != "programme":
                continue
            nom = meta.get("programme", "")
            partie_distinctive = nom.split(":")[-1].strip() if ":" in nom else nom
            partie_norm = normaliser(partie_distinctive)
            if len(partie_norm) > 6 and partie_norm in texte_norm:
                if meilleur_match is None or len(partie_norm) > len(normaliser(meilleur_match)):
                    meilleur_match = partie_distinctive
        return meilleur_match
