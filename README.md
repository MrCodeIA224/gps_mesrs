# GPS-MESRS

***Guide vers les Programmes et Spécialités du MESRS***

Assistant conversationnel basé sur une architecture **RAG (Retrieval-Augmented Generation)** conçu pour aider les bacheliers et étudiants guinéens à s'orienter dans leurs démarches administratives, critères d'orientation et choix de filières universitaires — à partir des guides officiels du **Ministère de l'Enseignement Supérieur et de la Recherche Scientifique (MESRS)** de la République de Guinée, sur la plateforme ParcourSup Guinée.

> Projet académique réalisé dans le cadre du Master 1 Intelligence Artificielle — Dakar Institute of Technology (DIT). S'appuie sur des données officielles du MESRS mais n'est **pas affilié** au Ministère.

---

## Sommaire

- [À propos du projet](#à-propos-du-projet)
- [Fonctionnalités](#fonctionnalités)
- [Comment ça marche](#comment-ça-marche)
- [Architecture du projet](#architecture-du-projet)
- [Stack technique](#stack-technique)
- [Installation et démarrage](#installation-et-démarrage)
- [Évaluation](#évaluation)
- [Limites connues](#limites-connues-non-corrigées-par-choix)
- [Points à surveiller](#points-à-surveiller)
- [Équipe](#équipe)
- [Contribuer](#contribuer)

---

## À propos du projet

Chaque année, de nombreux bacheliers guinéens manquent d'informations claires et centralisées pour choisir leur filière universitaire : quelles options sont autorisées selon leur série de bac, quels débouchés existent pour chaque programme, quelles démarches suivre pour s'inscrire sur ParcourSup Guinée...

**GPS-MESRS** répond à ces questions en langage naturel, en s'appuyant exclusivement sur les guides officiels du Ministère — pour donner des réponses fiables, sourcées et à jour, sans halluciner d'informations.

## Fonctionnalités

- 💬 Chat conversationnel en langage naturel, avec mémoire de la conversation
- 🎓 Réponses basées sur les guides officiels du MESRS (programmes, séries autorisées, seuils, débouchés, procédures d'orientation)
- 🔍 Recherche hybride (sémantique + mots-clés) dans la base de connaissances, avec reranking
- 🛡️ Masquage des données sensibles et dispositif anti-hallucination à plusieurs niveaux
- 📄 Export de la conversation en PDF et système d'évaluation de satisfaction
- 🖥️ Interface Streamlit simple et intuitive

## Comment ça marche

### Les données (4 sources → 665 fiches)

Le guide d'orientation officiel (PDF) et les tableaux de programmes/débouchés/établissements sont transformés en un corpus unique et homogène :

| Source | Fiches |
| --- | --- |
| Guide d'orientation (procédures) | 67 |
| Programmes | 383 |
| Débouchés | 197 |
| Établissements | 18 |

Voir `scripts/1_decouper_guide.py` à `scripts/4_indexer_chroma.py` pour le détail du pipeline d'ingestion.

### Retrieval à double chemin

- **"Fait"** — pour les questions précises (ex : *"seuil bac pour la médecine à l'UGANC ?"*) : recherche hybride BM25 + embeddings (BGE-M3), fusion RRF, puis reranking (BGE-reranker-v2-m3) avec seuil de confiance.
- **"Liste"** — pour les questions qui demandent d'énumérer des programmes/universités (ex : *"quels programmes à Kankan ?"*) : filtrage structuré sur les métadonnées (ville, établissement, domaine, profil de bac, moyenne), sans limite de résultats — une liste doit être complète.
- **"Hors sujet"** et **"clarification"** : court-circuits dédiés, avec double vérification pour le hors-sujet (une question jugée pertinente une seule fois suffit à l'accepter ; un rejet doit être confirmé deux fois avant d'être définitif).

Une question de procédure générale (*"comment faire mon orientation ?"*) est explicitement traitée comme un "fait", jamais comme une "liste" — le chemin liste ne cherche que parmi les programmes/universités et ne trouverait jamais la procédure.

### Compréhension du langage naturel

`scripts/pretraitement.py` combine un dictionnaire d'abréviations (SM, sc maths, univ...), un vocabulaire du domaine extrait automatiquement du corpus, et une correction orthographique floue protégée par un dictionnaire français — pour absorber fautes de frappe et raccourcis étudiants sans corrompre de vrais mots français.

### Mémoire et dialogue naturel

Mémoire structurée par "slots" (ville, profil de bac, moyenne, projet professionnel), persistante toute la session, combinée à une fenêtre glissante des derniers échanges transmise au LLM à chaque réponse — pour un dialogue cohérent d'un message à l'autre.

### Sécurité

- Masquage des données sensibles (codes Orange Money, mots de passe, codes SMS, INE) **avant** l'appel au LLM, avec remasquage en défense en profondeur avant écriture des logs.
- Défense anti-prompt-injection : tout ce qui vient du contexte documentaire ou du message utilisateur est traité comme une donnée à analyser, jamais comme une instruction.

### Anti-hallucination

Dispositif à plusieurs niveaux, pas seulement une consigne dans le prompt :

1. **Prompt système** (`scripts/llm.py`, 37 sections) — interdit explicitement toute invention (programme, débouché, seuil, numéro de téléphone...), avec une règle de vérification systématique avant d'écrire un fait précis, et des exemples concrets tirés de cas réellement observés en test plutôt que des consignes abstraites.
2. **Refus programmatique** — si le retrieval ne trouve rien, le LLM n'est même pas appelé ; un message fixe est renvoyé directement.
3. **Vérification post-génération, indépendante du prompt** — la réponse générée est relue automatiquement après coup pour repérer deux catégories de contenu vérifiables par comparaison littérale avec le contexte fourni :
   - montants et numéros de téléphone ;
   - noms de programmes/diplômes (« Licence en ... », « Doctorat en ... »).

   Tout élément absent du contexte fourni pour cette question précise déclenche un avertissement visible dans la réponse, sans bloquer la conversation. Ce filet ne dépend pas de l'obéissance du LLM aux instructions — c'est une vérification en code, sur le texte réellement généré.

### Observabilité

Logs anonymisés (`data/logs_conversations.jsonl`) : intention métier détectée à des fins statistiques uniquement (n'influence jamais la recherche ni la génération), taux de questions sans réponse, répartition géographique des demandes.

## Architecture du projet

```text
gps_mesrs/
├── data/                        # Données : JSON bruts, corpus fusionné, index Chroma, logs
│   ├── raw/                     # Sources brutes (guide, programmes, débouchés, établissements)
│   ├── processed/               # Corpus fusionné + résultats d'évaluation
│   └── jeu_de_test_annote.json  # Jeu de test pour les évaluations (voir Évaluation)
├── scripts/                     # Cœur RAG
│   ├── 1_decouper_guide.py      # Ingestion : découpage du guide en fiches
│   ├── 2_corriger_referentiel.py
│   ├── 3_fusionner_corpus.py
│   ├── 4_indexer_chroma.py      # Construction de l'index vectoriel
│   ├── retrieval.py             # Retrieval hybride à double chemin
│   ├── pretraitement.py         # Normalisation / correction du langage naturel
│   ├── memoire.py               # Mémoire conversationnelle + slots structurés
│   ├── securite.py              # Masquage des données sensibles
│   ├── salutations.py           # Court-circuits politesse
│   ├── logs.py                  # Journalisation anonymisée
│   ├── llm.py                   # Prompt système, appel Mistral, anti-hallucination
│   ├── evaluer_precision_recall.py
│   └── evaluer_ragas.py
├── fonctions.py                 # Orchestration du pipeline de scripts/ pour l'interface
├── main.py                      # Point d'entrée de l'application — interface Streamlit
└── README.md                    # Ce fichier
```

`main.py` est l'unique point d'entrée (`streamlit run main.py`) : il gère l'interface (bouton de chat flottant, export PDF, évaluation de satisfaction) et délègue toute la logique métier à `fonctions.py`, qui orchestre à son tour le pipeline RAG de `scripts/`.

## Stack technique

| Composant | Choix |
| --- | --- |
| Langage | Python |
| Frontend | Streamlit |
| Embeddings | BAAI/bge-m3 |
| Base vectorielle | Chroma |
| Reranker | BAAI/bge-reranker-v2-m3 |
| LLM | Mistral (`mistral-large-latest`, alias stable) |
| Données | JSON structuré, issu des guides officiels du MESRS |

## Installation et démarrage

```bash
# 1. Cloner le dépôt
git clone https://github.com/MrCodeIA224/gps_mesrs.git
cd gps_mesrs

# 2. Créer un environnement virtuel (recommandé)
python3 -m venv .venv
source .venv/bin/activate   # sous Linux/Mac

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer la clé Mistral
cp .env.example .env
# puis ouvrir .env et coller ta clé (gratuite sur https://console.mistral.ai)

# 5. Corriger un bug d'import connu (ragas 0.3.9 + version récente de langchain-community)
python scripts/patch_ragas.py   # à faire une seule fois

# 6. Construire le corpus et l'index vectoriel (une seule fois, ou après
#    modification des données sources)
cd scripts
python 1_decouper_guide.py
python 2_corriger_referentiel.py
python 3_fusionner_corpus.py
python 4_indexer_chroma.py      # télécharge BGE-M3 + reranker (~3 Go), 1ère fois seulement
cd ..

# 7. Lancer l'application
streamlit run main.py
```

L'application s'ouvre automatiquement dans le navigateur (généralement sur `http://localhost:8501`).

> Si des messages `ModuleNotFoundError: No module named 'torchvision'` apparaissent au lancement : bruit inoffensif (Streamlit inspecte des modules de vision par ordinateur de la bibliothèque `transformers`, jamais utilisés ici). Pour les masquer : `streamlit run main.py --server.fileWatcherType none`.

## Évaluation

Le dispositif de mesure repose sur un **jeu de test unique et partagé**, `data/jeu_de_test_annote.json` (20 scénarios), utilisé par les deux scripts d'évaluation ci-dessous — pour éviter que les deux dérivent avec des questions différentes au fil du temps.

```bash
cd scripts
python evaluer_precision_recall.py   # retrieval "fait" + "liste" + classification hors-sujet/clarification
python evaluer_ragas.py              # qualité de la réponse générée (chemin "fait" uniquement)
```

- **`evaluer_precision_recall.py`** mesure trois choses distinctes : le recall du retrieval sur le chemin "fait" (la bonne fiche est-elle dans le top-5 ?), le recall sur le chemin "liste" (toutes les fiches attendues sont-elles retournées, sans limite ?) **plus une vérification anti-hallucination automatique sur les réponses générées à partir de ces listes** (angle mort corrigé : ce chemin n'était auparavant couvert par aucune mesure automatique), et la conformité de la classification hors-sujet/clarification.
- **`evaluer_ragas.py`** mesure la fidélité de la réponse générée au contexte (Faithfulness), sa pertinence (Answer Relevancy), et la qualité du retrieval (Context Precision/Recall) via un LLM-juge (Mistral), sur les questions de type "fait" du jeu de test partagé.

> **À noter pour l'équipe :** le jeu de test de 20 scénarios ci-dessus a été reconstitué à partir de vraies fiches du corpus (le fichier `data/jeu_de_test_annote.json` avait été perdu / jamais commité). Il n'a pas encore été exécuté sur un index construit — les scores obtenus lors d'un tout premier essai avec un jeu de 5 questions (Faithfulness ~0.89, Answer Relevancy ~0.93, Context Precision/Recall ~1.0) restent à re-valider avec ce jeu plus large avant de les considérer comme représentatifs. Lancez les deux scripts ci-dessus après avoir construit l'index et complétez cette section avec les résultats obtenus.

## Limites connues, non corrigées par choix

- Duplication de la logique d'orchestration entre `fonctions.py` (appelé par `main.py`) et `evaluer_precision_recall.py`.
- La vérification anti-hallucination post-génération (montants, numéros, noms de programme) est un filet de sécurité heuristique, pas une preuve formelle : elle réduit le risque sans le supprimer entièrement, et ne couvre pas tout type de fait (ex. une règle de procédure reformulée de façon incorrecte sans chiffre ni nom propre ne serait pas détectée).
- Dashboard de monitoring visuel, interface en cartes structurées, déploiement public : repoussés en fin de projet.

## Points à surveiller

- Le premier appel réel à l'API après un changement de clé/fichier `.env` nécessite un redémarrage complet de l'application (la clé et le code ne sont chargés qu'une seule fois au démarrage).
- Les quotas gratuits des fournisseurs de LLM évoluent régulièrement — se fier au comportement observé plutôt qu'à un chiffre fixe.

## Équipe

Projet réalisé à 3, dans le cadre du Master 1 IA — DIT :

| Membre | Rôle |
| --------------------| -------------------------------------------------------------------------------- |
| Mody Amadou DIALLO  | Données (extraction, nettoyage) + Cœur RAG (embeddings, retrieval, LLM, prompts) |
| Azizatou BALDE      | Données (extraction, nettoyage) + Cœur RAG (embeddings, retrieval, LLM, prompts) |
| Mamadou Tahirou BAH | Données (extraction, nettoyage, chunking) + Interface & évaluation               |

## Contribuer

Pour éviter de se marcher dessus et garder un historique propre, on suit une organisation simple à base de **branches par tâche** et de **Pull Requests (PR)**.

### Règle d'or

> **On ne travaille jamais directement sur `main`.** La branche `main` doit toujours rester stable et fonctionnelle.

### 1. Créer une branche pour chaque tâche

Avant de commencer un nouveau bout de travail, on part toujours de `main` à jour et on crée une branche dédiée :

```bash
git checkout main
git pull origin main
git checkout -b type/description-courte
```

**Convention de nommage des branches :**

| Préfixe | Utilisation                      | Exemple                                     |
| ------- | ---------------------------------| ------------------------------------------- |
| `feat/` | Nouvelle fonctionnalité          | `feat/interface-chat`                       |
| `fix/`  | Correction de bug                | `fix/erreur-chunking`                       |
| `data/` | Ajout ou modification de données | `data/ajout-referentiel-programmes`         |
| `docs/` | Documentation                    | `docs/readme-installation`                  |

### 2. Commiter régulièrement, avec des messages clairs

```bash
git add .
git commit -m "feat: ajout de la fonction de recherche par embeddings"
```

Mieux vaut plusieurs petits commits clairs qu'un seul gros commit "divers changements".

### 3. Avant d'ouvrir la Pull Request

Si le changement touche `scripts/retrieval.py`, `scripts/llm.py`, `scripts/pretraitement.py` ou aux données du corpus, relancer les scripts d'évaluation (voir [Évaluation](#évaluation)) et vérifier qu'aucune régression n'apparaît par rapport au dernier résultat connu — il n'y a pas encore de vérification automatique (CI) sur ce dépôt, donc c'est actuellement une étape manuelle.

### 4. Pousser la branche et ouvrir une Pull Request

```bash
git push origin type/description-courte
```

Ensuite, sur GitHub :

1. Ouvrir une **Pull Request** de sa branche vers `main`
2. Décrire brièvement ce que fait la PR (quoi, pourquoi)
3. Demander une relecture à au moins un autre membre de l'équipe avant de merger
4. Une fois validée, **merger la PR**

### 5. Rester à jour

Avant de commencer une nouvelle tâche, toujours mettre à jour sa copie locale :

```bash
git checkout main
git pull origin main
```
