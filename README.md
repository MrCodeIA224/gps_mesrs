# GPS-MESRS

***Guide vers les Programmes et Spécialités du MESRS***

Assistant conversationnel basé sur une architecture **RAG (Retrieval-Augmented Generation)** conçu pour aider les bacheliers et étudiants guinéens à s'orienter dans leurs démarches administratives, critères d'orientation et choix de filières universitaires — à partir des guides officiels du **Ministère de l'Enseignement Supérieur et de la Recherche Scientifique (MESRS)** de la République de Guinée.

> Projet académique réalisé dans le cadre du Master 1 Intelligence Artificielle — Dakar Institute of Technology (DIT). S'appuie sur des données officielles du MESRS mais n'est **pas affilié** au Ministère encore.

---

## À propos du projet

Chaque année, de nombreux bacheliers guinéens manquent d'informations claires et centralisées pour choisir leur filière universitaire : quelles options sont autorisées selon leur option du bac, quels débouchés existent pour chaque programme, quelles démarches suivre pour s'inscrire...

**GPS-MESRS** répond à ces questions en langage naturel, en s'appuyant exclusivement sur les guides officiels du Ministère — pour donner des réponses fiables, sourcées et à jour, sans halluciner d'informations.

## Fonctionnalités

- 💬 Chat conversationnel en langage naturel
- 🎓 Réponses basées sur les guides officiels du MESRS (programmes, options autorisées, débouchés, procédures d'orientation)
- 🔍 Recherche sémantique dans la base de connaissances (embeddings + retrieval)
- 🖥️ Interface simple et intuitive (Streamlit)

## Architecture du projet

```
gps_mesrs/
├── data/           # Données du projet (JSON structurés issus des guides officiels)
├── fonction.py     # Fonctions utilitaires du projet (chunking, embeddings, retrieval, appels LLM...)
├── main.py         # Point d'entrée de l'application — interface Streamlit (frontend)
└── README.md       # Ce fichier
```

*Architecture volontairement simple pour l'instant — elle pourra évoluer (ex. séparation en modules) si le projet grossit.*

## 🚀 Installation

```bash
# Cloner le dépôt
git clone https://github.com/MrCodeIA224/gps_mesrs.git
cd gps_mesrs

# Créer un environnement virtuel (recommandé)
python3 -m venv venv
source venv/bin/activate   # sous Linux/Mac

# Installer les dépendances
pip install -r requirements.txt
```

## ▶️ Utilisation

```bash
streamlit run main.py
```

L'application s'ouvre automatiquement dans le navigateur (généralement sur `http://localhost:8501`).

## 👥 Équipe

Projet réalisé à 3, dans le cadre du Master 1 IA — DIT :

| Membre              | Rôle                                                                             |
| --------------------| -------------------------------------------------------------------------------- |
| Mody Amadou DIALLO  | Données (extraction, nettoyage) + Cœur RAG (embeddings, retrieval, LLM, prompts) |
| Azizatou BALDE      | Données (extraction, nettoyage) + Cœur RAG (embeddings, retrieval, LLM, prompts) |
| Mamadou Tahirou BAH | Données (extraction, nettoyage, chunking) + Interface & évaluation               |
+--------------------------------------------------------------------------------------------------------+

---

## Méthode de travail collaboratif (Git & GitHub)

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

### 3. Pousser la branche et ouvrir une Pull Request

```bash
git pull origin main 
```

Si des changements surviennent, repeter les commandes precedentes (git add et git commit)

```bash
git push origin type/description-courte
```

Ensuite, sur GitHub :

1. Ouvrir une **Pull Request** de sa branche vers `main`
2. Décrire brièvement ce que fait la PR (quoi, pourquoi)
3. Demander une relecture à au moins un autre membre de l'équipe avant de merger
4. Une fois validée, **merger la PR**

### 4. Rester à jour

Avant de commencer une nouvelle tâche, toujours mettre à jour sa copie locale :

```bash
git checkout main
git pull origin main
```

## 🛠️ Stack technique

- **Langage :** Python
- **Frontend :** Streamlit
- **RAG :** embeddings + base vectorielle + LLM (détails à préciser au fil du projet)
- **Données :** JSON structuré, issu des guides officiels du MESRS



### azizatou 

# Chatbot ParcourSup Guinée

Assistant conversationnel RAG (Retrieval-Augmented Generation) pour
l'orientation universitaire des bacheliers guinéens sur ParcourSup Guinée.

## Stack technique

| Composant | Choix |
|---|---|
| Embeddings | BAAI/bge-m3 |
| Base vectorielle | Chroma |
| Reranker | BAAI/bge-reranker-v2-m3 |
| LLM | Mistral (`mistral-large-latest`, alias stable) |
| Interface | Streamlit |

## Ce que couvre ce projet

### Ingestion et données (4 sources -> 665 fiches)
Guide d'orientation (67 fiches), programmes (383), débouchés (197),
établissements (18). Voir `1_decouper_guide.py` à `4_indexer_chroma.py`.

### Retrieval à double chemin
- **"Fait"** : recherche hybride (BM25 + BGE-M3) + reranking, pour les
  questions précises.
- **"Liste"** : filtrage structuré sur les métadonnées (ville, IES,
  domaine, profil, moyenne), réservé aux questions qui demandent
  d'énumérer des programmes/universités -- une question de procédure
  générale ("comment faire mon orientation") est explicitement exclue de
  ce chemin et redirigée vers "fait" (bug réel corrigé : ces questions
  tombaient dans "liste" et ne trouvaient jamais rien, le chemin liste ne
  cherchant que dans les fiches de type "programme").
- **"Hors sujet"** et **"clarification"** : court-circuits dédiés.
- Filtre numérique moyenne/seuil, avec distinction "aucune info" vs
  "inéligible" (le chatbot explique une inéligibilité plutôt que de dire
  "je ne sais pas").
- Filet de sécurité : "Guinée" (nom du pays/de la plateforme) ne peut
  jamais être interprété comme un filtre de ville.

### Compréhension du langage naturel (`pretraitement.py`)
Dictionnaire d'abréviations, vocabulaire du domaine extrait
automatiquement, correction floue protégée par un dictionnaire français
(`pyspellchecker`), liste d'exceptions ciblées (ex: "bas" -> "bac").

### Mémoire et dialogue naturel
M�moire structurée par slots (ville, profil, moyenne, projet
professionnel), persistante toute la session. Les 2 derniers échanges de
la conversation sont transmis au LLM à chaque réponse (tours
user/assistant natifs), pour un dialogue cohérent d'un message à l'autre.

### Sécurité
- Masquage des données sensibles **avant l'appel au LLM**, avec
  remasquage en défense en profondeur avant écriture des logs.
- Motifs de détection couvrant : codes Orange Money, mots de passe,
  codes SMS, et l'INE (Identifiant National Étudiant).
- Défense anti-prompt-injection.

### Anti-hallucination
Le prompt système (36 sections) interdit explicitement toute invention
d'information : programmes ou écoles non confirmés par le corpus,
numéros de téléphone non présents littéralement dans le contexte fourni,
villes de filtrage incorrectes. Chaque règle est accompagnée d'exemples
concrets (plus efficaces qu'une consigne abstraite seule) tirés de cas
réellement observés en test, pour guider le LLM vers un refus honnête
("je ne sais pas") plutôt qu'une réponse plausible mais inventée.

### Observabilité
Logs anonymisés (`data/logs_conversations.jsonl`), intention métier
détectée à des fins statistiques uniquement -- n'influence jamais la
recherche ni la génération.

### Évaluation — résultats obtenus
| Métrique | Score |
|---|---|
| Faithfulness | 0.889 |
| Answer Relevancy | 0.928 |
| Context Precision | 1.000 |
| Context Recall | 1.000 |

## Démarrage

```bash
pip install -r requirements.txt
cp .env.example .env   # puis coller ta clé Mistral (console.mistral.ai)

python patch_ragas.py  # obligatoire une fois (bug d'import connu dans
                        # ragas 0.3.9 combiné à une version récente de
                        # langchain-community)

cd scripts
python 1_decouper_guide.py
python 2_corriger_referentiel.py
python 3_fusionner_corpus.py
python 4_indexer_chroma.py   # télécharge BGE-M3 + reranker (~3 Go), 1ère fois seulement

streamlit run app.py
```

Si des messages `ModuleNotFoundError: No module named 'torchvision'`
apparaissent au lancement de Streamlit : bruit inoffensif (Streamlit
inspecte des modules de vision par ordinateur de la bibliothèque
`transformers` que le projet n'utilise jamais). Pour les masquer :
`streamlit run app.py --server.fileWatcherType none`.

## Évaluation

```bash
cd scripts
python evaluer_precision_recall.py
python evaluer_ragas.py
```

## Points à surveiller

- Le premier appel réel à l'API après un changement de clé/fichier
  nécessite un redémarrage complet du kernel (Python charge la clé et le
  code une seule fois au démarrage).
- Les quotas gratuits des fournisseurs de LLM évoluent régulièrement --
  se fier au comportement observé plutôt qu'à un chiffre fixe.

## Limites connues, non corrigées par choix

- Duplication de la logique d'orchestration entre `app.py` et
  `evaluer_precision_recall.py`.
- Dashboard de monitoring visuel, interface en cartes structurées,
  déploiement public : repoussés en fin de projet (voir documentation
  complète pour le détail).
