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
