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
