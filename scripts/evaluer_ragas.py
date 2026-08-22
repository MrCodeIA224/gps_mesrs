"""
EVALUATION RAGAS — Adaptée à l'architecture v2 (retrieval à double chemin).

Important : les questions de type "liste" (ex: "quels programmes à Kankan")
sortent du cadre classique de RAGAS Faithfulness/ContextPrecision, pensé
pour un retrieval par similarité -- on les exclut de cette évaluation
quantitative (voir evaluer_precision_recall.py, qui couvre spécifiquement
ce chemin avec une mesure de recall + une vérification anti-hallucination
dédiée). Seules les questions de type "fait" (hybride + reranking) sont
évaluées ici.

Le jeu de test est chargé depuis data/jeu_de_test_annote.json -- la MÊME
source que evaluer_precision_recall.py -- plutôt qu'une liste séparée
codée en dur ici : évite que les deux scripts dérivent avec deux jeux de
questions différents au fil du temps. Seuls les cas qui portent un champ
"reference" (une réponse de référence en langage naturel, nécessaire à
RAGAS) sont repris.

RAGAS utilise Mistral comme juge (cohérent avec le reste du projet), et
BGE-M3 comme modèle d'embeddings pour Answer Relevancy.
"""
import json
import os
import pandas as pd
from dotenv import load_dotenv

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall

from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings

from retrieval import MoteurRecherche
from llm import appeler_llm

load_dotenv()

JEU_DE_TEST_PATH = "../data/jeu_de_test_annote.json"


def charger_jeu_de_test() -> list:
    with open(JEU_DE_TEST_PATH, encoding="utf-8") as f:
        tous_les_cas = json.load(f)
    return [
        {"question": cas["question"], "reference": cas["reference"]}
        for cas in tous_les_cas
        if cas.get("reference")
    ]


JEU_DE_TEST = charger_jeu_de_test()


def construire_dataset(moteur: MoteurRecherche) -> EvaluationDataset:
    echantillons = []
    for cas in JEU_DE_TEST:
        question = cas["question"]
        recherche = moteur.rechercher(question)

        if recherche["intention"] in ("liste", "hors_sujet", "clarification"):
            print(f"[ignoré, hors cadre RAGAS classique - {recherche['intention']}] {question}")
            continue

        resultats = recherche["resultats"]
        reponse = appeler_llm(question, resultats)
        contextes = [r["texte"] for r in resultats] if resultats else ["(aucun contexte récupéré)"]

        echantillons.append(SingleTurnSample(
            user_input=question, response=reponse, retrieved_contexts=contextes,
            reference=cas["reference"],
        ))
        print(f"[fait] {question}")

    return EvaluationDataset(samples=echantillons)


def main():
    print("Chargement du moteur de recherche...")
    moteur = MoteurRecherche()

    print("\nExécution du pipeline sur le jeu de test...")
    dataset = construire_dataset(moteur)

    print("\nConfiguration du juge RAGAS (Groq)...")
    evaluator_llm = LangchainLLMWrapper(
        ChatMistralAI(model="mistral-large-latest", mistral_api_key=os.environ["MISTRAL_API_KEY"])
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    )

    print("\nCalcul des métriques...")
    resultat = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    df = resultat.to_pandas()
    print("\n=== Résultats détaillés ===")
    print(df[["user_input", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]])
    print("\n=== Moyennes ===")
    print(df[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].mean())

    df.to_csv("../data/processed/resultats_ragas.csv", index=False)
    print("\nSauvegardé -> data/processed/resultats_ragas.csv")


if __name__ == "__main__":
    main()
