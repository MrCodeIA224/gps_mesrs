"""
EVALUATION RAGAS — Adaptée à l'architecture v2 (retrieval à double chemin).

Important : les questions de type "liste" (ex: "quels programmes à Kankan")
sortent du cadre classique de RAGAS Faithfulness/ContextPrecision, pensé
pour un retrieval par similarité -- on les exclut de cette évaluation
quantitative et on les teste manuellement (voir notebook, section dédiée).
Seules les questions de type "fait" (hybride + reranking) sont évaluées ici.

RAGAS utilise Groq comme juge (pas OpenAI par défaut), et BGE-M3 comme
modèle d'embeddings pour Answer Relevancy, cohérent avec le reste du projet.
"""
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

JEU_DE_TEST = [
    {"question": "Quels sont les débouchés en biologie ?",
     "reference": "Les débouchés incluent notamment technicien de laboratoire, enseignant, poursuite en master."},
    {"question": "Combien coûtent les frais d'orientation ?",
     "reference": "Les frais d'orientation sont de 50 000 GNF, payables par Orange Money."},
    {"question": "Comment récupérer mon mot de passe oublié ?",
     "reference": "Cliquer sur mot de passe oublié, choisir SMS ou e-mail, renseigner son INE, recevoir un nouveau mot de passe."},
    {"question": "J'ai eu 10/20 au bac, puis-je faire l'architecture ?",
     "reference": "Le seuil requis pour le programme d'architecture (ISAU) est de 13/20, donc 10/20 est insuffisant."},
    {"question": "Quelle est la capitale de la France ?",
     "reference": "Question hors périmètre, le chatbot doit se recentrer sur l'orientation."},
]


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
