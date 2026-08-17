
from rag.hybrid_retriever_v2 import process_question
from rag.generator import build_prompt, generate_answer


def extract_sources(retrieval_result):
    intent = retrieval_result["intent"]
    result = retrieval_result["result"]

    sources = []

    if intent == "verification_eligibilite":
        metadata = result["metadata"]

        sources.append({
            "source": metadata.get("source"),
            "programme": metadata.get("programme"),
            "etablissement": metadata.get("ies"),
            "ville": metadata.get("ville")
        })

    else:
        for item in result:
            metadata = item["metadata"]

            source = {
                "source": metadata.get("source"),
                "programme": metadata.get("programme"),
                "etablissement": metadata.get("ies"),
                "ville": metadata.get("ville")
            }

            if source not in sources:
                sources.append(source)

    return sources


def rag(question, model, tokenizer):
    retrieval_result = process_question(question)

    prompt = build_prompt(retrieval_result)

    answer = generate_answer(
        prompt,
        model,
        tokenizer
    )

    sources = extract_sources(retrieval_result)

    return {
        "question": question,
        "intent": retrieval_result["intent"],
        "profile": retrieval_result["profile"],
        "answer": answer,
        "sources": sources,
        "retrieval": retrieval_result["result"]
    }
