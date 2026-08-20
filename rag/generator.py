
import torch


SYSTEM_PROMPT = """
Tu es un assistant d'orientation universitaire en Guinée.

Règles obligatoires :
- Réponds uniquement à partir des informations fournies.
- N'invente jamais une formation, un établissement, une série de bac,
  un seuil ou une condition d'admission.
- Respecte toujours la décision produite par les règles métier.
- Si l'information disponible est insuffisante, indique-le clairement.
- Réponds en français, de manière claire, concise et factuelle.
"""


def build_prompt(retrieval_result):
    question = retrieval_result["question"]
    intent = retrieval_result["intent"]
    profile = retrieval_result["profile"]
    result = retrieval_result["result"]

    if intent == "verification_eligibilite":
        metadata = result["metadata"]

        decision = (
            "ÉLIGIBLE"
            if result["eligible"]
            else "NON ÉLIGIBLE"
        )

        reasons = (
            "\n".join(result["reasons"])
            if result["reasons"]
            else "Toutes les contraintes vérifiées sont satisfaites."
        )

        return f"""
QUESTION :
{question}

PROFIL DÉTECTÉ :
Série : {profile.get("serie")}
Moyenne : {profile.get("moyenne")}
Ville : {profile.get("ville")}

CONTEXTE OFFICIEL :
Programme : {metadata.get("programme")}
Établissement : {metadata.get("ies")}
Ville : {metadata.get("ville")}
Séries autorisées : {metadata.get("options_autorisees")}
Seuil minimal : {metadata.get("seuil_bac")}/20
Source : {metadata.get("source")}

RÉSULTAT DES RÈGLES MÉTIER :
Série compatible : {result["checks"].get("serie")}
Moyenne compatible : {result["checks"].get("moyenne")}
Ville compatible : {result["checks"].get("ville")}
Décision : {decision}
Motif : {reasons}

Rédige la réponse destinée à l'étudiant.
"""

    contexts = []

    for item in result:
        metadata = item["metadata"]

        contexts.append(
            f"""
Programme : {metadata.get("programme")}
Établissement : {metadata.get("ies")}
Ville : {metadata.get("ville")}
Séries autorisées : {metadata.get("options_autorisees")}
Seuil : {metadata.get("seuil_bac")}
Source : {metadata.get("source")}
"""
        )

    return f"""
QUESTION :
{question}

PROFIL :
{profile}

PROGRAMMES RETROUVÉS :
{''.join(contexts)}

Réponds uniquement à partir de ces informations.
"""


def generate_answer(prompt, model, tokenizer):
    messages = [
        {
            "role": "system",
            "content": "SYSTEM_PROMPT"     
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=False
        )

    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return answer.strip()
