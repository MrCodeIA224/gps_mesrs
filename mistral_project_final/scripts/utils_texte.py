"""Fonctions utilitaires partagées."""
import re
import unicodedata


def normaliser(texte: str) -> str:
    if not texte:
        return ""
    texte = texte.lower()
    texte = unicodedata.normalize("NFKD", texte)
    texte = "".join(c for c in texte if not unicodedata.combining(c))
    texte = re.sub(r"[^a-z0-9\s]", " ", texte)
    texte = re.sub(r"\s+", " ", texte).strip()
    return texte
