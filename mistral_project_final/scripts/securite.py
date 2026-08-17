"""
SECURITE — Masque toute donnée sensible AVANT qu'elle ne soit stockée en
mémoire de session ou en logs (exigence du prompt système, sections 11-13).

Important : ceci est complémentaire au prompt système, pas un remplacement.
Le prompt empêche le LLM de RÉPÉTER une donnée sensible dans sa réponse ;
ce module empêche qu'elle soit ne serait-ce que CONSERVÉE quelque part.
"""
import re

# Motifs volontairement larges (mieux vaut sur-masquer que sous-masquer)
MOTIFS_SENSIBLES = [
    # Code USSD Orange Money complet (ex: *144*4*2*1234#)
    (re.compile(r"\*\d{2,4}(\*\d+)*\*?\d*#?"), "[CODE ORANGE MONEY MASQUÉ]"),
    # Suite de 4 à 8 chiffres isolée, précédée d'un mot évocateur (code, mot
    # de passe...), avec ou sans ponctuation entre les deux (formulation
    # naturelle : "mon mot de passe est 12345" doit aussi être masqué)
    (re.compile(r"(?i)(code|mot de passe|password|pin)\s*(?:est|:|-)?\s*(\d{4,8})"), r"\1 [MASQUÉ]"),
    # INE (Identifiant National Étudiant) : suite alphanumérique de 6 à 15
    # caractères précédée du mot "INE", avec ou sans ponctuation -- même
    # principe que le mot de passe (bug réel signalé : l'INE n'était pas
    # détecté comme donnée sensible alors qu'il doit être protégé au même
    # titre, voir prompt système section 13).
    (re.compile(r"(?i)\b(mon ine|ine)\s*(?:est|:|-|=)?\s*([a-z0-9]{6,15})\b"), r"\1 [MASQUÉ]"),
    # Suite de 6 chiffres isolée (format probable d'un code SMS)
    (re.compile(r"\b\d{6}\b"), "[CODE MASQUÉ]"),
]


def masquer_donnees_sensibles(texte: str) -> str:
    """Retourne une version du texte avec les données sensibles remplacées.
    Utilisée avant tout stockage en mémoire structurée ou en logs."""
    resultat = texte
    for motif, remplacement in MOTIFS_SENSIBLES:
        resultat = motif.sub(remplacement, resultat)
    return resultat


if __name__ == "__main__":
    tests = [
        "mon code est *144*4*2*1234#",
        "mon mot de passe : 12345",
        "j'ai reçu le code 483920 par SMS",
        "quels sont les programmes de droit",  # ne doit rien masquer
        "mon INE est GN2024001234",
        "mon ine : 123456789",
        "voici mon INE GN2024001234, pouvez-vous vérifier mon dossier ?",
    ]
    for t in tests:
        print(f"{t!r} -> {masquer_donnees_sensibles(t)!r}")
