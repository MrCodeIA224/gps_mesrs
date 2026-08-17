"""
PATCH RAGAS (v2) — Corrige directement le bug d'import VertexAI dans le
fichier installé de RAGAS, SANS jamais faire `import ragas` (qui échouerait
avec exactement l'erreur qu'on cherche à corriger -- problème de l'œuf et
la poule dans la première version de ce script).

On localise le fichier via importlib.util.find_spec(), qui trouve
l'emplacement d'un paquet sans exécuter son code.

À exécuter UNE SEULE FOIS. Redémarrer le kernel du notebook après.
"""
import importlib.util
import os

spec = importlib.util.find_spec("ragas")
if spec is None or not spec.submodule_search_locations:
    raise RuntimeError("Impossible de localiser le paquet ragas. Est-il bien installé "
                        "dans l'environnement actif ?")

dossier_ragas = spec.submodule_search_locations[0]
chemin_fichier = os.path.join(dossier_ragas, "llms", "base.py")
print(f"Fichier ciblé : {chemin_fichier}")

if not os.path.exists(chemin_fichier):
    raise RuntimeError(f"Le fichier {chemin_fichier} n'existe pas -- structure du "
                        f"paquet différente de celle attendue.")

with open(chemin_fichier, encoding="utf-8") as f:
    contenu = f.read()

ANCIEN_IMPORT = "from langchain_community.chat_models.vertexai import ChatVertexAI\nfrom langchain_community.llms import VertexAI"

NOUVEAU_IMPORT = """try:
    from langchain_google_vertexai import ChatVertexAI, VertexAI
except ImportError:
    try:
        from langchain_community.chat_models.vertexai import ChatVertexAI
        from langchain_community.llms import VertexAI
    except ImportError:
        ChatVertexAI = None
        VertexAI = None"""

if ANCIEN_IMPORT not in contenu:
    print("\nATTENTION : le texte exact attendu n'a pas été trouvé dans le fichier.")
    print("Vérifie manuellement les lignes contenant 'vertexai' :")
    for i, ligne in enumerate(contenu.split("\n"), 1):
        if "vertexai" in ligne.lower():
            print(f"  ligne {i}: {ligne}")
else:
    contenu_corrige = contenu.replace(ANCIEN_IMPORT, NOUVEAU_IMPORT)
    with open(chemin_fichier, "w", encoding="utf-8") as f:
        f.write(contenu_corrige)
    print("\nPatch appliqué avec succès.")
    print("Redémarre maintenant le kernel du notebook, puis retente l'import RAGAS.")
