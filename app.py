from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from fastapi.middleware.cors import CORSMiddleware
import os
from contextlib import asynccontextmanager

# Import du nouveau module d'analyse
import optique_analytics as analytics

# --- CONFIGURATION ---
INDEX_PATH = "index/faiss_index"
STATIC_DIR = "static"
model_cache = {}

# --- CYCLE DE VIE DE L'APPLICATION (LIFESPAN) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Démarrage de l'API et chargement des modèles...")

    # Chargement des embeddings et de la base vectorielle pour le RAG
    model_cache["embeddings"] = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists(INDEX_PATH):
        model_cache["db"] = FAISS.load_local(
            INDEX_PATH,
            model_cache["embeddings"],
            allow_dangerous_deserialization=True
        )
        model_cache["rag_ready"] = True
        print("✅ Modèles RAG chargés.")
    else:
        model_cache["db"] = None
        model_cache["rag_ready"] = False
        print("⚠️ Attention : Index FAISS introuvable. Les fonctionnalités de coaching IA seront limitées.")

    yield

    print("Arrêt de l'API et nettoyage des ressources...")
    model_cache.clear()

# --- INITIALISATION DE L'APPLICATION FASTAPI ---
app = FastAPI(
    title="AI Dashboard Optique",
    lifespan=lifespan,
    description="API pour le tableau de bord de pilotage d'un magasin d'optique."
)

# Montage du répertoire statique (pour CSS, JS, images, etc.)
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODÈLES DE DONNÉES (PYDANTIC) ---
class AnalyseRequest(BaseModel):
    nom_collaborateur: str
    model_name: str = "mistral:instruct"

# --- ROUTES DE L'INTERFACE (FRONTEND) ---
@app.get("/", include_in_schema=False)
async def read_index():
    """Sert la page principale de l'application."""
    return FileResponse(os.path.join(STATIC_DIR, 'index.html'))

# --- ROUTES DE L'API (BACKEND) ---
@app.get("/api/collaborateurs", summary="Liste des collaborateurs")
def get_collaborateurs():
    """Retourne la liste complète des noms des collaborateurs."""
    collaborateurs = analytics.get_collaborateurs_list()
    if not collaborateurs:
        raise HTTPException(status_code=404, detail="Aucun collaborateur trouvé.")
    return collaborateurs

@app.get("/api/kpi-data/{nom_collaborateur}", summary="Données KPI d'un collaborateur")
def get_kpi_data(nom_collaborateur: str):
    """Retourne les KPIs d'un collaborateur et la moyenne du magasin."""
    collaborateur_data = analytics.get_collaborateur_data(nom_collaborateur)
    if not collaborateur_data:
        raise HTTPException(status_code=404, detail=f"Collaborateur '{nom_collaborateur}' non trouvé.")
    
    store_average = analytics.get_store_average()
    
    return {
        "collaborateur": collaborateur_data,
        "moyenne_magasin": store_average
    }

@app.post("/api/analyze-collaborateur", summary="Génère un plan de coaching IA")
def analyze_collaborateur(payload: AnalyseRequest):
    """Analyse les KPIs d'un collaborateur et génère des conseils de coaching."""
    if not model_cache.get("rag_ready", False):
        raise HTTPException(status_code=503, detail="Le service RAG n'est pas prêt. Index FAISS manquant.")

    nom = payload.nom_collaborateur
    collaborateur_data = analytics.get_collaborateur_data(nom)
    if not collaborateur_data:
        raise HTTPException(status_code=404, detail=f"Collaborateur '{nom}' non trouvé.")

    store_average = analytics.get_store_average()

    # Construction du prompt enrichi avec les données
    kpi_collab = collaborateur_data.get("Taux_2eme_Paire_VA", 0)
    kpi_moyenne = store_average.get("Taux_2eme_Paire_VA", 0)

    prompt_data = (
        f"Voici les chiffres de {nom}. "
        f"Il est à {kpi_collab}% sur la 2ème paire VA (Moyenne du magasin : {kpi_moyenne}%). "
        "En utilisant les procédures de vente dans nos PDF, propose un plan d'action de coaching précis."
    )

    # Étape de Retrieval (RAG)
    db = model_cache["db"]
    # Recherche de documents pertinents pour le coaching de vente
    docs = db.similarity_search(prompt_data, k=3)
    context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])

    # Construction du prompt final pour le LLM
    final_prompt = f"""
    Tu es un coach expert pour les professionnels de l'optique.
    Utilise le contexte suivant pour formuler ta réponse.

    Contexte des procédures de vente:
    {context_text}

    Question:
    {prompt_data}

    Réponse:
    """

    try:
        llm = Ollama(model=payload.model_name)
        answer = llm.invoke(final_prompt)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la communication avec le LLM : {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
