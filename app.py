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
        print("⚠️ Attention : Index FAISS introuvable.")
    yield
    print("Arrêt de l'API...")
    model_cache.clear()

# --- INITIALISATION DE L'APPLICATION FASTAPI ---
app = FastAPI(
    title="AI Dashboard Optique",
    lifespan=lifespan,
    description="API pour le tableau de bord de pilotage d'un magasin d'optique."
)

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

# --- ROUTES ---
@app.get("/", include_in_schema=False)
async def read_index():
    return FileResponse(os.path.join(STATIC_DIR, 'index.html'))

@app.get("/api/collaborateurs", summary="Liste des collaborateurs")
def get_collaborateurs():
    return analytics.get_collaborateurs_list()

@app.get("/api/kpi-data/{nom_collaborateur}", summary="Données KPI d'un collaborateur")
def get_kpi_data(nom_collaborateur: str):
    collaborateur_data = analytics.get_collaborateur_data(nom_collaborateur)
    if not collaborateur_data:
        raise HTTPException(status_code=404, detail=f"Collaborateur '{nom_collaborateur}' non trouvé.")
    store_average = analytics.get_store_average()
    return {
        "collaborateur": collaborateur_data,
        "moyenne_magasin": store_average
    }

@app.post("/api/analyze-collaborateur", summary="Génère un plan de coaching IA")
async def analyze_collaborateur(payload: AnalyseRequest):
    if not model_cache.get("rag_ready", False):
        raise HTTPException(status_code=503, detail="Service RAG non prêt.")

    nom = payload.nom_collaborateur
    collaborateur_data = analytics.get_collaborateur_data(nom)
    store_average = analytics.get_store_average()

    kpi_collab = collaborateur_data.get("Taux_2eme_Paire_VA", 0)
    kpi_moyenne = store_average.get("Taux_2eme_Paire_VA", 0)

    prompt_data = (
        f"Voici les chiffres de {nom}. "
        f"Il est à {kpi_collab}% sur la 2ème paire VA (Moyenne : {kpi_moyenne}%). "
        "En utilisant les procédures de vente dans nos PDF, propose un plan d'action de coaching précis."
    )

    db = model_cache["db"]
    # Utilisation de la méthode asynchrone pour la recherche
    docs = await db.asimilarity_search(prompt_data, k=3)
    context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])

    final_prompt = f"""
    Tu es un coach expert pour les professionnels de l'optique.
    Utilise le contexte suivant pour formuler ta réponse.
    Contexte: {context_text}
    Question: {prompt_data}
    Réponse:
    """

    try:
        llm = Ollama(model=payload.model_name)
        # Utilisation de la méthode asynchrone pour l'invocation
        answer = await llm.ainvoke(final_prompt)

        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur LLM : {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
