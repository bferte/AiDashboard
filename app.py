from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from fastapi.middleware.cors import CORSMiddleware
import os
import time
from contextlib import asynccontextmanager

# Configuration des chemins
INDEX_PATH = "index/faiss_index"

# Dictionnaire pour stocker les modèles chargés et l'état de l'API
model_cache = {}

# Template de prompt pour le LLM
PROMPT_TEMPLATE = """
Tu es un assistant expert. Réponds à la question suivante en utilisant UNIQUEMENT le contexte fourni ci-dessous.
Si la réponse n'est pas dans le contexte, dis que tu ne sais pas.
Cite les sources à la fin de ta réponse.

Contexte :
{context}

Question :
{question}

Réponse :
"""

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestionnaire de cycle de vie de l'application.
    Charge les modèles au démarrage et les nettoie à l'arrêt.
    """
    # --- DÉMARRAGE DE L'APPLICATION ---
    print("Démarrage de l'API et chargement des modèles...")

    # Chargement des embeddings
    model_cache["embeddings"] = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Chargement de la base vectorielle FAISS
    if os.path.exists(INDEX_PATH):
        model_cache["db"] = FAISS.load_local(
            INDEX_PATH,
            model_cache["embeddings"],
            allow_dangerous_deserialization=True
        )
        model_cache["api_ready"] = True
        print("✅ Modèles chargés. L'API est prête à recevoir des requêtes.")
    else:
        model_cache["db"] = None
        model_cache["api_ready"] = False
        print("⚠️ Attention : Index FAISS introuvable. L'API démarrera, mais sera en état non opérationnel.")

    yield  # L'application est maintenant en cours d'exécution

    # --- ARRÊT DE L'APPLICATION ---
    print("Arrêt de l'API et nettoyage des ressources...")
    model_cache.clear()
    print("Ressources nettoyées.")

app = FastAPI(
    title="Mon RAG Local Gratuit",
    lifespan=lifespan,
    description="Une API pour interroger un LLM local avec un contexte issu d'une base vectorielle."
)

# Configuration indispensable pour le HTML/JS (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

class Ask(BaseModel):
    question: str
    model_name: str = "mistral:instruct"
    top_k: int = 4

@app.get("/status", summary="Vérifie l'état de l'API")
def get_status():
    """
    Retourne le statut de l'API.
    - Si `status: ok`, l'API est prête à être utilisée.
    - Si une erreur 503 est retournée, les modèles ne sont pas chargés.
    """
    if model_cache.get("api_ready", False):
        return {"status": "ok", "message": "API prête et modèles chargés."}
    else:
        raise HTTPException(
            status_code=503,
            detail="Service indisponible : L'index vectoriel (FAISS) n'a pas été chargé. Lancez ingest.py."
        )

@app.post("/ask", summary="Pose une question au RAG")
def ask(payload: Ask):
    """
    Reçoit une question, recherche les documents pertinents et génère une réponse.
    """
    # Vérification de la disponibilité de l'API à chaque appel
    if not model_cache.get("api_ready", False):
        raise HTTPException(
            status_code=503,
            detail="Service indisponible : L'index vectoriel (FAISS) n'a pas été chargé. Lancez ingest.py."
        )

    start_time = time.time()
    db = model_cache["db"] # Récupération de la DB depuis le cache
    
    # --- ÉTAPE 1 : RETRIEVAL (Récupération avec Score) ---
    docs_and_scores = db.similarity_search_with_score(payload.question, k=payload.top_k)
    
    # Calcul d'un score de pertinence
    if docs_and_scores:
        avg_distance = sum([score for _, score in docs_and_scores]) / len(docs_and_scores)
        relevance_score = max(0, min(100, 100 - (avg_distance * 50)))
    else:
        relevance_score = 0

    context_text = "\n\n---\n\n".join(
        [f"Source: {os.path.basename(doc.metadata['source'])}\n{doc.page_content}" for doc, score in docs_and_scores]
    )

    # --- ÉTAPE 2 : GÉNÉRATION (LLM) ---
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=payload.question)

    llm = Ollama(model=payload.model_name)
    
    try:
        gen_start = time.time()
        answer = llm.invoke(prompt)
        gen_duration = time.time() - gen_start
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la communication avec le LLM : {e}"
        )

    # --- ÉTAPE 3 : CALCUL DES KPI ---
    total_duration = time.time() - start_time
    
    words = len(answer.split())
    tokens_estimate = words * 1.3
    tps = round(tokens_estimate / gen_duration, 2) if gen_duration > 0 else 0

    return {
        "answer": answer, 
        "sources": list(set([os.path.basename(doc.metadata['source']) for doc, score in docs_and_scores])),
        "relevance": float(round(relevance_score, 2)),
        "tps": float(tps),
        "duration": float(round(total_duration, 2))
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)