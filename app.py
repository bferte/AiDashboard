from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from fastapi.middleware.cors import CORSMiddleware
import os
import time

# Configuration des chemins
INDEX_PATH = "index/faiss_index"

app = FastAPI(title="Mon RAG Local Gratuit")

# Configuration indispensable pour le HTML/JS (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Chargement des outils au démarrage
print("Chargement des embeddings et de l'index FAISS...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Chargement de la base vectorielle
if os.path.exists(INDEX_PATH):
    db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    print("⚠️ Attention : Index FAISS introuvable. Lancez d'abord ingest.py")

class Ask(BaseModel):
    question: str
    model_name: str = "mistral:instruct"
    top_k: int = 4

@app.get("/")
def read_root():
    return {"status": "L'API RAG est en ligne !"}

@app.post("/ask")
def ask(payload: Ask):
    start_time = time.time()
    
    # --- ÉTAPE 1 : RETRIEVAL (Récupération avec Score) ---
    # On utilise similarity_search_with_score pour obtenir la distance L2
    docs_and_scores = db.similarity_search_with_score(payload.question, k=payload.top_k)
    
    # Calcul d'un score de pertinence (Inverse de la distance)
    # Plus la distance est proche de 0, plus le score est proche de 100%
    if docs_and_scores:
        avg_distance = sum([score for _, score in docs_and_scores]) / len(docs_and_scores)
        # Normalisation arbitraire pour le confort visuel (la distance L2 peut varier)
        relevance_score = max(0, min(100, 100 - (avg_distance * 50)))
    else:
        relevance_score = 0

    # Construction du contexte pour le LLM
    context_text = "\n\n---\n\n".join(
        [f"Source: {os.path.basename(doc.metadata['source'])}\n{doc.page_content}" for doc, score in docs_and_scores]
    )

    # --- ÉTAPE 2 : GÉNÉRATION (LLM) ---
    prompt = f"""
    Tu es un assistant expert. Réponds à la question suivante en utilisant UNIQUEMENT le contexte fourni ci-dessous.
    Si la réponse n'est pas dans le contexte, dis que tu ne sais pas.
    Cite les sources à la fin de ta réponse.

    Contexte :
    {context_text}

    Question :
    {payload.question}

    Réponse :
    """

    # Initialisation du modèle choisi
    llm = Ollama(model=payload.model_name)
    
    # On mesure le temps de génération uniquement
    gen_start = time.time()
    answer = llm.invoke(prompt)
    gen_duration = time.time() - gen_start

    # --- ÉTAPE 3 : CALCUL DES KPI ---
    total_duration = time.time() - start_time
    
    # Estimation des tokens : 1 mot ≈ 1.3 tokens
    words = len(answer.split())
    tokens_estimate = words * 1.3
    tps = round(tokens_estimate / gen_duration, 2) if gen_duration > 0 else 0

    return {
        "answer": answer, 
        "sources": list(set([os.path.basename(doc.metadata['source']) for doc, score in docs_and_scores])),
        "relevance": float(round(relevance_score, 2)), # Ajout de float()
        "tps": float(tps),                            # Ajout de float()
        "duration": float(round(total_duration, 2))    # Ajout de float()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)