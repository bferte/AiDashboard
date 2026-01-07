from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import time
from typing import Any, Dict, Optional

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.llms import Ollama # Import réel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

import os

# --- Configuration ---
INDEX_PATH = "faiss_index"
MODEL_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"
# MODEL_LLM = "mistral:instruct" # Modèle réel

model_cache = {}

# --- Simulation du LLM (Compatible avec LangChain) ---
class MockOllama(Runnable):
    """Un LLM simulé qui gère correctement les entrées de la chaîne LCEL."""
    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> str:
        print("INFO: Appel simulé du LLM.")
        # Convertir l'entrée (qui peut être un ChatPromptValue) en chaîne de caractères
        prompt_str = str(input)

        question_marker = "Question :"
        question_index = prompt_str.rfind(question_marker) # Utiliser rfind pour trouver la dernière occurrence
        question = prompt_str[question_index + len(question_marker):].strip() if question_index != -1 else "inconnue"

        time.sleep(1) # Simuler un délai de génération
        return f"[Réponse simulée du RAG pour la question : '{question}']"

# --- Gestionnaire Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Chargement des modèles et de l'index FAISS...")
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_EMBEDDING)

    if os.path.exists(INDEX_PATH):
        db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        model_cache["retriever"] = db.as_retriever(search_kwargs={'k': 5})
        print("Index FAISS chargé avec succès.")
    else:
        print(f"AVERTISSEMENT: Index FAISS non trouvé.")
        model_cache["retriever"] = None

    # Utiliser le LLM simulé. Pour le réel, changez la ligne ci-dessous par:
    # from langchain_community.llms import Ollama
    # model_cache["llm"] = Ollama(model=MODEL_LLM)
    model_cache["llm"] = MockOllama()
    print("Chargement terminé.")

    yield

    print("Nettoyage des ressources...")
    model_cache.clear()

app = FastAPI(lifespan=lifespan)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

# --- Chaîne RAG ---
def setup_rag_chain():
    prompt_template = """
    Contexte :
    {context}

    Question :
    {question}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    def format_docs(docs):
        return "\n\n---\n\n".join(
            f"Source: {os.path.basename(doc.metadata.get('source', 'Inconnue'))}\n{doc.page_content}"
            for doc in docs
        )

    rag_chain = (
        {"context": model_cache["retriever"] | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | prompt
        | model_cache["llm"]
    )
    return rag_chain

def get_sources(docs):
    return list(set(os.path.basename(doc.metadata.get('source', 'Inconnue')) for doc in docs))

@app.post("/api/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    if model_cache.get("retriever") is None:
        return QueryResponse(answer="Erreur: Index FAISS non chargé.", sources=[])

    retrieved_docs = model_cache["retriever"].invoke(request.question)

    rag_chain = setup_rag_chain()
    answer = rag_chain.invoke(request.question)

    sources = get_sources(retrieved_docs)

    return QueryResponse(answer=answer, sources=sources)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse('static/index.html')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
