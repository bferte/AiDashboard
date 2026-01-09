from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import time
from typing import Any, Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableConfig, RunnablePassthrough, RunnableLambda
from langchain_core.messages import AIMessage, HumanMessage

import os

# --- Configuration ---
INDEX_PATH = "faiss_index"
MODEL_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_LLM = "mistral:instruct" # Assurez-vous que ce modèle est disponible via Ollama
USE_MOCK_LLM = False # Mettre à True pour utiliser la simulation sans Ollama

model_cache = {}

# --- Simulation du LLM (pour le développement) ---
class MockOllama(Runnable):
    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> str:
        print("INFO: Appel simulé du LLM.")
        prompt_str = str(input)
        question_marker = "Human:"
        question_index = prompt_str.rfind(question_marker)
        question = prompt_str[question_index + len(question_marker):].strip() if question_index != -1 else "inconnue"

        time.sleep(1)
        return f"[Réponse simulée pour la question : '{question.replace('}', '')}']"

# --- Gestionnaire Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Chargement des modèles et de l'index FAISS...")
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_EMBEDDING)

    if os.path.exists(INDEX_PATH):
        db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        model_cache["retriever"] = db.as_retriever(search_kwargs={'k': 3})
        print("Index FAISS chargé.")
    else:
        print("AVERTISSEMENT: Index FAISS non trouvé.")
        model_cache["retriever"] = None

    if USE_MOCK_LLM:
        print("Utilisation du LLM simulé.")
        model_cache["llm"] = MockOllama()
    else:
        print(f"Utilisation du LLM réel : {MODEL_LLM}")
        model_cache["llm"] = Ollama(model=MODEL_LLM)

    print("Chargement terminé.")
    yield
    print("Nettoyage des ressources...")
    model_cache.clear()

app = FastAPI(lifespan=lifespan)

# --- Modèles Pydantic ---
class ChatMessage(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    question: str
    history: List[ChatMessage]
    use_rag: bool

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

# --- Chaîne RAG & Chat ---
def format_docs(docs):
    return "\n\n---\n\n".join(
        f"Source: {os.path.basename(doc.metadata.get('source', 'Inconnue'))}\n{doc.page_content}"
        for doc in docs
    )

def get_sources(docs):
    return list(set(os.path.basename(doc.metadata.get('source', 'Inconnue')) for doc in docs))

@app.post("/api/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    chat_history = [AIMessage(content=msg.content) if msg.role == 'ai' else HumanMessage(content=msg.content) for msg in request.history]

    if request.use_rag and model_cache.get("retriever"):
        print("Mode RAG activé.")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Tu es un assistant expert. Réponds à la question en te basant sur le contexte suivant et l'historique de la conversation. Cite tes sources à la fin."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "Contexte:\n{context}\n\nQuestion: {question}")
        ])

        retriever = model_cache["retriever"]
        retrieved_docs = retriever.invoke(request.question)
        context = format_docs(retrieved_docs)
        sources = get_sources(retrieved_docs)

        chain = prompt | model_cache["llm"]
        response = chain.invoke({
            "context": context,
            "question": request.question,
            "chat_history": chat_history
        })

        return QueryResponse(answer=response, sources=sources)

    else:
        print("Mode Chat Direct activé.")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Tu es un assistant conversationnel. Réponds directement à la question en tenant compte de l'historique de la conversation."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{question}")
        ])

        chain = prompt | model_cache["llm"]
        response = chain.invoke({
            "question": request.question,
            "chat_history": chat_history
        })

        return QueryResponse(answer=response, sources=[])

# --- Servir le Frontend ---
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse('static/index.html')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
