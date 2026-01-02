from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document  
from langchain_community.document_loaders import PyPDFLoader, TextLoader # Ajout de TextLoader
import os, glob

PDF_FOLDER = "data/pdfs"
HTML_FOLDER = "data/html" # Ajout du dossier HTML
INDEX_PATH = "index/faiss_index"

def load_documents():
    docs = []
    # Charger les PDF
    for path in glob.glob(os.path.join(PDF_FOLDER, "*.pdf")):
        loader = PyPDFLoader(path)
        docs.extend(loader.load())
    
    # Charger les textes scrappés (HTML)
    for path in glob.glob(os.path.join(HTML_FOLDER, "*.txt")):
        loader = TextLoader(path, encoding="utf-8")
        docs.extend(loader.load())
        
    return docs
    
def main():
    print("Chargement des documents...")
    docs = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=180,
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_PATH)
    print("Index FAISS enregistré :", INDEX_PATH)

if __name__ == "__main__":
    main()
