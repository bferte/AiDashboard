from unstructured.partition.auto import partition
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from codecarbon import EmissionsTracker
import os
import ollama

# --- Fonctions de description d'image (réelle et simulée) ---

def describe_image_with_ollama_real(image_path, model="llava", prompt="Décris cette image en détail."):
    """(Version réelle)"""
    try:
        with open(image_path, 'rb') as image_file:
            image_bytes = image_file.read()
        response = ollama.chat(model=model, messages=[{'role': 'user', 'content': prompt, 'images': [image_bytes]}])
        return response['message']['content']
    except Exception as e:
        print(f"ERREUR: Impossible de se connecter à Ollama. {e}")
        raise

def mock_describe_image(image_path, model="llava", prompt="Décris cette image en détail."):
    """(Version simulée)"""
    print(f"INFO: Appel simulé de la description d'image pour {image_path}")
    return f"[Description textuelle simulée pour l'image : {os.path.basename(image_path)}]"

# --- Le sélecteur de fonction ---
# IMPORTANT: Pour activer la description d'image réelle, changez la ligne ci-dessous pour :
# describe_image_with_ollama = describe_image_with_ollama_real
describe_image_with_ollama = mock_describe_image

# --- Fonctions de traitement ---

def process_documents(data_path="data/"):
    """Charge les documents, gère les images et les PDF de manière robuste."""
    all_elements = []
    image_paths = []

    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        if not os.path.isfile(file_path):
            continue

        print(f"Traitement du fichier : {file_path}")
        try:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(file_path)
            else:
                elements = partition(
                    filename=file_path,
                    extract_images_in_pdf=True,
                    infer_table_structure=True,
                    image_output_dir_path="figures/"
                )
                all_elements.extend(elements)

        except ImportError as e:
            if "poppler" in str(e).lower():
                print(f"AVERTISSEMENT: Le traitement de '{filename}' a été sauté car la dépendance système 'poppler' est manquante.")
                print("Pour activer l'extraction de texte et d'images à partir de PDF, veuillez installer 'poppler-utils'.")
            else:
                print(f"Erreur inattendue lors du traitement de {filename}: {e}")
        except Exception as e:
            print(f"Erreur lors du traitement de {filename}: {e}")

    extracted_image_elements = [el for el in all_elements if el.category == "Image"]
    for img_el in extracted_image_elements:
        if img_el.metadata.image_path:
            image_paths.append(img_el.metadata.image_path)

    text_elements = [el for el in all_elements if el.category != "Image"]
    return text_elements, list(set(image_paths))

def create_chunks(text_elements, image_paths):
    """Crée des chunks adaptatifs."""
    chunks = []
    image_descriptions = {path: describe_image_with_ollama(path) for path in image_paths}

    for img_path, desc in image_descriptions.items():
        chunks.append(Document(page_content=desc, metadata={"source": img_path, "type": "image_description"}))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    langchain_docs = [Document(page_content=el.text, metadata={"source": el.metadata.filename, "type": "text"}) for el in text_elements]
    chunks.extend(text_splitter.split_documents(langchain_docs))
    return chunks

def create_and_save_vectorstore(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2", index_path="faiss_index"):
    """Crée et sauvegarde la base vectorielle FAISS."""
    if not chunks:
        print("\nAVERTISSEMENT: Aucun chunk n'a été créé. La base vectorielle sera vide.")
        return
    print("\n--- Création de la base vectorielle FAISS ---")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    vectorstore.save_local(index_path)
    print(f"Base vectorielle sauvegardée dans : {index_path}")

if __name__ == "__main__":
    tracker = EmissionsTracker(project_name="RAG_Multimodal_Ingestion")
    try:
        tracker.start()
        print("--- Démarrage du processus d'ingestion ---")

        raw_text_elements, image_paths = process_documents()
        all_chunks = create_chunks(raw_text_elements, image_paths)

        print(f"\nNombre total de chunks créés : {len(all_chunks)}")
        create_and_save_vectorstore(all_chunks)

    finally:
        emissions: float = tracker.stop()
        print("\n--- Suivi énergétique CodeCarbon ---")
        print(f"Émissions de CO2 pour cette exécution : {emissions} kg")
        print("Un rapport détaillé a été généré dans 'emissions.csv'.")
