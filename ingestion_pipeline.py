import os
import re
from langchain_community.document_loaders  import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma

def load_documents(docs_path="docs"):
    """Load all the text files from the directory"""
    print(f"Loading documents from: {docs_path}...")

    if not os.path.exists((docs_path)):
        raise FileNotFoundError(f"The specified path '{docs_path}' does not exist.")
    
    loader = DirectoryLoader(path=docs_path, glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No documents found in the specified path '{docs_path}'.")
    
    for i, doc in enumerate(documents):
        print(f"\nDocument {i+1}:")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content Length: {len(doc.page_content)} characters")
        print(f"Content Preview: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")

    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split the documents into smaller chunks"""
    print("Splitting documents into chunks...")

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Source: {chunk.metadata['source']}")
        print(f"Chunk Length: {len(chunk.page_content)} characters")
        print(f"Chunk Content Preview: {chunk.page_content[:100]}...")
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist the vector store using Chroma and Ollama embeddings"""
    print("Creating embeddings and storing in Chroma DB...")
    embedding_model = OllamaEmbeddings(model="mistral")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    print(f"Persisting vector store to: {persist_directory}")

    return vector_store

def clean_documents(documents):
    for doc in documents:
        doc.page_content = re.sub(
            r'^Source:.*?\nURL:.*?\n\n',
            '',
            doc.page_content,
            flags=re.DOTALL
        )
    return documents

def main():
    print("Main function")

    #Loading the files
    documents = load_documents(docs_path="docs")

    #Clean the documents
    documents = clean_documents(documents)

    #Chunking the files
    chunks = split_documents(documents)

    #Creating the vector store
    vector_store = create_vector_store(chunks)

if __name__ == "__main__":
    main()
    

    
