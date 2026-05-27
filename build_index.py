import json
from pathlib import Path

import chromadb
from llama_index.core import (VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings,)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.readers.file import PyMuPDFReader 

from config import (DATA_DIR, DB_DIR, LLM_MODEL, EMBED_MODEL,
                    COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP, LLM_TIMEOUT)

INVENTORY_PATH = Path(__file__).parent / "docs" / "document_inventory.json"

EXCLUDE_FROM_LLM = ["file_path", "file_name", "file_size", "file_type", "creation_date", "last_modified_date", "source_url",]

EXCLUDE_FROM_EMBED = ["file_path", "file_name", "file_size", "file_type", "creation_date", "last_modified_date", "source_url",]

def load_inventory():
    with open(INVENTORY_PATH) as file:
        return json.load(file)

def attach_metadata(documents, inventory):
    """ Add inventory metadata for each document """
    enriched = 0
    missing = []
    for doc in documents:
        filename = Path(doc.metadata.get("file_path", "")).name
        if filename in inventory:
            doc.metadata.update(inventory[filename])
            doc.excluded_llm_metadata_keys = EXCLUDE_FROM_LLM
            doc.excluded_embed_metadata_keys = EXCLUDE_FROM_EMBED
            enriched +=1
        else:
            missing.append(filename)
    return enriched, missing

def build_index():
    print(f"Loading inventory from {INVENTORY_PATH}...")
    inventory = load_inventory()
    print(f"Inventory has {len(inventory)} documents.\n")
    
    print(f"Loading PDFs from {DATA_DIR}...")
    Settings.llm = Ollama(model=LLM_MODEL, request_timeout=LLM_TIMEOUT)
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)
    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP

    pdf_reader = PyMuPDFReader()

    documents = SimpleDirectoryReader(input_dir=str(DATA_DIR),recursive=True,required_exts=[".pdf"], file_extractor={".pdf": pdf_reader}).load_data()
    print(f"Loaded {len(documents)} document pages")

    enriched, missing = attach_metadata(documents, inventory)
    print(f"Attached metadata to {enriched} pages.")
    if missing:
        print(f"{len(set(missing))} files have no inventory entry:")
        for m in sorted(missing):
            print(f"    -{m}")

    DB_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(DB_DIR))

    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'.")
    except Exception:
        pass

    chroma_collection = chroma_client.create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Generating embeddings and building index... (this takes a few minutes)")
    VectorStoreIndex.from_documents(documents, storage_context=storage_context,show_progress=True)

    print(f"\nIndex built successfully.")
    print(f"    Pages: {len(documents)}")
    print(f"    Collection: {COLLECTION_NAME}")
    print(f"    Persisted to: {DB_DIR}")

if __name__ == "__main__":
    build_index()

