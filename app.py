""" NRC Microreactor Knowledge Agent interface """
import json
from pathlib import Path

import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import (DB_DIR, LLM_MODEL, EMBED_MODEL, COLLECTION_NAME, LLM_TIMEOUT)

INVENTORY_PATH = Path(__file__).parent / "docs" / "documentory_invenotry.json"

NRC_SYSTEM_PROMPT = """ 
You are NRC regulatory research assistant for research purposes to understand documentation about microreactors. Answer questions about NRC microreactor licensing, policy, and regulatory documents using ONLY the provided context.

CRITICAL RULES:
1. Cite specific accession numbers when referencing documents. Use the format "SECY-20-0093 (ML20129J985) or "SRM-SECY-24-0008 (ML25168A133)".
2. Never invent document titles, accession numbers, or content. If the context does not support an answer: say: "I don't have that information in the indexed documents."
3. Distinguish document typs clearly: SECY papers are staff recommendations to the Commission; SRMs are Commission decisions; Enclosure provide supporting detail attached to a parent SECY paper.
4. When asked about a specific document, use the document_id, subject, and accession from the context metadata to give precise answers.
5. Be precise about regulatory status - distinguish between what NRC has approved, proposed, recommended, or merely discussed.
6. If multiple documents are relevant to a question, cite each one.
"""

HELP_TEXT = """
Commands:
    help - show this menu
    sources -  show full text of sources from last answer
    clear - clear chat history
    exit - quit

Otherwise, just type a question about NRC microreactor regulation.
"""

def load_index():
    """ Load existing index from Chroma DB """
    Settings.llm = Ollama(model=LLM_MODEL, request_timeout=LLM_TIMEOUT)
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)

    chroma_client = chromadb.PersistentClient(path=str(DB_DIR))
    chroma_collection = chroma_client.get_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store)

    return index, chroma_collection.count()

def format_citations(response):
    """ Creates a citation list after each answer from the agent """
    seen = set()
    lines = []
    for node in response.source_nodes:
        m = node.metadata
        accession = m.get("accession", "unknown")
        doc_id = m.get("document_id", m.get("file_name", "unknown"))
        subject = m.get("subject", "")
        page = m.get("source", m.get("page","?"))

        key = (accession, page)
        if key in seen:
            continue
        seen.add(key)

        index = len(lines) + 1
        line = f"[{index}] {doc_id} ({accession}), p. {page}"
        if subject:
            line += f"\n    {subject}"
        lines.append(line)
    return "\n".join(lines) if lines else " (no sources)"

def format_full_sources(response):
    """ Detailed source view if user entered 'source' prompt """
    if not response.source_nodes:
        return "(no sources from last answer)"
    blocks = []

    for i, node in enumerate(response.source_nodes, 1):
        m = node.metadata
        header = (
            f"--- Source {i} ---\n"
            f"Document: {m.get('document_id', 'unknown')}"
            f"({m.get('accession', '?')})\n"
            f"Subject: {m.get('subject', '')}\n"
            f"Page: {m.get('source', m.get('page', '?'))}\n"
            f"Score: {node.score:.3f}\n"
            f"URL: {m.get('source_url', '')}\n"
        )
        blocks.append(f"{header}\n{node.text.strip()}\n")
    return "\n".join(blocks)


def main():
    print("Loading NRC microreactor index...")
    try:
        index, chunk_count = load_index()
    except (ValueError, Exception) as e:
        print(f"\nCould not load indexL {e}")
        print(" Run `python build_index.py` first")
        return
    
    try:
        with open(INVENTORY_PATH) as file:
            doc_count = len(json.load(file))
    except FileNotFoundError:
        doc_count = "?"
    
    memory = ChatMemoryBuffer.from_defaults(token_limit=3000) #to prevent system crash
    chat_engine = index.as_chat_engine(chat_mode="context", memory=memory, system_prompt=NRC_SYSTEM_PROMPT, similarity_top_k=4 ,verbose=False)

    print("NRC Microreactor Knowledge Agent")
    print("Powered by Mistral + nomic-embed-text (fully local)")
    print(f"{chunk_count} chunks indexed across {doc_count} NRC documents")
    print("Ask me anything about NRC Microreactor regulation.")
    print("Type help to see help menu.\n")

    last_response = None

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
            
        if not question:
            continue

        cmd = question.lower()
        if cmd == "help":
            print(HELP_TEXT)
            continue

        if cmd in {"exit", "quit"}:
            print("Goodbye!")
            break

        if cmd == "sources":
            if last_response is None:
                print("\nNo sources yet - ask a question first please.\n")
            else:
                print("\n" + format_full_sources(last_response))
            continue

        if cmd == "clear":
            memory.reset()
            last_response = None
            print("\nChat History cleared\n")
            continue

        response = chat_engine.chat(question)
        last_response = response
        print(f"\nAgent: {response}\n")
        print("Sources:")
        print(format_citations(response))
        print()

if __name__ == "__main__":
    main()