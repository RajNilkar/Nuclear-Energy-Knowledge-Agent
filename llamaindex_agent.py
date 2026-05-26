import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

Settings.llm = Ollama(model="mistral", request_timeout=300)
Settings.embed_model = OllamaEmbedding(model_name="mistral")

documents = SimpleDirectoryReader('docs').load_data()
print(f"Loaded {len(documents)} documents.")

chroma_client = chromadb.PersistentClient(path="db/llamaindex_chroma_db")
chroma_collection = chroma_client.get_or_create_collection(name="nuclear_energy")
vector_store = ChromaVectorStore(chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

chat_engine = index.as_chat_engine(chat_mode="condense_question")

print("Welcome to the Nuclear Energy Knowledge Agent!")
print("Powered by Mistral + RAG (Local)")
print("Ask any question about nuclear energy, microreactors, or related topics, and I'll do my best to provide an answer based on the documents I have.")
print("Type 'exit' to end the conversation.\n")

while True:
    question = input("You: ").strip()
    if not question:
        continue
    if question.lower() == "help":
        print("\nAvailable commands:")
        print("help - Show this help message")
        print("sources - Show sources used in the last answer")
        print("clear - Clear the conversation history")
        print("exit - End the conversation")
        continue
    if question.lower() in ["exit", "quit"]:
        print("\nGoodbye!")
        break
    
    response = chat_engine.chat(question)
    print(f"\nAgent: {response.response}")

    if response.source_nodes:
        print("\nSources:")
        sources_seen = set()
        for node in response.source_nodes:
            source = node.node.metadata.get("source", "unknown source")
            score = node.score if node.score else 0
            if source not in sources_seen:
                sources_seen.add(source)
                print(f"Source: {source} (relevance: {score:.0%})")
    print()