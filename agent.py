from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import time

persistent_directory = "db/chroma_db"

embedding_model = OllamaEmbeddings(model="mistral")

db = Chroma(persist_directory=persistent_directory, embedding_function=embedding_model, collection_metadata={"hnsw:space":"cosine"})

model = OllamaLLM(model="mistral")

#Store conversation history
chat_history= []

last_sources = []

def ask_question(user_question):
    global last_sources
    search_question = user_question

    start_time = time.time()

    if chat_history:
        messages = [
            SystemMessage(content="You are a helpful assistant. Given the chat history, rewrite the new question to be clear and self-contained. Return only the rewritten question.")
        ] + chat_history + [
            HumanMessage(content=user_question)
        ]
        result = model.invoke(messages)
        search_question = result.strip()

    # retriever = db.as_retriever(search_kwargs={"k": 5})
    # docs = retriever.invoke(search_question)

    results = db.similarity_search_with_score(search_question, k=5)
    docs = [doc for doc, score in results]
    scores = [score for doc, score in results]

    if scores[0] > 0.50:
        print("\nAgent: I couldn't find any relevant documents to answer that question. Please try asking something else or rephrase your question.\n")
        return
    combined_input = f"""Based on the following documents, please answer the question: {user_question}
        
    Documents:
    {chr(10).join(f"- {doc.page_content}" for doc in docs)}

    Please provie a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents.
    """

    messages = [
        SystemMessage(content="You are a knowledgeable assistant specializing in nuclear energy topics. Answer questions accurately based on the provided documents.Do not combine or infer information across different documents. If two documents contain conflicting or unrelated details, prioritize the most specific source.")
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]

    result = model.invoke(messages)
    answer = result

    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    last_sources = docs
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\nAgent: {answer}")
    print("\nSources:")
    sources_seen = set()
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown source")
        if source not in sources_seen:
            sources_seen.add(source)
            print(f"  - {source} (relevance: {((1 - scores[i]) * 100):.0f}%)")
    print(f"(Response time: {elapsed_time:.1f} seconds)\n")

    return answer

def start_agent():
    collection = db._collection
    doc_count = collection.count()
    print("Welcome to the Nuclear Energy Knowledge Agent!")
    print("Powered by Mistral + RAG (Local)")
    print(f"  Knowledge base: {doc_count} document chunks indexed")
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
        if question.lower() == "sources":
            if last_sources:
                print("\nSources used in the last answer:")
                sources_seen = set()
                for doc in last_sources:
                    source = doc.metadata.get("source", "unknown source")
                    if source not in sources_seen:
                        sources_seen.add(source)
                        print(f"- {source}")
                print()
            else:
                print("\nNo previous sources found.\n")
            continue
        if question.lower() == "clear":
            chat_history.clear()
            last_sources.clear()
            print("\nAgent: Conversation history cleared. Ask me a new question!\n")
            continue
        if any(word in question.lower() for word in ["thanks", "thank you"]):
            print("\nYou're welcome! If you have any more questions, feel free to ask.")
            continue
        if any(word == question.lower() for word in ["hi", "hello", "hey"]):
            print("\nAgent: Hello! Ask me anything about Nuclear Energy, microreactors, or nuclear technology.\n")
            continue
        if question.lower() in ["exit", "quit"]:
            print("\nAgent: Goodbye!")
            break
        ask_question(question)

if __name__ == "__main__":
    start_agent()