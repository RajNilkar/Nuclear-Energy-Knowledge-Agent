# Nuclear Energy Knowledge Agent
A locally hosted conversational AI agent that answers questions about NuCube Energy, nuclear microreactors, and TRISO fuel technology using Retrieval-Augmented Generation (RAG). The entire system runs locally with no external API calls — no data ever leaves your machine.

## Why I Built This
NuCube Energy is developing a locally hosted AI agent for their engineering and research teams to provide fast, accurate access to internal documentation. This project is a working prototype of that concept, built around publicly available nuclear energy and NuCube-specific documentation to demonstrate the full RAG pipeline end to end.

## Tech Stack

* LLM: Mistral (running locally via Ollama)
* RAG Framework: LangChain
* Vector Database: ChromaDB (cosine similarity)
* Embeddings: Ollama Embeddings (Mistral)
* Language: Python

## How It Works
The system is split into two pipelines:
Ingestion Pipeline (ingestion_pipeline.py)

1. Loads text documents from the docs/ folder
2. Cleans metadata headers from document content
3. Splits documents into chunks (1000 characters, 200 overlap)
4. Generates vector embeddings for each chunk
5. Stores embeddings in ChromaDB for fast similarity search

Conversational Agent (agent.py)

1. User asks a question in natural language
2. If conversation history exists, the question is rewritten for clarity using prior context
3. The question is embedded and compared against stored chunks using cosine similarity
4. The top 5 most relevant chunks are retrieved with relevance scores
5. Retrieved chunks are passed to Mistral as context to generate a grounded answer
6. Sources and relevance percentages are displayed with every response
7. Conversation history is maintained for natural follow-up questions

## Features

* Conversational memory — follow-up questions understand context from previous exchanges
* Source citations — every answer shows which documents were used and their relevance scores
* Relevance threshold — irrelevant questions are caught before reaching the LLM, preventing hallucination on off-topic queries
* History-aware retrieval — ambiguous follow-up questions are rewritten into self-contained queries before searching
* Response timing — each answer displays how long it took to generate
* Built-in commands — help, sources, clear, and exit for session management

## Project Structure
nuclear-energy-knowledge-agent/
├── docs/                      # Source documents (8 text files)
├── db/
│   └── chroma_db/             # ChromaDB vector store (auto-generated)
├── ingestion_pipeline.py      # Document loading, chunking, embedding, storage
├── agent.py                   # Conversational RAG agent
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md