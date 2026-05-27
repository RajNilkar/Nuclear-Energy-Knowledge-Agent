# NRC Microreactor Knowledge Agent
 
A locally-hosted conversational AI agent that answers questions about NRC microreactor regulation by retrieving and citing real documents from the NRC's ADAMS public database. The entire system runs locally — no external API calls, no data leaves the machine.
 
## What it does
 
Ask natural language questions about NRC microreactor licensing, policy, factory fabrication, NOAK deployment, and related regulatory topics. The agent retrieves relevant content from indexed NRC documents and answers with accession-number citations on every response.
 
Example:
 
```
You: what is SECY-20-0093 about?
 
Agent: SECY-20-0093 is about "Policy and Licensing Considerations Related to
Micro-Reactors." This document provides recommendations on the policy and
licensing considerations related to micro-reactors, as dated October 6, 2020.
The accession number for this document is ML20129J985.
 
Sources:
  [1] SECY-20-0093 (ML20129J985), p. 7
      Policy and Licensing Considerations Related to Micro-Reactors
  [2] SECY-24-0008 (ML23207A250), p. 29
      Micro-Reactor Licensing and Deployment Considerations
  [3] SECY-25-0052 (ML24309A261), p. 17
      Nth-of-a-Kind Microreactor Licensing and Deployment Considerations
  [4] SECY-20-0093 Enclosure 1 (ML20254A365), p. 15
      Technical, Licensing, and Potential Policy Issues for Micro-Reactors
```
 
## Why I built this
 
The purpose is to learn how to build an agent that can help answer questions about NRC documentation on microreactors.
 
## Tech stack
 
- **LLM:** Mistral (running locally via Ollama)
- **Embeddings:** nomic-embed-text (also via Ollama)
- **Vector database:** ChromaDB (persistent)
- **RAG framework:** LlamaIndex
- **PDF extraction:** PyMuPDF
- **Language:** Python 3.10+
No external API calls. No cloud dependencies. Everything runs on the developer's machine.
 
## Indexed documents
 
Twelve documents from the NRC's ADAMS public database, focused on microreactor licensing policy:
 
| Document | Accession | Type |
|----------|-----------|------|
| SECY-20-0093 — Policy and Licensing Considerations Related to Micro-Reactors | ML20129J985 | SECY Paper |
| SECY-20-0093 Enclosure 1 — Technical, Licensing, and Potential Policy Issues | ML20254A365 | Enclosure |
| SECY-20-0093 Enclosure 2 — Possible Near-Term Licensing Approaches | ML20254A366 | Enclosure |
| SECY-24-0008 — Fuel Loading and Operational Testing at a Factory | ML23207A250 | SECY Paper |
| SECY-24-0008 Enclosure — Factory-Fabricated Micro-Reactors | ML23207A251 | Enclosure |
| SECY-25-0052 — Nth-of-a-Kind Microreactor Licensing | ML24309A261 | SECY Paper |
| SECY-25-0052 Enclosure 1 — Standardization of Operational Programs | ML24309A262 | Enclosure |
| SECY-25-0052 Enclosure 2 — Environmental Reviews for NOAK | ML24309A263 | Enclosure |
| SECY-25-0052 Enclosure 3 — Technical, Licensing, and Policy Considerations for NOAK | ML24309A260 | Enclosure |
| SECY-25-0052 Enclosure 4 — Licensing Steps and Estimated Timeframes for NOAK | ML24318C452 | Enclosure |
| SRM-SECY-24-0008 — Commission's Response on Factory Fuel Loading | ML25168A133 | SRM |
| Microreactor Activities Integration Tables | ML25225A024 | Staff Document |
 
## Setup
 
Prerequisites:
- Python 3.10+
- [Ollama](https://ollama.com) installed locally
```bash
# Pull the models
ollama pull mistral
ollama pull nomic-embed-text
 
# Install dependencies
pip install -r requirements.txt
 
# Build the index (run once after adding PDFs)
python build_index.py
 
# Start the chat
python app.py
```
 
## Commands
 
While running `app.py`:
 
- `help` — show available commands
- `sources` — show the full text of the chunks retrieved for the last answer (with relevance scores and ADAMS URLs)
- `clear` — reset chat history
- `exit` — quit
## Project structure
 
```
nrc-microreactor-agent/
├── app.py                       # Chat interface
├── build_index.py               # One-time index builder
├── config.py                    # Models, paths, chunking config
├── debug.py                     # Retrieval inspection script
├── data/raw/licensing/          # NRC PDFs
├── db/chroma_db/                # ChromaDB persistence (gitignored)
├── docs/
│   └── document_inventory.json  # Metadata for each indexed PDF
├── requirements.txt
└── README.md
```
 
## Known limitations
 
This is a single-developer demo project, not production-ready. Current limitations I'm actively working on:
 
- **Authority vs length in retrieval.** Shorter authoritative documents (like SRMs) can get out-retrieved by their longer parent SECY papers, since longer documents produce more chunks for semantic search. The agent occasionally describes a staff proposal as "under consideration" when the Commission's SRM has actually decided on it.
- **Acronym hallucination.** Mistral sometimes fabricates abbreviations (e.g. "design certification rule (OCR)" instead of DCR). A more rigorous evaluation pass would surface these systematically.
- **No automated evaluation harness yet.** Quality is currently spot-checked via the `sources` command. A golden-question set would catch regressions across changes.
## What I'd build next
 
- Document-type filtering and hybrid keyword+semantic retrieval to address the authority-vs-length issue
- ADAMS auto-downloader: pull PDFs by accession number directly from NRC's public database
- Automated evaluation harness with golden Q&A pairs
- Expanded corpus across other regulatory areas (safety analysis, reactor designs, digital twins, operational oversight)
- MCP server wrapper to expose retrieval as a tool callable by other agents