""" Configuration for NRC Microreactor Agent """
from pathlib import Path

#Project Paths
PROJECT_ROOT = Path(__file__).parent #Root dir
DATA_DIR = PROJECT_ROOT
DB_DIR = PROJECT_ROOT / "db" / "chroma_db"

#Models
LLM_MODEL = "mistral"
EMBED_MODEL = "nomic-embed-text"

#Vector store
COLLECTION_NAME = "nrc_microreactors"

#Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

#LLM Timeout
LLM_TIMEOUT = 180
