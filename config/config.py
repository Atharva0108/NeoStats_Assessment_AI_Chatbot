# config/config.py
import os
from dotenv import load_dotenv

# Load .env in local development
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)

# Google AI Studio API key (Gemini)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", None)

# Embeddings / FAISS files
EMBEDDINGS_DIR = os.getenv("EMBEDDINGS_DIR", os.path.join(BASE_DIR, "data", "embeddings"))
FAISS_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(EMBEDDINGS_DIR, "metadata.json")

# Embedding model name (sentence-transformers)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# RAG retrieval settings
TOP_K = int(os.getenv("TOP_K", 4))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))  # approx words per chunk; adjust as needed
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

# Gemini model configuration
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # change if you prefer a different variant
GEMINI_MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", 512))
