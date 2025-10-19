# models/embeddings.py
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from config.config import EMBEDDING_MODEL_NAME

# Cache model in memory during runtime to avoid repeated downloads
_MODEL = None

def load_embedding_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _MODEL

def embed_texts(texts):
    """
    Accepts a list of strings and returns a numpy array of embeddings.
    """
    model = load_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings
