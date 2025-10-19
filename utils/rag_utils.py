# utils/rag_utils.py
import os
import json
import faiss
import numpy as np
from tqdm import tqdm
from PyPDF2 import PdfReader
from models.embeddings import embed_texts
from config.config import CHUNK_SIZE, CHUNK_OVERLAP, FAISS_INDEX_PATH, METADATA_PATH

def load_text_from_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        text = []
        with open(path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text.append(page_text)
        return "\n".join(text)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Simple word-based chunking. Returns list of strings.
    """
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

def build_faiss_index(docs_dir: str, save_dir: str):
    """
    Walks docs_dir, builds embeddings for chunks, creates FAISS index, and saves index + metadata.
    """
    os.makedirs(save_dir, exist_ok=True)
    all_texts = []
    metadata = []

    print("Loading documents and chunking...")
    for root, _, files in os.walk(docs_dir):
        for fname in files:
            if fname.startswith("."):
                continue
            path = os.path.join(root, fname)
            text = load_text_from_file(path)
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                all_texts.append(chunk)
                metadata.append({
                    "source": path,
                    "chunk_index": i,
                    "text": chunk[:1000]  # store truncated preview; full text is in chunk itself in all_texts
                })

    if not all_texts:
        raise ValueError("No documents found in docs_dir. Put files inside data/knowledge_base/")

    print(f"Generating embeddings for {len(all_texts)} chunks...")
    embeddings = embed_texts(all_texts)  # numpy array shape (N, D)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    print("Adding embeddings to FAISS index...")
    index.add(embeddings.astype("float32"))

    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved FAISS index to {FAISS_INDEX_PATH} and metadata to {METADATA_PATH}")
    return FAISS_INDEX_PATH, METADATA_PATH

def load_faiss_index(index_path=FAISS_INDEX_PATH, metadata_path=METADATA_PATH):
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError("FAISS index or metadata not found. Run build_embeddings.py first.")
    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

def query_faiss(query: str, top_k: int, index=None, metadata=None):
    """
    Returns top_k metadata items and their texts for a query string.
    """
    if index is None or metadata is None:
        index, metadata = load_faiss_index()
    q_emb = embed_texts([query]).astype("float32")
    D, I = index.search(q_emb, top_k)
    results = []
    for i in I[0]:
        if i < len(metadata):
            # fetch the chunk text stored in metadata['text'] (truncated preview) OR you could store full chunks separately
            results.append(metadata[i].get("text", ""))
    return results
