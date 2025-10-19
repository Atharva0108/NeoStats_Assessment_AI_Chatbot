# build_embeddings.py
import argparse
from utils.rag_utils import build_faiss_index

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index from documents")
    parser.add_argument("--docs", type=str, default="data/knowledge_base", help="Directory with knowledge base files")
    parser.add_argument("--out", type=str, default="data/embeddings", help="Output directory for FAISS index and metadata")
    args = parser.parse_args()
    build_faiss_index(args.docs, args.out)
