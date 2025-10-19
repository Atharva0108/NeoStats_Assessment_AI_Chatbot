# utils/helper.py
import streamlit as st
from models.embeddings import load_embedding_model

@st.cache_resource
def get_embedding_model():
    return load_embedding_model()

def pretty_sources(metadata_items):
    """
    metadata_items: list of dicts or strings; returns a formatted string
    """
    lines = []
    for i, m in enumerate(metadata_items):
        if isinstance(m, dict):
            lines.append(f"{i+1}. {m.get('source', 'unknown')} (chunk {m.get('chunk_index', '-')})")
        else:
            lines.append(f"{i+1}. {str(m)[:120]}...")
    return "\n".join(lines)
