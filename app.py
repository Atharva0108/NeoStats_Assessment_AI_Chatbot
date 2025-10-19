# app.py
import streamlit as st
from config.config import TOP_K, FAISS_INDEX_PATH, METADATA_PATH
from utils.rag_utils import load_faiss_index, query_faiss
from utils.prompt_builder import build_prompt
from models.llm import generate_gemini_response

st.set_page_config(page_title="SmartContext Chatbot (Gemini + RAG)", layout="wide")

st.title("SmartContext Chatbot — Gemini + RAG")
st.markdown("A context-aware chatbot using local knowledge base (RAG) + Google Gemini. Choose response mode and ask questions.")

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Response mode:", ["Concise", "Detailed"])
    st.write("Top-K retrieval")
    k = st.slider("Top K:", min_value=1, max_value=10, value=TOP_K)
    show_context = st.checkbox("Show retrieved context", value=True)
    uploaded = st.file_uploader("Upload a document to knowledge base (PDF or TXT)", type=["pdf", "txt"])
    if uploaded is not None:
        # Save to knowledge base folder - note: in Streamlit Cloud this will not persist across sessions unless you push to repo
        import os
        save_path = os.path.join("data", "knowledge_base", uploaded.name)
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"Saved to {save_path}. You should rebuild embeddings locally and push to GitHub for persistent use.")

st.write("---")
query = st.text_area("Ask anything from the knowledge base:", height=120)
if st.button("Get Answer"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        try:
            # 1. Load FAISS + metadata
            index, metadata = load_faiss_index()
            # 2. Retrieve top-k chunks
            retrieved = query_faiss(query, top_k=k, index=index, metadata=metadata)
            # 3. Build prompt
            prompt = build_prompt(retrieved, query, mode=mode)
            # 4. Show context if desired
            if show_context:
                with st.expander("Retrieved context (top chunks)"):
                    for i, r in enumerate(retrieved, start=1):
                        st.markdown(f"**Chunk {i}:**")
                        st.write(r)
            # 5. Call Gemini
            with st.spinner("Generating answer with Gemini..."):
                answer = generate_gemini_response(prompt, max_tokens=None)
            st.markdown("### Answer")
            st.write(answer)
        except FileNotFoundError as fe:
            st.error(str(fe) + "\n\nYou need to run `python build_embeddings.py` locally and push data/embeddings to repo.")
        except Exception as e:
            st.error(f"Error: {e}")
