# app.py
import os
import streamlit as st
from config.config import TOP_K
from utils.rag_utils import load_faiss_index, query_faiss
from utils.prompt_builder import build_prompt
from models.llm import generate_gemini_response
from utils.web_search_utils import web_search_answer  # ✅ NEW IMPORT

st.set_page_config(page_title="SmartContext Chatbot (Gemini + RAG + Web Search)", layout="wide")

st.title("🌐 SmartContext Chatbot — Gemini + RAG + Web Search")
st.markdown(
    "A context-aware chatbot using your **local knowledge base (RAG)** and **Google Gemini**. "
    "If no relevant context is found locally, it will automatically **search the web** for real-time information."
)

# Sidebar configuration
with st.sidebar:
    st.header("Settings ⚙️")
    mode = st.radio("Response mode:", ["Concise", "Detailed"])
    st.write("Top-K retrieval")
    k = st.slider("Top K:", min_value=1, max_value=10, value=TOP_K)
    show_context = st.checkbox("Show retrieved context", value=True)

    # Optional file upload for knowledge base
    uploaded = st.file_uploader("Upload a document (PDF or TXT)", type=["pdf", "txt"])
    if uploaded is not None:
        save_path = os.path.join("data", "knowledge_base", uploaded.name)
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"✅ Saved to `{save_path}`. Rebuild embeddings locally for persistence (`python build_embeddings.py`).")

# Chat section
st.write("---")
query = st.text_area("💬 Ask anything:", height=120)

if st.button("Get Answer"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        try:
            # 1. Load FAISS + metadata
            index, metadata = load_faiss_index()

            # 2. Retrieve top-k chunks
            retrieved = query_faiss(query, top_k=k, index=index, metadata=metadata)

            # 3. Check if we have context; otherwise use web search fallback
            if not any(retrieved):
                st.info("⚠️ No relevant context found in local knowledge base. Using Gemini Web Search...")
                with st.spinner("🔍 Searching the web..."):
                    answer = web_search_answer(query)
                st.markdown("### 🌍 Answer (via Web Search)")
                st.write(answer)
            else:
                # 4. Build prompt with context
                prompt = build_prompt(retrieved, query, mode=mode)

                # 5. Optionally show context
                if show_context:
                    with st.expander("📚 Retrieved context (Top Chunks)"):
                        for i, r in enumerate(retrieved, start=1):
                            st.markdown(f"**Chunk {i}:**")
                            st.write(r)

                # 6. Call Gemini for context-based response
                with st.spinner("🤖 Generating answer with Gemini..."):
                    answer = generate_gemini_response(prompt)

                st.markdown("### 💡 Answer (via Knowledge Base)")
                st.write(answer)

        except FileNotFoundError as fe:
            st.error(str(fe) + "\n\nRun `python build_embeddings.py` locally to create FAISS index and metadata.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
