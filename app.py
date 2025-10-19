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
    "A chatbot that answers using your **local knowledge base** (RAG) and **Google Gemini**. "
    "If context is missing or irrelevant, it automatically performs a **live web search**."
)

with st.sidebar:
    st.header("Settings ⚙️")
    mode = st.radio("Response mode:", ["Concise", "Detailed"])
    k = st.slider("Top K:", 1, 10, TOP_K)
    show_context = st.checkbox("Show retrieved context", True)

    uploaded = st.file_uploader("Upload a document (PDF or TXT)", type=["pdf", "txt"])
    if uploaded:
        save_path = os.path.join("data", "knowledge_base", uploaded.name)
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"✅ Saved to `{save_path}`. Run `python build_embeddings.py` locally to rebuild the index.")

st.write("---")
query = st.text_area("💬 Ask anything:", height=120)

if st.button("Get Answer"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        try:
            index, metadata = load_faiss_index()
            retrieved, scores = query_faiss(query, top_k=k, index=index, metadata=metadata)

            avg_score = sum(scores) / len(scores) if scores else 0
            similarity_threshold = 0.25  # you can tune this (lower = stricter)

            if not retrieved or avg_score < similarity_threshold:
                st.info("⚠️ Local context missing or irrelevant. Using Gemini Web Search...")
                with st.spinner("🌍 Searching the web..."):
                    answer = web_search_answer(query)
                st.markdown("### 🌍 Answer (via Web Search)")
                st.write(answer)
            else:
                prompt = build_prompt(retrieved, query, mode=mode)
                if show_context:
                    with st.expander("📚 Retrieved context (Top Chunks)"):
                        for i, (r, s) in enumerate(zip(retrieved, scores), 1):
                            st.markdown(f"**Chunk {i}** — similarity: `{s:.3f}`")
                            st.write(r)
                with st.spinner("🤖 Generating answer with Gemini..."):
                    answer = generate_gemini_response(prompt)
                st.markdown("### 💡 Answer (via Knowledge Base)")
                st.write(answer)
        except FileNotFoundError as fe:
            st.error(str(fe) + "\n\nRun `python build_embeddings.py` to create FAISS index.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
