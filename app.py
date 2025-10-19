# app.py
import os
import streamlit as st
from config.config import TOP_K
from utils.rag_utils import load_faiss_index, query_faiss
from utils.prompt_builder import build_prompt
from models.llm import generate_gemini_response
from utils.web_search_utils import web_search_answer

st.set_page_config(page_title="SmartContext Chatbot", layout="wide")

st.title("🌐 SmartContext Chatbot — Gemini + RAG + Web Search")
st.markdown(
    "Answer using your **local knowledge base** (RAG). "
    "If local context is insufficient, Gemini searches the web automatically."
)

with st.sidebar:
    st.header("Settings ⚙️")
    mode = st.radio("Response mode:", ["Concise", "Detailed"])
    k = st.slider("Top K:", 1, 10, TOP_K)
    show_context = st.checkbox("Show retrieved context", True)
    similarity_threshold = st.slider("Similarity threshold:", 0.0, 1.0, 0.5, 0.05)

    st.markdown("---")
    st.markdown("### 📁 Upload Documents")
    uploaded = st.file_uploader("Upload a document (PDF or TXT)", type=["pdf", "txt"])
    if uploaded:
        os.makedirs(os.path.join("data", "knowledge_base"), exist_ok=True)
        save_path = os.path.join("data", "knowledge_base", uploaded.name)
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"✅ Saved to `{save_path}`. Run `python build_embeddings.py` locally to rebuild the index.")

st.write("---")
query = st.text_area("💬 Ask anything:", height=120)

if st.button("Get Answer", type="primary"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        try:
            # Try to load FAISS index
            index, metadata = load_faiss_index()
            retrieved, scores = query_faiss(query, top_k=k, index=index, metadata=metadata)
            avg_score = sum(scores)/len(scores) if scores else 0

            # Check if local context is sufficient
            if not retrieved or avg_score < similarity_threshold:
                st.info("⚠️ Local context insufficient. Using Gemini with web search...")
                
                with st.spinner("🌍 Searching the web and generating answer..."):
                    answer = web_search_answer(query)
                
                st.markdown("### 🌍 Answer (Web-aware Gemini)")
                st.write(answer)
                
                # Show why web search was triggered
                with st.expander("ℹ️ Why web search?"):
                    if not retrieved:
                        st.write("No relevant documents found in local knowledge base.")
                    else:
                        st.write(f"Average similarity score ({avg_score:.3f}) below threshold ({similarity_threshold}).")
                        st.write("Retrieved documents were not relevant enough.")
            else:
                # Use local RAG
                prompt = build_prompt(retrieved, query, mode=mode)
                
                if show_context:
                    with st.expander("📚 Retrieved context (Top Chunks)"):
                        for i, (r, s) in enumerate(zip(retrieved, scores), 1):
                            st.markdown(f"**Chunk {i}** — similarity: `{s:.3f}`")
                            st.text_area(f"Content {i}", r, height=100, key=f"chunk_{i}")
                
                with st.spinner("🤖 Generating answer with local knowledge..."):
                    answer = generate_gemini_response(prompt)
                
                st.markdown("### 💡 Answer (Local Knowledge Base)")
                st.write(answer)
                
                # Show confidence
                st.info(f"✅ Answer based on local documents (avg similarity: {avg_score:.3f})")
                
        except FileNotFoundError as fe:
            st.error(str(fe) + "\n\n**Action required:** Run `python build_embeddings.py` to create FAISS index.")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            
            # Fallback to web search on error
            if st.button("Try web search instead"):
                with st.spinner("🌍 Falling back to web search..."):
                    answer = web_search_answer(query)
                st.markdown("### 🌍 Fallback Answer (Web Search)")
                st.write(answer)

# Add footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "💡 Tip: Adjust similarity threshold in sidebar to control when web search is triggered"
    "</div>",
    unsafe_allow_html=True
)