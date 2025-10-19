````markdown
# SmartContext Chatbot — Gemini 2.5 Flash + RAG

**SmartContext Chatbot** is a context-aware AI chatbot that combines **Google Gemini 2.5 Flash** LLM with **RAG (Retrieval-Augmented Generation)** using local documents. It allows you to ask questions against your knowledge base (PDF/TXT) and get **accurate, context-driven answers** via a Streamlit interface.

**Live Demo:** [Open in Streamlit](<YOUR_STREAMLIT_URL>)

---

## Features

- Context-aware responses using FAISS + embeddings  
- Choose **Concise** or **Detailed** answer modes  
- Display retrieved context for transparency  
- Upload documents via UI (local rebuild required for persistence)  

---

## Setup (Local)

1. **Clone repo & create virtualenv**

```bash
git clone 
cd smartcontext-chatbot
python3 -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\Activate.ps1
````

2. **Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. **Add Google Gemini API key**

Create a `.env` file at project root:

```
GOOGLE_API_KEY=your_google_ai_studio_api_key_here
```

4. **Add documents**

Place your PDFs/TXT in:

```
data/knowledge_base/
```

5. **Build embeddings & FAISS index**

```bash
python build_embeddings.py --docs data/knowledge_base --out data/embeddings
```

6. **Run the app**

```bash
streamlit run app.py
```

---

## Deployment (Streamlit Cloud)

1. Push repository (including `data/knowledge_base` and `data/embeddings`) to GitHub
2. Create new app on **Streamlit Cloud** linked to your repo
3. Add secret:

```
GOOGLE_API_KEY=your_google_ai_studio_api_key_here
```

4. Deploy. The app will load FAISS index and Gemini 2.5 Flash model automatically.

---

## Notes

* Uploaded files in the UI **do not persist** across sessions; rebuild embeddings locally for permanent updates.
* FAISS index (`faiss_index.bin`) and metadata (`metadata.json`) can be committed to GitHub; not secrets.
* Adjust **Top-K retrieval** and **chunk size** in `config/config.py` as needed.

