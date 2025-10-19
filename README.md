# 🌐 SmartContext Chatbot — Gemini 2.0 Flash + RAG + Web Search

SmartContext Chatbot is an intelligent AI assistant that combines **Google Gemini 2.0 Flash** with **RAG (Retrieval-Augmented Generation)** and **real-time web search**. Ask questions against your local knowledge base (PDF/TXT) and get accurate answers. When local context is insufficient, the bot automatically searches the web for up-to-date information.

**Live Demo:** [https://neostatsaichatbot.streamlit.app/](https://neostatsaichatbot.streamlit.app/)

---

## ✨ Features

- 🧠 **Hybrid Intelligence**: Local RAG + Web Search fallback
- 📚 **Multi-Source Retrieval**: FAISS vector search with semantic similarity
- 🌍 **Real-Time Web Search**: Google Search grounding via Gemini 2.0 Flash
- 🎯 **Adaptive Responses**: Choose between Concise or Detailed answer modes
- 🔍 **Transparency**: View retrieved context chunks and similarity scores
- 📤 **Easy Upload**: Add documents via UI (rebuild required for persistence)
- ⚙️ **Configurable**: Adjust similarity threshold, Top-K retrieval, and more

---

## 🚀 How It Works

1. **Local RAG First**: Searches your knowledge base using FAISS + embeddings
2. **Smart Fallback**: If similarity score is below threshold → triggers web search
3. **Web-Aware Responses**: Gemini 2.0 Flash uses Google Search grounding for current info
4. **Safety Handled**: Robust error handling with relaxed safety filters

---

## 📋 Prerequisites

- Python 3.8+
- Google AI Studio API key ([Get one here](https://makersuite.google.com/app/apikey))
- Basic knowledge of terminal/command line

---

## 🛠️ Setup (Local)

### 1. Clone and Setup Environment

```bash
git clone https://github.com/<your-username>/smartcontext-chatbot.git
cd smartcontext-chatbot
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate              # macOS/Linux
# OR
.\venv\Scripts\Activate.ps1           # Windows PowerShell
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Required packages:**
```txt
streamlit>=1.31.0
google-generativeai>=0.3.0
faiss-cpu>=1.7.4
PyPDF2>=3.0.0
python-dotenv>=1.0.0
sentence-transformers>=2.2.2
```

### 3. Configure API Key

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_ai_studio_api_key_here
```

### 4. Add Knowledge Base Documents

Place your PDF or TXT files in:

```
data/knowledge_base/
```

**Example structure:**
```
data/
├── knowledge_base/
│   ├── document1.pdf
│   ├── document2.txt
│   └── reference.pdf
└── embeddings/
    ├── faiss_index.bin
    └── metadata.json
```

### 5. Build FAISS Index

```bash
python build_embeddings.py --docs data/knowledge_base --out data/embeddings
```

**What this does:**
- Chunks documents into semantic segments
- Generates embeddings using `sentence-transformers`
- Creates FAISS index for fast similarity search
- Saves metadata for document tracking

### 6. Run the Application

```bash
streamlit run app.py
```

Open browser at: `http://localhost:8501`

---

## ☁️ Deployment (Streamlit Cloud)

### Step 1: Prepare Repository

1. **Build embeddings locally first** (important!)
2. **Commit all files including:**
   - `data/embeddings/faiss_index.bin`
   - `data/embeddings/metadata.json`
   - `data/knowledge_base/` (your documents)

```bash
git add .
git commit -m "Add embeddings and knowledge base"
git push origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Click **"New app"**
3. Connect your GitHub repository
4. Set main file: `app.py`
5. Add **Secrets** (click Advanced settings):

```toml
GOOGLE_API_KEY = "your_google_ai_studio_api_key_here"
```

6. Click **Deploy** 🚀

### Step 3: Verify Deployment

- App should load FAISS index automatically
- Test with a query from your knowledge base
- Test with a query requiring web search (e.g., "latest news about AI")

---

## 🎛️ Configuration

Edit `config/config.py` to customize:

```python
# Model settings
GEMINI_MODEL = "gemini-2.0-flash-exp"  # or "gemini-1.5-flash"
GEMINI_MAX_TOKENS = 2048
TEMPERATURE = 0.2

# RAG settings
TOP_K = 5                              # Number of chunks to retrieve
CHUNK_SIZE = 500                       # Characters per chunk
CHUNK_OVERLAP = 50                     # Overlap between chunks
SIMILARITY_THRESHOLD = 0.5             # Web search trigger threshold
```

---

## 📖 Usage Tips

### Local Knowledge Base Queries
✅ **Good for:** Company docs, technical manuals, research papers
```
"What is the refund policy?"
"Explain the architecture from the design doc"
```

### Web Search Queries
✅ **Good for:** Current events, real-time data, recent information
```
"What's the latest iPhone model?"
"Current weather in Mumbai"
"Recent AI developments in 2025"
```

### Adjust Similarity Threshold
- **Lower (0.3-0.4)**: More reliant on local documents
- **Higher (0.6-0.7)**: Triggers web search more often
- Use sidebar slider to experiment

---

## 🔧 Troubleshooting

### Error: `response.text` quick accessor failed
✅ **Fixed!** Updated code handles safety blocks and empty responses gracefully.

### Web search not working
- Ensure using `gemini-2.0-flash-exp` model
- Check API key has web search permissions
- Verify internet connection

### FAISS index not found
```bash
python build_embeddings.py --docs data/knowledge_base --out data/embeddings
```

### Low-quality responses
- Add more relevant documents to knowledge base
- Rebuild embeddings after adding documents
- Adjust `CHUNK_SIZE` in config (try 300-700)
- Lower similarity threshold

---

## 📁 Project Structure

```
smartcontext-chatbot/
├── app.py                          # Main Streamlit app
├── build_embeddings.py             # FAISS index builder
├── requirements.txt                # Python dependencies
├── .env                            # API keys (not committed)
├── config/
│   └── config.py                   # Configuration settings
├── models/
│   └── llm.py                      # Gemini API wrapper
├── utils/
│   ├── rag_utils.py                # FAISS search utilities
│   ├── web_search_utils.py         # Web search with grounding
│   └── prompt_builder.py           # Prompt templates
└── data/
    ├── knowledge_base/             # Your documents (PDF/TXT)
    └── embeddings/                 # FAISS index + metadata
        ├── faiss_index.bin
        └── metadata.json
```

---

## 🔐 Security Notes

- ✅ **FAISS index and metadata** can be committed to GitHub (not sensitive)
- ❌ **Never commit** `.env` or API keys
- ✅ Use Streamlit Cloud secrets for production
- ✅ Safety filters configured to `BLOCK_NONE` for flexibility (adjust per use case)

---

## 🆕 Recent Updates

### v2.0 (Current)
- ✨ Added real-time web search with Google Search grounding
- 🔧 Fixed safety filter blocking issues
- 🎯 Smart fallback system (local RAG → web search)
- 📊 Similarity threshold slider in UI
- 🛡️ Robust error handling for all response types

### v1.0
- Initial release with RAG-only functionality

---

## 📝 Important Notes

- **Uploaded files via UI**: Do not persist across sessions. For permanent updates, add to `data/knowledge_base/` locally and rebuild embeddings.
- **Model updates**: Gemini 2.0 Flash may be replaced with newer models. Update `GEMINI_MODEL` in config as needed.
- **Rate limits**: Google AI Studio has rate limits. Consider upgrading for production use.
- **Web search availability**: Grounding feature requires Gemini 2.0 Flash (or newer) and may have regional restrictions.

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📄 License

MIT License - feel free to use this project for personal or commercial purposes.

---

## 🙏 Acknowledgments

- Google Gemini API
- Streamlit
- FAISS by Facebook Research
- Sentence Transformers

---

## 📞 Support

Issues? Questions? Open an issue on GitHub or contact via the live demo.

**Happy chatting!** 🚀