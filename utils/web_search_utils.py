# utils/web_search_utils.py
import google.generativeai as genai
from config.config import GOOGLE_API_KEY, GEMINI_MODEL

genai.configure(api_key=GOOGLE_API_KEY)

def web_search_answer(query: str):
    """
    Use Gemini with web grounding to fetch real-time information
    when local RAG context doesn't have relevant data.
    """
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            query,
            # this enables web grounding if supported by the model
            tools=[{"name": "google_search"}],
            safety_settings={
                "HARASSMENT": "BLOCK_NONE",
                "HATE": "BLOCK_NONE",
                "SEXUAL": "BLOCK_NONE",
                "DANGEROUS": "BLOCK_NONE",
            }
        )
        return response.text.strip()
    except Exception as e:
        return f"Web search failed: {e}"
