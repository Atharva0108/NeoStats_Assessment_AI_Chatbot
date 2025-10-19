# models/llm.py
import google.generativeai as genai
from config.config import GOOGLE_API_KEY, GEMINI_MODEL, GEMINI_MAX_TOKENS

# Configure Gemini API
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set. Please add it to your .env or Streamlit secrets.")

genai.configure(api_key=GOOGLE_API_KEY)

def generate_gemini_response(prompt: str, max_tokens: int = GEMINI_MAX_TOKENS, temperature: float = 0.2):
    """
    Generate a text response from Google Gemini using the current stable API.
    """
    try:
        # Initialize model (use 'gemini-1.5-flash' or 'gemini-1.5-pro' depending on availability)
        model = genai.GenerativeModel(GEMINI_MODEL)

        # Generate content
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature
            )
        )

        # Return plain text
        return response.text.strip()
    except Exception as e:
        return f"LLM generation error: {e}"
