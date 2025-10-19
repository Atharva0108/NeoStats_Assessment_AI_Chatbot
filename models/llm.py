# models/llm.py
import google.generativeai as genai
from config.config import GOOGLE_API_KEY, GEMINI_MODEL, GEMINI_MAX_TOKENS

# Configure Gemini API
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set. Please add it to your .env or Streamlit secrets.")

genai.configure(api_key=GOOGLE_API_KEY)

def generate_gemini_response(prompt: str, max_tokens: int = GEMINI_MAX_TOKENS, temperature: float = 0.2):
    """
    Generate a text response from Gemini safely with proper error handling.
    Returns fallback message if no text is returned.
    """
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        # Relaxed safety settings
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature
            ),
            safety_settings=safety_settings
        )

        # Comprehensive response handling
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            
            # Check finish reason
            if candidate.finish_reason == 2:  # SAFETY
                return "⚠️ Response blocked by safety filters. Please rephrase your question."
            elif candidate.finish_reason == 3:  # RECITATION
                return "⚠️ Response blocked due to recitation concerns."
            elif candidate.finish_reason == 4:  # OTHER
                return "⚠️ Response generation stopped for unknown reasons."
            
            # Try to extract text
            if hasattr(response, 'text') and response.text:
                return response.text.strip()
            elif candidate.content and candidate.content.parts:
                # Extract text from parts
                text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text')]
                if text_parts:
                    return " ".join(text_parts).strip()
        
        return "⚠️ Gemini could not generate a response. Try rephrasing your question."

    except ValueError as ve:
        # This catches the "response.text quick accessor" error
        return f"⚠️ Response generation failed. Please try a different question."
    except Exception as e:
        return f"⚠️ Gemini API error: {str(e)}"