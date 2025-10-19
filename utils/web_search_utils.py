# utils/web_search_utils.py
import google.generativeai as genai
from config.config import GOOGLE_API_KEY

genai.configure(api_key=GOOGLE_API_KEY)

def web_search_answer(query: str):
    """
    Use Gemini 2.0 Flash with Google Search grounding to fetch real-time information.
    Falls back to regular Gemini if grounding fails.
    """
    try:
        # Use Gemini 2.0 Flash with grounding
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            tools="google_search_retrieval"  # Enable Google Search grounding
        )
        
        # Relaxed safety settings
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        response = model.generate_content(
            query,
            safety_settings=safety_settings
        )
        
        # Safe text extraction
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            
            # Check finish reason
            if candidate.finish_reason == 2:  # SAFETY block
                return "⚠️ Response blocked by safety filters. Please rephrase your question."
            
            # Check if response has text
            if hasattr(response, 'text') and response.text:
                return response.text.strip()
            elif candidate.content and candidate.content.parts:
                # Try to extract text from parts
                text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text')]
                if text_parts:
                    return " ".join(text_parts).strip()
        
        return "⚠️ No response generated. Please try rephrasing your question."
        
    except Exception as e:
        # Fallback to regular Gemini without grounding
        return fallback_gemini_answer(query, str(e))


def fallback_gemini_answer(query: str, original_error: str = ""):
    """
    Fallback to regular Gemini model if grounding fails.
    """
    try:
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        enhanced_query = f"""Answer this question based on your knowledge: {query}
        
If you need current information, please state that you cannot access real-time data."""
        
        response = model.generate_content(
            enhanced_query,
            safety_settings=safety_settings,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=2048
            )
        )
        
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            
            if candidate.finish_reason == 2:
                return "⚠️ Response blocked by safety filters. Please rephrase your question."
            
            if hasattr(response, 'text') and response.text:
                return response.text.strip()
        
        return f"⚠️ Could not generate response. Original error: {original_error}"
        
    except Exception as e:
        return f"⚠️ Both web search and fallback failed. Error: {str(e)}"