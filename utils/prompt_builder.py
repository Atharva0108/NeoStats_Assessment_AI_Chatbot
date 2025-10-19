# utils/prompt_builder.py
from config.config import TOP_K

def build_prompt(retrieved_chunks, user_query, mode="Detailed"):
    """
    Constructs a prompt that instructs Gemini to answer using the retrieved context.
    """

    # Join retrieved chunks with separators for clarity, label each source
    ctx = "\n\n---\n\n".join(retrieved_chunks) if retrieved_chunks else ""

    if mode.lower().startswith("conc"):
        instruction = "Provide a short, direct, and clear answer. Keep it concise (3-4 sentences)."
    else:
        instruction = "Provide a detailed, structured explanation. Use examples where helpful, and include steps or bullet points if applicable."

    prompt = f"""{instruction}

Context (top {len(retrieved_chunks)} relevant excerpts):
{ctx}

User Question:
{user_query}

Using ONLY the context above (and your general knowledge if context is missing), answer the user's question.
If the answer is not present in the context, explicitly say \"I don't have enough context to answer that fully; consider checking these sources or ask follow-up questions.\"
"""
    return prompt
