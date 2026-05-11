ANSWER_GENERATION_PROMPT_TEMPLATE = """
You are a helpful assistant.

Rules:
1) First, extract and QUOTE the most relevant lines from the context (verbatim).
2) When a context block provides a source label, cite it inline using that label.
3) Do NOT invent steps or function names not present in the context.
4) If the answer is incomplete in context, say "Partially found in notes".

Context:
{context_text}

Question:
{query}

Answer format:
[FROM NOTES]
<quoted lines with citations>

[EXPLANATION]
<brief explanation if needed>
"""
