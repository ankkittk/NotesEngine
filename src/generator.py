import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


def generate_answer(query, contexts):
    context_text = "\n\n".join(contexts) if contexts else "No relevant context found."

    prompt = f"""
You are a helpful assistant.

Rules:
1) First, extract and QUOTE the most relevant lines from the context (verbatim).
2) Then, you may add a short explanation ONLY if needed.
3) Do NOT invent steps or function names not present in the context.
4) If the answer is incomplete in context, say "Partially found in notes".

Context:
{context_text}

Question:
{query}

Answer format:
[FROM NOTES]
<quoted lines>

[EXPLANATION]
<brief explanation if needed>
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()
