import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


def generate_answer(query, contexts):
    context_text = "\n\n".join(contexts)

    prompt = f"""
You are a helpful assistant.

Use ONLY the provided context to answer the question.
If the answer is not present, say "Not found in notes".

Context:
{context_text}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content
