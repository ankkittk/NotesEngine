import os
from dotenv import load_dotenv
from openai import OpenAI

from ..core.config import (
    GENERATION_BASE_URL,
    GENERATION_MODEL_NAME,
    GENERATION_TEMPERATURE,
)
from .prompts import ANSWER_GENERATION_PROMPT_TEMPLATE

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url=GENERATION_BASE_URL
)


def generate_answer(query, contexts):
    context_text = "\n\n".join(contexts) if contexts else "No relevant context found."

    prompt = ANSWER_GENERATION_PROMPT_TEMPLATE.format(
        context_text=context_text,
        query=query
    )

    response = client.chat.completions.create(
        model=GENERATION_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=GENERATION_TEMPERATURE
    )

    return response.choices[0].message.content.strip()
