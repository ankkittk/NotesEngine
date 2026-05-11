import os

from dotenv import load_dotenv
from openai import OpenAI

from ..core.config import GENERATION_BASE_URL, GENERATION_MODEL_NAME

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url=GENERATION_BASE_URL
)

QUERY_REWRITE_PROMPT = """
You are a retrieval query optimizer.

Rewrite the user's query into a concise but information-rich search query for academic notes retrieval.

Rules:
- Preserve the user's intent.
- Expand vague phrasing, acronyms, and shorthand when useful.
- Add likely domain terms if they help retrieval.
- Do not answer the question.
- Do not add explanations.
- Return only the rewritten query.
"""


def _clean_rewrite(text: str) -> str:
    text = (text or "").strip()

    if not text:
        return ""

    if "```" in text:
        parts = [part.strip() for part in text.split("```") if part.strip()]
        if parts:
            text = parts[0]

    if text.lower().startswith("rewritten retrieval query:"):
        text = text.split(":", 1)[1].strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        text = lines[0]

    return text.strip('"\'')


def rewrite_query(query: str) -> str:
    query = (query or "").strip()

    if not query:
        return query

    prompt = f"""{QUERY_REWRITE_PROMPT}

User query:
{query}

Rewritten retrieval query:
"""

    try:
        response = client.chat.completions.create(
            model=GENERATION_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=128,
        )
        rewritten = response.choices[0].message.content
        cleaned = _clean_rewrite(rewritten)
        return cleaned or query
    except Exception:
        return query
