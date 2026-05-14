import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from ..core.config import (
    GENERATION_BASE_URL,
    GENERATION_MODEL_NAME,
)

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url=GENERATION_BASE_URL
)

QUERY_ANALYZER_PROMPT = """
You are an intelligent query analysis system for an academic Agentic RAG assistant.

Your task is to analyze the user's query and determine:
1. query_type
2. whether clarification is needed
3. recommended retrieval branch
4. confidence score
5. short reasoning

Allowed query_type values:
- direct
- ambiguous
- comparative
- conversational_followup
- out_of_domain

Allowed recommended_branch values:
- direct_retrieval
- clarification_candidate
- comparative_retrieval
- out_of_domain_response

Rules:
- Very short vague queries may be ambiguous.
- Comparison-style questions are comparative.
- Follow-up conversational references like "compare it with svm" are conversational_followup.
- Queries unrelated to academic knowledge retrieval are out_of_domain.
- Return STRICT JSON only.
- No markdown.
"""

DEFAULT_ANALYSIS = {
    "query_type": "direct",
    "needs_clarification": False,
    "confidence": 0.5,
    "recommended_branch": "direct_retrieval",
    "reason": "Fallback analysis"
}


def _safe_json_load(text: str):
    text = (text or "").strip()

    if not text:
        return None

    if "```json" in text:
        text = text.split("```json")[-1]

    if "```" in text:
        text = text.split("```")[0]

    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        return None


def analyze_query(query: str) -> dict:
    query = (query or "").strip()

    if not query:
        result = dict(DEFAULT_ANALYSIS)
        result["query_type"] = "ambiguous"
        result["needs_clarification"] = True
        result["recommended_branch"] = "clarification_candidate"
        result["reason"] = "Empty query"
        return result

    prompt = f"""
{QUERY_ANALYZER_PROMPT}

User Query:
{query}

JSON Response:
"""

    try:
        response = client.chat.completions.create(
            model=GENERATION_MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,
            max_tokens=400,
        )

        text = response.choices[0].message.content
        print("\n--- RAW ANALYZER RESPONSE ---\n")
        print(text)

        parsed = _safe_json_load(text)

        if not parsed:
            return dict(DEFAULT_ANALYSIS)

        result = dict(DEFAULT_ANALYSIS)

        result["query_type"] = parsed.get(
            "query_type",
            result["query_type"]
        )

        result["needs_clarification"] = parsed.get(
            "needs_clarification",
            parsed.get(
                "clarification_needed",
                result["needs_clarification"]
            )
        )

        result["confidence"] = parsed.get(
            "confidence",
            parsed.get(
                "confidence_score",
                result["confidence"]
            )
        )

        result["recommended_branch"] = parsed.get(
            "recommended_branch",
            result["recommended_branch"]
        )

        result["reason"] = parsed.get(
            "reason",
            parsed.get(
                "reasoning",
                result["reason"]
            )
        )

        return result

    except Exception:
        return dict(DEFAULT_ANALYSIS)
