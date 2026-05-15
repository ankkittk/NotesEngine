import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

from ..core.config import GENERATION_BASE_URL, GENERATION_MODEL_NAME

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url=GENERATION_BASE_URL,
)

DEFAULT_TOPIC_INFO = {
    "main_topic": "",
    "comparison_topics": [],
    "query_kind": "unknown",
}

_COMPARATIVE_PATTERNS = [
    r"(?i)^compare\s+(?P<a>.+?)\s+(?:and|with|vs\.?|versus|to)\s+(?P<b>.+?)\s*$",
    r"(?i)^difference between\s+(?P<a>.+?)\s+and\s+(?P<b>.+?)\s*$",
    r"(?i)^compare\s+(?P<a>.+?)\s+to\s+(?P<b>.+?)\s*$",
]

_DIRECT_PREFIXES = [
    "what is ",
    "what are ",
    "explain ",
    "define ",
    "describe ",
    "tell me about ",
    "give me ",
    "write about ",
    "what does ",
    "who is ",
    "how does ",
]

_FOLLOWUP_PREFIXES = [
    "applications of ",
    "advantages of ",
    "limitations of ",
    "benefits of ",
    "uses of ",
    "disadvantages of ",
    "examples of ",
    "summary of ",
    "meaning of ",
    "purpose of ",
    "features of ",
]


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def _clean_topic(text: str) -> str:
    text = _normalize_text(text)
    text = text.strip('"\''"“”`")
    text = re.sub(r"^[\s:;,\-–—]+", "", text)
    text = re.sub(r"[\s:;,\-–—]+$", "", text)
    text = re.sub(r"^(the|a|an)\s+", "", text, flags=re.I)
    return _normalize_text(text)


def _safe_json_load(text: str):
    text = (text or "").strip()

    if "```json" in text:
        text = text.split("```json", 1)[1]

    if "```" in text:
        text = text.split("```", 1)[0]

    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        return None


def _normalize_comparison_topics(value):
    if isinstance(value, list):
        return [_clean_topic(v) for v in value if _clean_topic(v)]

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []

        parts = re.split(r"\s*(?:,| and | vs\.? | versus | with | to )\s*", raw, flags=re.I)
        cleaned = [_clean_topic(p) for p in parts if _clean_topic(p)]
        return cleaned

    return []


def _heuristic_topic_info(query: str, branch: str = "") -> dict:
    query = _normalize_text(query)

    if not query:
        return dict(DEFAULT_TOPIC_INFO)

    lower = query.lower()

    if branch == "out_of_domain_response":
        return dict(DEFAULT_TOPIC_INFO)

    if branch == "clarification_candidate" and re.search(r"\b(it|this|that|them|those|these|its|their)\b", lower):
        return dict(DEFAULT_TOPIC_INFO)

    for pattern in _COMPARATIVE_PATTERNS:
        match = re.match(pattern, query)
        if match:
            a = _clean_topic(match.group("a"))
            b = _clean_topic(match.group("b"))
            comparison_topics = [t for t in [a, b] if t]
            return {
                "main_topic": comparison_topics[0] if comparison_topics else "",
                "comparison_topics": comparison_topics,
                "query_kind": "comparative",
            }

    for prefix in _DIRECT_PREFIXES:
        if lower.startswith(prefix):
            topic = _clean_topic(query[len(prefix) :])
            return {
                "main_topic": topic,
                "comparison_topics": [],
                "query_kind": "direct",
            }

    for prefix in _FOLLOWUP_PREFIXES:
        if lower.startswith(prefix):
            topic = _clean_topic(query[len(prefix) :])
            return {
                "main_topic": topic,
                "comparison_topics": [],
                "query_kind": "followup",
            }

    followup_style_match = re.match(
        r"(?i)^(applications|advantages|limitations|benefits|uses|disadvantages|examples|summary|meaning|purpose|features)\??\s+of\s+(.+)$",
        query,
    )
    if followup_style_match:
        topic = _clean_topic(followup_style_match.group(2))
        return {
            "main_topic": topic,
            "comparison_topics": [],
            "query_kind": "followup",
        }

    if branch != "comparative_retrieval":
        for prep in (" in ", " for ", " about ", " on "):
            if prep in lower and not lower.startswith(prep.strip() + " "):
                head = _clean_topic(query.split(prep, 1)[0])
                if head:
                    return {
                        "main_topic": head,
                        "comparison_topics": [],
                        "query_kind": "followup",
                    }

    if branch == "clarification_candidate":
        cleaned = _clean_topic(query)
        if cleaned:
            return {
                "main_topic": cleaned,
                "comparison_topics": [],
                "query_kind": "ambiguous",
            }

    return dict(DEFAULT_TOPIC_INFO)


def _llm_topic_info(query: str) -> dict:
    prompt = f"""
You are a topic extraction system for an academic conversational RAG assistant.

Extract the PRIMARY semantic topic from the user query.

Return STRICT JSON only with this shape:
{{
  "main_topic": "compact noun phrase",
  "comparison_topics": ["topic A", "topic B"],
  "query_kind": "direct|comparative|followup|ambiguous|out_of_domain"
}}

Rules:
- main_topic should be concise.
- If the query is comparative, comparison_topics must contain the compared items.
- If the query is a follow-up like applications/limitations/advantages, infer the topic from the query wording.
- Do not add explanations.
- Do not wrap in markdown.

Query:
{query}

JSON Response:
"""

    response = client.chat.completions.create(
        model=GENERATION_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=160,
    )

    text = response.choices[0].message.content or ""
    parsed = _safe_json_load(text)

    if not parsed:
        return dict(DEFAULT_TOPIC_INFO)

    info = dict(DEFAULT_TOPIC_INFO)
    info["main_topic"] = _clean_topic(parsed.get("main_topic", ""))
    info["comparison_topics"] = _normalize_comparison_topics(parsed.get("comparison_topics", []))
    info["query_kind"] = str(parsed.get("query_kind", "unknown")).strip() or "unknown"

    if not info["main_topic"] and info["comparison_topics"]:
        info["main_topic"] = info["comparison_topics"][0]

    return info


def extract_topic_info(query: str, branch: str = "", memory_topic: str = "") -> dict:
    query = _normalize_text(query)

    heuristic = _heuristic_topic_info(query, branch=branch)

    if heuristic["main_topic"] or heuristic["comparison_topics"] or heuristic["query_kind"] != "unknown":
        return heuristic

    try:
        info = _llm_topic_info(query)
        if not info["main_topic"] and memory_topic:
            info["main_topic"] = _clean_topic(memory_topic)
        return info
    except Exception:
        fallback = dict(DEFAULT_TOPIC_INFO)
        fallback["main_topic"] = _clean_topic(memory_topic) if memory_topic else _clean_topic(query)
        fallback["query_kind"] = "unknown"
        return fallback
