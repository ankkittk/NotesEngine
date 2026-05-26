import os
import re

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

QUERY_NORMALIZATION_PROMPT = """
You are a retrieval query normalizer for academic RAG.

Your task:
Normalize the query for vector retrieval while preserving original terminology.

VERY IMPORTANT:
- Preserve the original core technical terms.
- Do NOT paraphrase aggressively.
- Do NOT rewrite into natural language explanations.
- Do NOT replace terms with abstract alternatives.
- Keep important keywords exactly as they appear.
- Keep model names unchanged.
- Keep acronyms unchanged unless expansion is useful.
- Preserve comparison entities exactly.

Allowed operations:
- remove conversational fluff
- remove unnecessary question phrasing
- lightly expand abbreviations if useful
- add 1-3 retrieval keywords ONLY if strongly relevant

Good examples:

Input:
What are the applications of logistic regression?

Output:
logistic regression applications binary classification

Input:
Compare Naive Bayes with Bayesian Networks

Output:
Naive Bayes Bayesian Networks comparison

Input:
What is CNN?

Output:
CNN convolutional neural network

Bad examples:

Input:
What is CNN?

Bad Output:
deep learning architecture for image processing

Input:
Compare SVM with logistic regression

Bad Output:
comparison between large margin classifiers and probabilistic classifiers

Return ONLY the normalized retrieval query.
"""


TECHNICAL_TERMS = {
    "svm": "support vector machine",
    "cnn": "convolutional neural network",
    "rnn": "recurrent neural network",
    "nlp": "natural language processing",
    "knn": "k nearest neighbors",
    "pca": "principal component analysis",
    "lda": "linear discriminant analysis",
}


LIGHT_STOPWORDS = {
    "what",
    "is",
    "are",
    "the",
    "of",
    "a",
    "an",
    "please",
    "explain",
    "tell",
    "me",
    "about",
}


def _clean_text(text: str) -> str:
    text = (text or "").strip()

    if not text:
        return ""

    if "```" in text:
        parts = [
            p.strip()
            for p in text.split("```")
            if p.strip()
        ]

        if parts:
            text = parts[0]

    text = text.strip('"\'')
    text = " ".join(text.split())

    return text


def _simple_keyword_normalization(
    query: str
) -> str:
    query = (
        query.lower()
        .replace("?", " ")
        .strip()
    )

    words = re.findall(
        r"\b[a-zA-Z0-9\-\+]+\b",
        query
    )

    cleaned = []

    for word in words:
        if word in LIGHT_STOPWORDS:
            continue

        cleaned.append(word)

    expanded = []

    for word in cleaned:
        expanded.append(word)

        if word in TECHNICAL_TERMS:
            expanded.append(
                TECHNICAL_TERMS[word]
            )

    seen = set()
    final_words = []

    for word in expanded:
        if word not in seen:
            seen.add(word)
            final_words.append(word)

    return " ".join(final_words)


def _llm_normalize_query(
    query: str
) -> str:
    prompt = f"""
{QUERY_NORMALIZATION_PROMPT}

Query:
{query}

Normalized retrieval query:
"""

    response = client.chat.completions.create(
        model=GENERATION_MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.0,
        max_tokens=64,
    )

    text = (
        response
        .choices[0]
        .message
        .content
    )

    return _clean_text(text)


def rewrite_query(
    query: str
) -> str:
    query = (query or "").strip()

    if not query:
        return query

    lightweight = (
        _simple_keyword_normalization(
            query
        )
    )

    try:
        llm_result = (
            _llm_normalize_query(
                query
            )
        )

        llm_result = (
            _clean_text(llm_result)
        )

        if not llm_result:
            return lightweight

        original_terms = set(
            lightweight.lower().split()
        )

        rewritten_terms = set(
            llm_result.lower().split()
        )

        overlap = len(
            original_terms.intersection(
                rewritten_terms
            )
        )

        overlap_ratio = overlap / max(
            1,
            len(original_terms)
        )

        if overlap_ratio < 0.5:
            return lightweight

        return llm_result

    except Exception:
        return lightweight
