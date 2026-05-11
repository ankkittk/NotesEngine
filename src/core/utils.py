from pathlib import Path
from typing import Any


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def get_context_text(context: Any) -> str:
    if isinstance(context, dict):
        return str(context.get("text", ""))
    return str(context)


def format_citation(context: Any) -> str:
    if not isinstance(context, dict):
        return "Source: unknown"

    source = context.get("source", "unknown")
    page = context.get("page")

    if page is None:
        return f"Source: {source}"

    return f"Source: {source} | Page: {page}"


def context_to_prompt_block(context: Any) -> str:
    return f"[{format_citation(context)}]\n{get_context_text(context)}"


def unique_citations(contexts: Any) -> list[str]:
    seen = set()
    labels = []

    for context in contexts or []:
        label = format_citation(context)
        if label not in seen:
            seen.add(label)
            labels.append(label)

    return labels
