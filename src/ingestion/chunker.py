import hashlib

from ..core.config import CHUNK_OVERLAP, CHUNK_SIZE


def chunk_text(text):
    chunks = []
    paragraphs = text.split("\n\n")

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) <= CHUNK_SIZE:
            chunks.append(para)
        else:
            start = 0
            while start < len(para):
                end = start + CHUNK_SIZE
                chunk = para[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def _hash(text):
    return hashlib.md5(text.strip().encode()).hexdigest()


def chunk_documents(documents, file_names=None):
    all_chunks = []

    for i, doc in enumerate(documents):
        if isinstance(doc, dict):
            text = doc.get("text", "")
            source = doc.get("source", "unknown")
            page = doc.get("page")
        else:
            text = str(doc)
            source = file_names[i] if file_names and i < len(file_names) else "unknown"
            page = None

        if not text.strip():
            continue

        chunks = chunk_text(text)
        seen = set()

        for j, chunk in enumerate(chunks):
            h = _hash(chunk)
            if h in seen:
                continue

            seen.add(h)

            if page is not None:
                chunk_id = f"{source}_p{page}_{j}"
            else:
                chunk_id = f"{source}_{j}"

            all_chunks.append(
                {
                    "text": chunk,
                    "source": source,
                    "page": page,
                    "chunk_id": chunk_id,
                }
            )

    return all_chunks
