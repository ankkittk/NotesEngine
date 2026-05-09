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
    seen = set()

    for i, doc in enumerate(documents):
        file_name = file_names[i] if file_names else "unknown"
        chunks = chunk_text(doc)

        for j, chunk in enumerate(chunks):
            h = _hash(chunk)
            if h in seen:
                continue

            seen.add(h)

            all_chunks.append({
                "text": chunk,
                "source": file_name,
                "chunk_id": f"{file_name}_{j}"
            })

    return all_chunks
