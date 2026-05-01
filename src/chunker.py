from config import CHUNK_SIZE, CHUNK_OVERLAP


def is_valid_chunk(chunk):
    text = chunk.strip()

    if len(text) < 50:
        return False

    if text.count(" ") < 5:
        return False

    bad_chars = sum(1 for c in text if not c.isalnum() and c not in " .,()-:%/\n")
    if bad_chars / max(len(text), 1) > 0.35:
        return False

    return True


def chunk_text(text):
    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]

        if is_valid_chunk(chunk):
            chunks.append(chunk.strip())

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def chunk_documents(documents):
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_text(doc))
    return all_chunks
