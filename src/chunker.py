from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text):
    chunks = []

    # Step 1: split by paragraphs (preserve structure)
    paragraphs = text.split("\n\n")

    for para in paragraphs:
        para = para.strip()

        if not para:
            continue

        # Step 2: if small → keep as-is
        if len(para) <= CHUNK_SIZE:
            chunks.append(para)

        # Step 3: if large → split with overlap
        else:
            start = 0
            while start < len(para):
                end = start + CHUNK_SIZE
                chunk = para[start:end].strip()

                if chunk:
                    chunks.append(chunk)

                start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def chunk_documents(documents):
    all_chunks = []

    for doc in documents:
        all_chunks.extend(chunk_text(doc))

    return all_chunks
