import os
import re
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# Shared embedding model name + Qdrant factory — query1.py is the single source
# of truth so chunker (ingestion) and query1 (retrieval) can never drift apart.
from query1 import EMBED_MODEL_NAME, qdrant_client


# ─────────────────────────────────────────────
# STEP 1 : CHUNKING
# ─────────────────────────────────────────────
def chunk_document(text: str, source: str, pages_per_chunk: int = 1) -> list[dict]:
    """
    Split the document into chunks using '---' as page separators.
    Each chunk groups `pages_per_chunk` consecutive pages together.
    Page number (1-based) is stored in metadata as page_number.
    """

    # split on --- separators, recording each page's exact start position
    # so section detection is always accurate (no ambiguous text.find() search)
    raw_pages = []   # list of (page_text, start_position_in_original_text)
    page_starts = [0] + [m.end() for m in re.finditer(r'\n\s*---\s*\n', text)]

    for j in range(len(page_starts)):
        start     = page_starts[j]
        end       = page_starts[j + 1] if j + 1 < len(page_starts) else len(text)
        page_text = text[start:end].strip()
        if page_text:
            raw_pages.append((page_text, start))

    print(f"Total pages detected: {len(raw_pages)}")

    # find all markdown headers and their positions for section tagging
    headers = []
    position = 0
    for line in text.split("\n"):
        if line.strip().startswith("#") and " " in line.strip() and len(line.strip()) < 200:
            headers.append((position, line.strip()))
        position += len(line) + 1

    chunks = []
    for i in range(0, len(raw_pages), pages_per_chunk):
        page_group   = raw_pages[i:i + pages_per_chunk]
        chunk_text   = "\n\n---\n\n".join(p[0] for p in page_group)
        page_number  = i + 1
        chunk_pos    = page_group[0][1]   # exact position recorded during split

        # find the last markdown header that appeared before this chunk
        current_section = "no section"
        for header_pos, header_text in headers:
            if header_pos <= chunk_pos:
                current_section = header_text
            else:
                break

        chunks.append({
            "text": chunk_text,
            "metadata": {
                "file":        source,
                "section":     current_section,
                "chunk_id":    len(chunks),
                "page_number": page_number
            }
        })

    return chunks


# ─────────────────────────────────────────────
# STEP 2 : EMBEDDING AND STORING
# ─────────────────────────────────────────────
def embed_and_store(chunks: list[dict], collection_name: str) -> QdrantClient:

    # load the embedding model (single source of truth: query1.EMBED_MODEL_NAME)
    model = SentenceTransformer(EMBED_MODEL_NAME)

    # extract just the text from each chunk
    texts = [chunk["text"] for chunk in chunks]

    # embed all texts at once
    print(f"Embedding {len(texts)} chunks...")
    vectors = model.encode(texts, show_progress_bar=True)

    # connect to qdrant via the shared factory (single source of truth for connection config)
    client = qdrant_client()

    # always recreate the collection to avoid stale chunks from previous runs
    existing_collections = [c.name for c in client.get_collections().collections]
    if collection_name in existing_collections:
        client.delete_collection(collection_name)
        print(f"Collection '{collection_name}' deleted — recreating with new chunk settings")

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print(f"Collection '{collection_name}' created")

    # build the list of points to insert
    points = []
    for i, chunk in enumerate(chunks):
        point = PointStruct(
            id=chunk["metadata"]["chunk_id"],
            vector=vectors[i].tolist(),
            payload={
                "text":       chunk["text"],
                "file":       chunk["metadata"]["file"],
                "section":    chunk["metadata"]["section"],
                "chunk_id":   chunk["metadata"]["chunk_id"],
                "page_number": chunk["metadata"]["page_number"]
            }
        )
        points.append(point)

    # insert all points into qdrant at once
    client.upsert(
        collection_name=collection_name,
        points=points
    )

    print(f"Stored {len(points)} points in collection '{collection_name}'")
    return client


# ─────────────────────────────────────────────
# STEP 3 : FETCH CHUNKS FROM QDRANT
# ─────────────────────────────────────────────
def fetch_chunks_from_qdrant(client: QdrantClient, collection_name: str) -> list[dict]:

    all_chunks = []
    offset     = None

    while True:
        results, offset = client.scroll(
            collection_name=collection_name,
            limit=100,
            with_payload=True,
            with_vectors=False,
            offset=offset
        )

        for point in results:
            all_chunks.append({
                "text": point.payload["text"],
                "metadata": {
                    "chunk_id":   point.payload["chunk_id"],
                    "file":       point.payload["file"],
                    "section":    point.payload["section"],
                    "page_number": point.payload.get("page_number")
                }
            })

        if offset is None:
            break

    # sort by chunk_id so chunks are in document order
    all_chunks.sort(key=lambda c: c["metadata"]["chunk_id"])

    print(f"Fetched {len(all_chunks)} chunks from collection '{collection_name}'")
    return all_chunks


# ─────────────────────────────────────────────
# RUN DIRECTLY: rechunk + re-embed only (no LLM)
# usage: python chunker.py
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from loader import load_document
    load_dotenv()

    PAGES_PER_CHUNK = 1
    file_path       = "../data/LoewsCompany.md"

    text   = load_document(file_path)
    chunks = chunk_document(text, source=Path(file_path).name,
                            pages_per_chunk=PAGES_PER_CHUNK)

    print(f"Total chunks (pages_per_chunk={PAGES_PER_CHUNK}): {len(chunks)}")

    # save chunks to txt for inspection
    output_path = "../data/chunks_output.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(f"── Chunk {chunk['metadata']['chunk_id']} ──\n")
            f.write(f"Section    : {chunk['metadata']['section']}\n")
            f.write(f"Page       : {chunk['metadata']['page_number']}\n")
            f.write(f"Text       :\n{chunk['text']}\n")
            f.write("\n" + "─" * 60 + "\n\n")
    print(f"Chunks saved to {output_path}")

    collection_name = Path(file_path).stem
    embed_and_store(chunks, collection_name=collection_name)
    print("Done — Qdrant updated. You can now run the pipeline.")
