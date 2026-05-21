"""
test_rag.py — run this from the project root:

    python test_rag.py

It will:
  1. Load and split your documents
  2. Print one raw embedding vector so you can see what it looks like
  3. Build the Chroma vectorstore
  4. Query it and print the top 3 returned chunks
"""

import sys
import os

# Ensure the project root is on the path so 'app' is importable
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.rag import (
    load_documents,
    split_documents,
    get_embedding_function,
    build_vectorstore,
    load_vectorstore,
)


def main():
    # ── Step 1: Load & split ─────────────────────────────────────────────────
    print("\n=== Step 1: Load documents ===")
    docs = load_documents()
    for doc in docs:
        print(f"  Source: {doc.metadata.get('source')}  ({len(doc.page_content)} chars)")

    print("\n=== Step 2: Split into chunks ===")
    chunks = split_documents(docs)
    print(f"  First chunk preview:\n---\n{chunks[0].page_content}\n---")

    # ── Step 2: Show a raw embedding vector ──────────────────────────────────
    print("\n=== Step 3: Embed one chunk (raw vector) ===")
    embed_fn = get_embedding_function()
    sample_text = chunks[0].page_content
    vector = embed_fn.embed_query(sample_text)
    print(f"  Text : {sample_text[:80]}...")
    print(f"  Vector length : {len(vector)} dimensions")
    print(f"  First 10 values: {[round(v, 6) for v in vector[:10]]}")
    print("  (Each number encodes a tiny aspect of the text's meaning)")

    # ── Step 3: Build vectorstore ─────────────────────────────────────────────
    print("\n=== Step 4: Build & persist Chroma vectorstore ===")
    build_vectorstore(chunks)

    # ── Step 4: Query ─────────────────────────────────────────────────────────
    print("\n=== Step 5: Query — 'Who is this person?' ===")
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search("Who is this person?", k=3)
    for i, result in enumerate(results, 1):
        print(f"\n  [Chunk {i}] Source: {result.metadata.get('source')}")
        print(f"  {result.page_content[:300]}")
        print("  ...")

    print("\n✅ All steps passed. Chroma is working.\n")


if __name__ == "__main__":
    main()
