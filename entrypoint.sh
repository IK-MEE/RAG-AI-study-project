#!/bin/bash
set -e

# If chroma_db is empty (first run), build the vectorstore
if [ ! -f "$CHROMA_DB_PATH/chroma.sqlite3" ]; then
    echo "[entrypoint] Building vectorstore from documents..."
    python -c "
from app.rag import load_documents, split_documents, build_vectorstore
docs = load_documents()
chunks = split_documents(docs)
build_vectorstore(chunks)
print('[entrypoint] Vectorstore built successfully.')
"
fi

echo "[entrypoint] Starting FastAPI server..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
