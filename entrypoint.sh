#!/bin/bash
set -e

# Wait for Ollama to be reachable before doing anything
echo "[entrypoint] Waiting for Ollama at $OLLAMA_BASE_URL..."
until curl -sf "$OLLAMA_BASE_URL" > /dev/null 2>&1; do
    echo "[entrypoint] Ollama not ready yet, retrying in 5s..."
    sleep 5
done
echo "[entrypoint] Ollama is ready."

# Build vectorstore if it doesn't exist yet
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
