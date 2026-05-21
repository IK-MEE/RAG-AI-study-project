import sys
sys.path.insert(0, ".")
from app.rag import load_vectorstore

vs = load_vectorstore()

queries = [
    "What are this person's skills?",
    "What is his education?",
    "What is their work experience?",
]

for q in queries:
    print(f"\nQuery: {q}")
    results = vs.similarity_search(q, k=2)
    for r in results:
        print(f"  [{r.metadata.get('source')}] {r.page_content[:150]}")