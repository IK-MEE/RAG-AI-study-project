from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.llm import get_llm
from app.rag import get_retriever
from app.prompts import RAG_PROMPT


app = FastAPI(title="Q's RAG Chatbot")

# ── Request/response models ───────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str


# ── RAG chain ─────────────────────────────────────────────────────────────────

def build_rag_chain():
    """
    Builds the RAG chain:
      1. Retriever fetches top-k relevant chunks from Chroma
      2. Chunks are formatted into the prompt as {context}
      3. LLM receives prompt and generates an answer

    Returns a callable that accepts a question string and returns an answer string.
    """
    retriever = get_retriever()
    llm = get_llm()

    def chain(question: str) -> str:
        # Step 1: retrieve relevant chunks
        docs = retriever.invoke(question)

        # Step 2: join chunks into a single context string
        context = "\n\n".join(doc.page_content for doc in docs)

        # Step 3: build the full prompt (printed for debugging / interviews)
        prompt = RAG_PROMPT.format(context=context, question=question)
        print("\n" + "="*60)
        print("FULL PROMPT SENT TO LLM:")
        print("="*60)
        print(prompt)
        print("="*60 + "\n")

        # Step 4: send to LLM and return the answer
        response = llm.invoke(prompt)
        return response.content

    return chain


# Build chain once at startup
rag_chain = build_rag_chain()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    answer = rag_chain(request.question)
    return ChatResponse(answer=answer)


# ── Static files ──────────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory="static"), name="static")
