from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from app.config import settings


# ── 1. Load ──────────────────────────────────────────────────────────────────

def load_documents():
    """Read every .txt file in the data/ folder and return a list of Documents."""
    loader = DirectoryLoader(
        settings.data_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
    )
    docs = loader.load()
    print(f"[rag] Loaded {len(docs)} document(s)")
    return docs


# ── 2. Split ─────────────────────────────────────────────────────────────────

def split_documents(docs):
    """
    Cut documents into overlapping chunks.

    chunk_size=1000  — large enough to capture a full resume section (SKILLS,
                       EXPERIENCE, EDUCATION) in one chunk rather than splitting
                       mid-section. Tuned for short structured documents like resumes.
    chunk_overlap=100 — repeats the last 100 chars in the next chunk so context
                        is not lost at section boundaries.
    separators        — tries to split on blank lines first, then single newlines,
                        then sentences. Respects document structure.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"[rag] Split into {len(chunks)} chunk(s)")
    return chunks


# ── 3. Embed & store ──────────────────────────────────────────────────────────

def get_embedding_function():
    """Return the local sentence-transformers embedding model."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def build_vectorstore(chunks):
    """Embed chunks and persist them to Chroma on disk."""
    embeddings = get_embedding_function()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=settings.chroma_db_path,
    )
    print(f"[rag] Vectorstore saved to {settings.chroma_db_path}")
    return vectorstore


def load_vectorstore():
    """Load an already-built Chroma vectorstore from disk."""
    embeddings = get_embedding_function()
    return Chroma(
        persist_directory=settings.chroma_db_path,
        embedding_function=embeddings,
    )


# ── 4. Retrieve ───────────────────────────────────────────────────────────────

def get_retriever():
    """Return a retriever that fetches the top 3 most relevant chunks."""
    vectorstore = load_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": 3})
