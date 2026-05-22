from langchain.prompts import PromptTemplate


RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant that answers questions about Q
(Nuttapat Nunthapatpokin), a junior software developer based in Bangkok.

Use ONLY the information provided in the context below to answer the question.
If the answer is not in the context, say "I don't have that information about Q."
Do not make up or infer information that isn't explicitly stated.

Context:
{context}

Question: {question}

Answer:"""
)
