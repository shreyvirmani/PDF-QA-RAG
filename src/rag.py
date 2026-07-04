from src.llm import get_llm


def answer_question(vector_store, question):
    llm = get_llm()

    docs = vector_store.similarity_search(
        question,
        k=3
    )

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the information provided below.

If the answer is not present in the context,
say:

"I couldn't find that information in the uploaded PDF."

Context:

{context}

Question:

{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content, docs
