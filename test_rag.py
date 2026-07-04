import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from llm import get_llm
from src.loader import load_pdf
from src.splitter import split_documents
from src.embeddings import get_embedding_model
from src.vector_store import (
    create_vector_store,
    save_vector_store,
    load_vector_store
)
from src.rag import answer_question


PDF_PATH = r"documents\ANALOG ELECTRONICS -(II) pyqs.pdf"


def main():
    print("=" * 60)
    print("📄 PDF Question Answering using RAG")
    print("=" * 60)

    # Load embedding model
    print("\n🧠 Loading embedding model...")
    embedding_model = get_embedding_model()

    # Check if vector store already exists
    if os.path.exists("vector_store/index.faiss"):
        print("\n📦 Loading existing vector database...")
        vector_store = load_vector_store(embedding_model)
        print("✅ Vector database loaded successfully!")

    else:
        print("\n📖 Loading PDF...")
        documents = load_pdf(PDF_PATH)
        print(f"✅ Loaded {len(documents)} pages")

        print("\n✂ Splitting documents...")
        chunks = split_documents(documents)
        print(f"✅ Created {len(chunks)} chunks")

        print("\n📦 Creating vector database...")
        vector_store = create_vector_store(chunks, embedding_model)

        print("\n💾 Saving vector database...")
        save_vector_store(vector_store)

        print("✅ Vector database created and saved!")

    print("\n" + "=" * 60)
    print("💬 Chat with your PDF")
    print("Type 'exit' to quit")
    print("=" * 60)

    while True:
        question = input("\nYou: ")

        if question.lower() == "exit":
            print("👋 Goodbye!")
            break

        answer, docs = answer_question(vector_store, question)

        print("\n🤖 Answer:\n")
        print(answer)

        print("\n📚 Sources Used:\n")

        for i, doc in enumerate(docs, start=1):
            page = doc.metadata.get("page", "Unknown")
            if isinstance(page, int):
                page += 1

            print("-" * 60)
            print(f"Source {i}")
            print(f"Page   : {page}")
            print(f"File   : {doc.metadata.get('source')}")

            preview = doc.page_content[:250].replace("\n", " ")
            print(f"Preview: {preview}...")
            print("-" * 60)


if __name__ == "__main__":
    main()