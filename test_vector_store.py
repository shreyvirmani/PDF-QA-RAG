from src.loader import load_pdf
from src.splitter import split_documents
from src.embeddings import get_embedding_model
from src.vector_store import (
    create_vector_store,
    save_vector_store,
    load_vector_store
)


def main():
    # Step 1: Load PDF
    print("Loading PDF...")
    documents = load_pdf(r"documents\ANALOG ELECTRONICS -(II) pyqs.pdf")

    print(f"Loaded {len(documents)} pages\n")

    # Step 2: Split into chunks
    print("Splitting documents...")
    chunks = split_documents(documents)

    print(f"Created {len(chunks)} chunks\n")

    # Step 3: Load embedding model
    print("Loading embedding model...")
    embedding_model = get_embedding_model()

    # Step 4: Create FAISS vector store
    print("Creating FAISS vector store...")
    vector_store = create_vector_store(chunks, embedding_model)

    # Step 5: Save vector store
    print("Saving vector store...")
    save_vector_store(vector_store)

    print("Vector store saved successfully!\n")

    # Step 6: Load vector store
    print("Loading vector store...")
    vector_store = load_vector_store(embedding_model)

    print("Vector store loaded successfully!\n")

    # Step 7: Ask a question
    question = input("Ask a question: ")

    # Step 8: Retrieve top 3 similar chunks
    docs = vector_store.similarity_search(
        question,
        k=3
    )

    print("\nTop Matching Chunks:\n")

    for i, doc in enumerate(docs, start=1):
        print("=" * 80)
        print(f"Chunk {i}")
        print("=" * 80)
        print(doc.page_content)
        print("\nMetadata:")
        print(doc.metadata)
        print("\n")


if __name__ == "__main__":
    main()