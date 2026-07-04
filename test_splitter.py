from langchain_core.documents import Document
from src.splitter import split_documents

def test_split_documents():
    # 1. Create a dummy document with content > 1000 characters
    long_text = "This is a test sentence. " * 50  # Roughly 1250 characters
    sample_docs = [Document(page_content=long_text, metadata={"source": "test"})]

    # 2. Run your splitter
    chunks = split_documents(sample_docs)

    # 3. Print results for verification
    print(f"Original length: {len(long_text)}")
    print(f"Number of chunks: {len(chunks)}")
    
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} (Length: {len(chunk.page_content)}) ---")
        print(chunk.page_content[:100] + "...")

if __name__ == "__main__":
    test_split_documents()
