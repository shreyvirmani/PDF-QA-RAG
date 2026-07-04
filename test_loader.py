from src.loader import load_pdf
from src.splitter import split_documents  # Added import

# Provide a path to a sample PDF file on your computer
pdf_path = r"C:\Users\shrey\OneDrive\Desktop\AI ML\PDF-QA-RAG\documents\ANALOG ELECTRONICS -(II) pyqs.pdf"

try:
    print("Loading PDF...")
    pages = load_pdf(pdf_path)
    
    # Verify the results
    print(f"Successfully loaded {len(pages)} pages!")
    
    print("Splitting documents into chunks...") # Added step
    chunks = split_documents(pages)             # Added step
    print(f"Successfully created {len(chunks)} chunks!") # Added step

    print("\n--- First Page Content Preview ---")
    print(pages[0].page_content[:500]) # Prints first 500 characters
    # Print metadata for the first page
    print(f"Metadata: {pages[0].metadata}")

except Exception as e:
    print(f"An error occurred: {e}")

