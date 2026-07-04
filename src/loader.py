from langchain_community.document_loaders import PyPDFLoader

def load_pdf(pdf_path):
    # Initialize the loader with your file path
    loader = PyPDFLoader(pdf_path)
    
    # Load and extract the pages from the document
    pages = loader.load()
    
    return pages
