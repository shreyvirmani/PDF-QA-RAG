# 🤖 PDF AI Assistant (RAG)

An AI-powered PDF Question Answering application that uses **Retrieval-Augmented Generation (RAG)** to answer questions from uploaded PDF documents. The application combines **Google Gemini**, **LangChain**, **FAISS**, and **Hugging Face Embeddings** to provide accurate, context-aware responses.

---

## 🚀 Features

* 📄 Upload any PDF document
* 💬 Chat with your PDF using natural language
* 🤖 AI-generated answers using Google Gemini
* 🔍 Semantic search with FAISS Vector Database
* 🧠 Hugging Face Sentence Transformers for embeddings
* 📚 Displays source references used to generate answers
* ⚡ Fast and efficient Retrieval-Augmented Generation (RAG)
* 🎨 Clean Streamlit user interface

---

## 🛠️ Tech Stack

* Python
* Streamlit
* Google Gemini API
* LangChain
* FAISS
* Hugging Face Embeddings
* Sentence Transformers

---

## 📂 Project Structure

```text
PDF-AI-Assistant/
│
├── app.py
├── requirements.txt
├── README.md
├── .env
│
├── src/
│   ├── loader.py
│   ├── splitter.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── rag.py
│   └── llm.py
│
├── documents/
│
└── vector_store/
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/shreyvirmani/PDF-QA-RAG
```

Move into the project

```bash
cd PDF-QA-RAG
```

Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required packages

```bash
pip install -r requirements.txt
```

Create a `.env` file

```env
GOOGLE_API_KEY=YOUR_GEMINI_API_KEY
HF_TOKEN=YOUR_HUGGINGFACE_TOKEN
```

Run the application

```bash
streamlit run app.py
```

---

## 🧠 How It Works

1. Upload a PDF document.
2. The document is split into smaller chunks.
3. Hugging Face embeddings are created for each chunk.
4. The embeddings are stored inside a FAISS vector database.
5. When a question is asked:

   * Relevant chunks are retrieved using semantic search.
   * The retrieved context is sent to Google Gemini.
   * Gemini generates a context-aware answer.
6. The application also displays the source text used for answering.

---

## 📈 Future Improvements

* Multiple PDF support
* Conversation memory
* OCR support for scanned PDFs
* Streaming responses
* Support for additional LLMs

---

## 📚 Skills Demonstrated

* Retrieval-Augmented Generation (RAG)
* Large Language Models (LLMs)
* Prompt Engineering
* Vector Databases
* Semantic Search
* Hugging Face Embeddings
* LangChain
* FAISS
* Streamlit Development
* Python

---

## 👨‍💻 Author

**Shrey Virmani**

If you like this project, consider giving it a ⭐ on GitHub.
