# 🧠 PDF Summarizer with RAG (Retrieval-Augmented Generation)

This project is a Retrieval-Augmented Generation (RAG) pipeline built using [LangChain](https://www.langchain.com/), which summarizes PDF documents intelligently using a HuggingFace LLM and embedding model.

---

## 📚 Features

- 🧾 Load and read PDF files
- ✂️ Split content into manageable chunks
- 🧠 Generate vector embeddings using `all-MiniLM-L6-v2`
- 🔍 Retrieve relevant chunks with vector similarity search (Chroma DB)
- 📝 Summarize using `TinyLlama-1.1B-Chat` model via HuggingFace Hub

---

## 🛠️ Installation

Make sure you have Python 3.8+ installed.

Install core dependencies:

```bash
install all required packages using the requirements.txt file:

pip install -r requirements.txt

pip install --upgrade langchain langchain-community pypdf chromadb