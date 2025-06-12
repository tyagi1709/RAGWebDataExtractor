# 🤖 RAG_WEB_DATA_EXTRACTOR

A smart, retrieval-augmented generation (RAG) pipeline that extracts and summarizes data directly from web pages using [LangChain](https://www.langchain.com/) and Hugging Face's `TinyLlama` model. This version includes full error handling, logging, and a reusable function for flexible deployment.

---

## 🚀 Features

- 🌐 Extracts data from web pages using `WebBaseLoader`
- ✂️ Splits text into optimized chunks for processing
- 🧠 Generates semantic embeddings with `all-MiniLM-L6-v2`
- 🔍 Retrieves contextually relevant content using Chroma vector store
- 📝 Summarizes web content using `TinyLlama-1.1B-Chat` via HuggingFace Hub
- 🔁 Includes a reusable function: `create_rag_pipeline(url, query, api_token)`
- ✅ Enhanced error handling and debug-friendly output

---

## 🛠️ Installation

Ensure you have **Python 3.8+** installed.

Install all required packages using the `requirements.txt` file:

Install core dependencies:

```bash

pip install -r requirements.txt

pip install --upgrade langchain langchain-community pypdf chromadb
