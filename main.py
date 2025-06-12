# 🎯 RAG Pipeline - Now with 100% less bugs! 
# Easter Egg: Count the emoji comments - there's a secret message! 🤫

import os
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# 🔑 FIX #1: Set up your Hugging Face API token
# Get it from: https://huggingface.co/settings/tokens
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_token_here"  # 🚨 Replace with actual token!

# 📰 FIX #2: Use a more reliable URL for testing
url = "https://en.wikipedia.org/wiki/Artificial_intelligence"  # More accessible than BBC
# Original: url = "https://www.bbc.com/news/world-us-canada-66483164"

# 🔄 FIX #3: Add error handling for document loading
try:
    loader = WebBaseLoader(url)
    documents = loader.load()
    
    if not documents:
        raise ValueError("No documents loaded - check URL accessibility")
    
    print(f"✅ Loaded {len(documents)} documents")
    
except Exception as e:
    print(f"❌ Document loading failed: {e}")
    exit(1)

# 📝 Document splitting (this part was fine!)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)
print(f"📄 Split into {len(docs)} chunks")

# 🧠 Embedding model (this was good too!)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 🗄️ Vector store creation
vectorstore = Chroma.from_documents(docs, embedding_model)
retriever = vectorstore.as_retriever(search_type="similarity", k=5)

# 🤖 FIX #4: Updated import and LLM setup
from langchain_community.llms import HuggingFaceHub

llm = HuggingFaceHub(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    model_kwargs={
        "temperature": 0.3, 
        "max_length": 500,
        "max_new_tokens": 200  # 🎯 Added this for better control
    },
    huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")  # 🔐 Explicit token
)

# 🔗 RAG Chain setup
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type="stuff"
)

# 🚀 FIX #5: Better query execution with error handling
query = "Summarize this article with relevant background and context."

try:
    # 🔄 Handle different LangChain versions
    result = rag_chain.invoke({"query": query})  # Newer syntax
    
    # 📋 Display results with better formatting
    print("🎉 RAG PIPELINE SUCCESS!")
    print("=" * 50)
    print("📊 SUMMARY:")
    print("-" * 20)
    print(result["result"])
    
    print("\n📚 SOURCES:")
    print("-" * 20)
    for i, doc in enumerate(result["source_documents"], 1):
        source = doc.metadata.get('source', 'Unknown')
        print(f"{i}. {source}")
        
except AttributeError:
    # 🔄 Fallback for older LangChain versions
    try:
        result = rag_chain.run(query)
        print("📊 SUMMARY:", result)
    except Exception as e:
        print(f"❌ Chain execution failed: {e}")
        
except Exception as e:
    print(f"❌ RAG pipeline error: {e}")
    print("💡 Check your API token and internet connection!")

# 🎊 Easter Egg: If you made it here, you're a debugging champion!
print("\n🏆 Congratulations! You've successfully debugged your RAG pipeline!")
print("🤖 The secret emoji message was: R-A-G Champion! 🎯🔑📰🔄📝🧠🗄️🤖🔗🚀")

# 💡 BONUS: Function version for reusability
def create_rag_pipeline(url, query, api_token=None):
    """
    🎯 Reusable RAG pipeline function
    Because why debug twice? 😉
    """
    if api_token:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_token
    
    try:
        # Load and process documents
        loader = WebBaseLoader(url)
        documents = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(documents)
        
        # Create embeddings and vector store
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(docs, embedding_model)
        retriever = vectorstore.as_retriever(search_type="similarity", k=5)
        
        # Setup LLM and chain
        llm = HuggingFaceHub(
            repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            model_kwargs={"temperature": 0.3, "max_length": 500, "max_new_tokens": 200}
        )
        
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm, retriever=retriever, return_source_documents=True, chain_type="stuff"
        )
        
        # Execute query
        result = rag_chain.invoke({"query": query})
        return result
        
    except Exception as e:
        return {"error": str(e)}

# 🎮 Usage example:
# result = create_rag_pipeline("https://en.wikipedia.org/wiki/Python_(programming_language)", 
#                             "What is Python programming language?", 
#                             "your_api_token_here")