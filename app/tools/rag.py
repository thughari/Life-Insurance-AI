import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Determine which embeddings to use
def get_embeddings():
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings()
    elif os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    else:
        # Fallback to a dummy embedding if no key is found just so it compiles, 
        # but RAG will fail if actually used without keys.
        from langchain_community.embeddings import FakeEmbeddings
        return FakeEmbeddings(size=1536)

FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "faiss_index")

def build_faiss_index():
    if os.path.exists(FAISS_INDEX_PATH):
        return

    print("Building FAISS index for the first time...")
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    
    docs = []
    for file in pdf_files:
        loader = PyPDFLoader(file)
        docs.extend(loader.load())
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print("FAISS index built successfully.")

def retrieve_policy_context(query: str, k: int = 3) -> str:
    build_faiss_index()
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    
    docs = vectorstore.similarity_search(query, k=k)
    
    context_parts = []
    for i, doc in enumerate(docs):
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", 0) + 1
        content = doc.page_content.replace("\n", " ")
        context_parts.append(f"[Source: {source}, Page: {page}]\n{content}")
        
    return "\n\n".join(context_parts)
