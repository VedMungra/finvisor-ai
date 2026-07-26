import os
import glob
from dotenv import load_dotenv
load_dotenv()

from retriever import get_retriever
from langchain_groq import ChatGroq

def main():
    print("--- Database & Chunk Statistics ---")
    
    # Count source documents
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    md_files = glob.glob(os.path.join(data_dir, "**", "*.md"), recursive=True)
    num_docs = len(md_files)
    print(f"Total Source Documents (Markdown files): {num_docs}")
    
    for md_file in md_files:
        print(f" - {os.path.basename(md_file)}")

    # Initialize a dummy LLM to get the retriever
    dummy_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    retriever = get_retriever(dummy_llm)
    
    # Query ChromaDB directly
    data = retriever.vectorstore.get()
    ids = data.get("ids", [])
    num_chunks = len(ids)
    
    print(f"\nTotal Vectorized Chunks in ChromaDB: {num_chunks}")
    
    # Calculate average chunks per document
    if num_docs > 0:
        avg_chunks = num_chunks / num_docs
        print(f"Average Chunks per Document: {avg_chunks:.1f}")

if __name__ == "__main__":
    main()
