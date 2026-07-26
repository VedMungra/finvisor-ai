import os
from dotenv import load_dotenv
load_dotenv()
from retriever import get_retriever
from langchain_groq import ChatGroq

def main():
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    retriever = get_retriever(llm, k_initial=10, k_final=3)
    
    query = "What was Apple's total net sales in Q1 2024?"
    print(f"Query: {query}")
    
    embed = retriever.embeddings.embed_query(query)
    print(f"Query embedding length: {len(embed)}")
    print(f"First 5 elements: {embed[:5]}")
    
    data = retriever.vectorstore.get(include=["embeddings", "metadatas", "documents"])
    print(f"Total chunks in DB: {len(data['ids'])}")
    if len(data['ids']) > 0:
        db_embed = data['embeddings'][0]
        print(f"DB embedding length: {len(db_embed)}")
        print(f"DB First 5 elements: {db_embed[:5]}")
        
    docs = retriever.vectorstore.similarity_search_with_score(query, k=10)
    print(f"similarity_search_with_score returned {len(docs)} documents.")
    for d, score in docs:
        print(f"Score: {score}, metadata: {d.metadata}")

if __name__ == "__main__":
    main()
