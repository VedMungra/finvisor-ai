import os
from dotenv import load_dotenv
load_dotenv()
from retriever import get_retriever
from langchain_groq import ChatGroq

def main():
    print("--- Retrieval Accuracy Evaluation ---")
    
    # Initialize Retriever
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    # Using k_initial=10, k_final=3 as per agent.py
    retriever = get_retriever(llm, k_initial=10, k_final=3)
    
    test_set = [
        ("What was Apple's total net sales in Q1 2024?", "Apple_Q1_2024_Earnings.md"),
        ("How much did iPad revenue decline for Apple?", "Apple_Q1_2024_Earnings.md"),
        ("What did Tim Cook say about the installed base of active devices?", "Apple_Q1_2024_Earnings.md"),
        ("What was the regional performance in Greater China for Apple?", "Apple_Q1_2024_Earnings.md"),
        ("What cash dividend did Apple declare?", "Apple_Q1_2024_Earnings.md"),
        ("What was Tesla's total revenue for Q4 2025?", "tesla_q4_2025_earnings.md"),
        ("How many vehicles did Tesla deliver in the full year 2025?", "tesla_q4_2025_earnings.md"),
        ("What were the energy storage deployments for Tesla in Q4 2025?", "tesla_q4_2025_earnings.md"),
        ("How is Tesla's Optimus humanoid robot being used currently?", "tesla_q4_2025_earnings.md"),
        ("What is Elon Musk's outlook on AI compute clusters for 2026?", "tesla_q4_2025_earnings.md"),
        ("What were the total revenue and net income for GlobalTech in Q1 2026?", "sample_q1_earnings.md"),
        ("What is OmniMind Enterprise and its context window size?", "sample_q1_earnings.md"),
        ("What new features does CloudSecure version 4.0 introduce?", "sample_q1_earnings.md"),
        ("What is GlobalTech's projected revenue for Q2 2026?", "sample_q1_earnings.md"),
        ("How much is GlobalTech's ongoing share repurchase program?", "sample_q1_earnings.md")
    ]
    
    top_1_correct = 0
    top_3_correct = 0
    total = len(test_set)
    
    print(f"Evaluating {total} questions against ChromaDB...\n")
    
    for i, (question, expected_source) in enumerate(test_set):
        docs = retriever.invoke(question)
        
        sources = [doc.metadata.get("source", "") for doc in docs]
        
        is_top_1 = False
        is_top_3 = False
        
        if len(sources) > 0 and sources[0] == expected_source:
            is_top_1 = True
            top_1_correct += 1
            
        if expected_source in sources[:3]:
            is_top_3 = True
            top_3_correct += 1
            
        status = "PASS" if is_top_3 else "FAIL"
        print(f"[{i+1}/{total}] {status} | Q: '{question[:40]}...'")
        print(f"    Expected: {expected_source} | Retrieved Top-3: {sources}")
        
    print("\n--- Evaluation Results ---")
    print(f"Total Questions: {total}")
    print(f"Top-1 Accuracy: {top_1_correct}/{total} ({(top_1_correct/total)*100:.1f}%)")
    print(f"Top-3 Accuracy: {top_3_correct}/{total} ({(top_3_correct/total)*100:.1f}%)")

if __name__ == "__main__":
    main()
