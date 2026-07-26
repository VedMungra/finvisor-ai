import time
import uuid
import numpy as np
from agent import app as agent_app
from langchain_core.messages import HumanMessage
import logging
import warnings
warnings.filterwarnings('ignore')
logging.getLogger("Agent").setLevel(logging.ERROR)
logging.getLogger("API").setLevel(logging.ERROR)

def run_query(query: str, source_filename: str = None) -> float:
    thread_id = str(uuid.uuid4())
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "source_filename": source_filename
    }
    config = {"configurable": {"thread_id": thread_id}}
    
    start_time = time.perf_counter()
    final_state = agent_app.invoke(initial_state, config=config)
    end_time = time.perf_counter()
    
    return end_time - start_time

def main():
    print("--- LangGraph Pathway Latency Benchmark ---")
    print("Running 10 queries per pathway. This may take a few minutes...\n")

    pathways = {
        "RAG (Vector Search)": {
            "queries": [
                ("What was Apple's total net sales in Q1 2024?", "Apple_Q1_2024_Earnings.md"),
                ("How much did iPad revenue decline for Apple?", "Apple_Q1_2024_Earnings.md"),
                ("What was the regional performance in Greater China for Apple?", "Apple_Q1_2024_Earnings.md"),
                ("What cash dividend did Apple declare?", "Apple_Q1_2024_Earnings.md"),
                ("What was Tesla's total revenue for Q4 2025?", "tesla_q4_2025_earnings.md"),
                ("How many vehicles did Tesla deliver in the full year 2025?", "tesla_q4_2025_earnings.md"),
                ("What were the energy storage deployments for Tesla in Q4 2025?", "tesla_q4_2025_earnings.md"),
                ("How is Tesla's Optimus humanoid robot being used currently?", "tesla_q4_2025_earnings.md"),
                ("What is Elon Musk's outlook on AI compute clusters for 2026?", "tesla_q4_2025_earnings.md"),
                ("What were the total revenue and net income for GlobalTech in Q1 2026?", "sample_q1_earnings.md")
            ]
        },
        "Market Data (yfinance tools)": {
            "queries": [
                ("What is the current stock price and fundamental metrics for AAPL?", None),
                ("Compare the fundamental metrics of MSFT and GOOGL", None),
                ("What is the intrinsic value of TSLA?", None),
                ("Plot a 1-month stock chart for NVDA", None),
                ("What is the current stock price and fundamental metrics for AMZN?", None),
                ("Compare the fundamental metrics of AAPL and TSLA", None),
                ("What is the intrinsic value of META?", None),
                ("Plot a 1-month stock chart for NFLX", None),
                ("What is the current stock price and fundamental metrics for AMD?", None),
                ("Plot a 1-month stock chart for JPM", None)
            ]
        },
        "Web Search (Tavily Fallback)": {
            "queries": [
                ("What is the latest breaking news about the Federal Reserve interest rates?", None),
                ("Who won the Super Bowl in 2026?", None),
                ("What are the latest developments in quantum computing this month?", None),
                ("Who is the current President of the United States?", None),
                ("What is the weather forecast for New York City?", None),
                ("What are the top travel destinations for 2026?", None),
                ("What is the latest news regarding SpaceX Starship?", None),
                ("Who won the most recent Oscar for Best Picture?", None),
                ("What are the current geopolitical tensions in Europe?", None),
                ("What is the latest major breakthrough in cancer research?", None)
            ]
        }
    }

    results = {}

    for pathway_name, data in pathways.items():
        print(f"Benchmarking Pathway: {pathway_name}...")
        latencies = []
        for i, (query, source) in enumerate(data["queries"]):
            try:
                lat = run_query(query, source)
                latencies.append(lat)
                print(f"  [{i+1}/10] Query: '{query[:40]}...' -> {lat:.2f}s")
            except Exception as e:
                print(f"  [{i+1}/10] Query: '{query[:40]}...' -> FAILED ({e})")
        
        if latencies:
            results[pathway_name] = {
                "avg": np.mean(latencies),
                "median": np.median(latencies),
                "p95": np.percentile(latencies, 95)
            }
            print(f"  => {pathway_name} Average: {results[pathway_name]['avg']:.2f}s | Median: {results[pathway_name]['median']:.2f}s | p95: {results[pathway_name]['p95']:.2f}s\n")
        else:
            print(f"  => {pathway_name} Failed to get latencies.\n")

    print("\n--- Final Latency Report ---")
    for pathway_name, stats in results.items():
        print(f"Pathway: {pathway_name}")
        print(f"  Average : {stats['avg']:.2f}s")
        print(f"  Median  : {stats['median']:.2f}s")
        print(f"  p95     : {stats['p95']:.2f}s")

if __name__ == "__main__":
    main()
