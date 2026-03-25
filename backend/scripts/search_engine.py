import sys
import os

# Thêm thư mục gốc vào sys.path để import được app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.search_service import search_service

def main():
    # Example search
    test_query = "quyền dân sự là gì"
    print(f"\n[🔍] Searching for: '{test_query}'")
    
    results = search_service.search(test_query, top_k=5)
    
    for i, res in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"ID: {res['law_id']} | Source: {res['source']}")
        print(f"FAISS Score: {res.get('faiss_score', 0):.4f}")
        print(f"Final Score (Rerank): {res['score']:.4f}")
        print(f"Content: {res['content'][:200]}...")

if __name__ == "__main__":
    main()
