import os
import sys
import json

# Add root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.search_service import search_service

def test():
    query = "vô hiệu do không tuân thủ hình thức"
    print(f"[*] Testing search for: '{query}'")
    
    # Trigger lazy loading
    results = search_service.search(query, top_k=3)
    
    print(f"[*] Found {len(results)} results:")
    for i, res in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"ID: {res.get('law_id')}")
        print(f"Title: {res.get('title')}")
        print(f"Category: {res.get('category')}")
        print(f"Keywords: {res.get('keywords')}")
        print(f"Score: {res.get('score'):.4f}")
        print(f"Content Summary: {res.get('content')[:100]}...")

if __name__ == "__main__":
    test()
