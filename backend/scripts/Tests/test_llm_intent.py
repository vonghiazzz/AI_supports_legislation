import os
import sys
from typing import Dict, Any

# Add root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.services.llm_service import llm_service

def test_intent():
    query = "Tôi và vợ kết hôn năm 2010, nay muốn ly hôn thì chia tài sản là căn nhà mua năm 2015 như thế nào?"
    print(f"[*] Testing intent extraction for: '{query}'")
    
    intent = llm_service.extract_legal_intent(query)
    
    print("\n--- Extracted Intent ---")
    print(f"Queries (Vector Search): {intent.get('queries')}")
    print(f"Legal Keywords (BM25): {intent.get('keywords')}")
    print(f"Lexical Keywords (BM25): {intent.get('lexical_keywords')}")

def test_prompt():
    query = "Ly hôn chia tài sản"
    context_docs = [
        {
            "law_id": "33",
            "title": "Nghĩa vụ chung của vợ chồng",
            "content": "Vợ chồng có nghĩa vụ thực hiện các giao dịch nhằm đáp ứng nhu cầu thiết yếu của gia đình.",
            "keywords": ["vợ chồng", "nghĩa vụ chung", "nhu cầu thiết yếu"],
            "category": "hon_nhan_gia_dinh"
        }
    ]
    prompt = llm_service.build_prompt(query, context_docs)
    print("\n--- Generated Prompt (Fragment) ---")
    print(prompt[prompt.find("CĂN CỨ PHÁP LÝ:"):])

if __name__ == "__main__":
    test_intent()
    test_prompt()
