import os
import sys
import json

# Add root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.services.search_service import search_service
from app.services.llm_service import LLMService

def test_legal_qa(query: str):
    print(f"\n" + "="*50)
    print(f"🔍 CÂU HỎI: {query}")
    print("="*50)
    
    # 1. Tìm kiếm context
    print("[*] Đang tìm kiếm các điều luật liên quan...")
    results = search_service.search(query, top_k=5)
    
    if not results:
        print("❌ Không tìm thấy điều luật nào phù hợp.")
        return

    print(f"[*] Tìm thấy {len(results)} điều luật. Các điều luật chính:")
    for r in results:
        print(f"   - Điều {r.get('law_id')}: {r.get('title')} (Score: {r.get('score'):.4f})")

    # 2. Gọi LLM để trả lời
    print("\n[*] AI đang phân tích và trả lời...")
    llm = LLMService()
    
    # Lấy response từ Gemini
    answer, status = llm.ask_with_context(query, results)
    
    print("\n🤖 AI TRẢ LỜI:")
    print("-" * 50)
    print(answer)
    print("-" * 50)

if __name__ == "__main__":
    queries = [
        "Các nguyên tắc cơ bản của pháp luật dân sự là gì?",
        "Trẻ em 10 tuổi có được tự mình thực hiện giao dịch mua xe đạp điện trị giá 15 triệu đồng không?",
        "Người hàng xóm xây tường bao lấn sang đất nhà tôi 20cm thì tôi nên xử lý thế nào theo pháp luật?",
    ]
    
    for q in queries:
        test_legal_qa(q)
