import os
import sys

# Add root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.llm_service import llm_service

def test_logic():
    query = "Hợp đồng mua bán nhà đất chỉ viết tay, không công chứng. Bên mua đã trả 2/3 tiền. Hợp đồng có vô hiệu không?"
    print(f"[*] Testing judicial logic for: '{query}'")
    
    # Giả lập context docs liên quan đến Điều 129 Bộ luật Dân sự
    context_docs = [
        {
            "law_id": "129",
            "title": "Giao dịch dân sự vô hiệu do không tuân thủ quy định về hình thức",
            "content": "Giao dịch dân sự vi phạm quy định điều kiện có hiệu lực về hình thức thì vô hiệu... Tuy nhiên, nếu một bên hoặc các bên đã thực hiện ít nhất hai phần ba nghĩa vụ trong giao dịch thì theo yêu cầu của một bên hoặc các bên, Tòa án ra quyết định công nhận hiệu lực của giao dịch đó.",
            "keywords": ["vô hiệu", "hình thức", "hai phần ba nghĩa vụ", "công nhận"],
            "score": 0.8
        }
    ]
    
    prompt = llm_service.build_prompt(query, context_docs)
    print("\n--- Generated Judicial Prompt (Fragment) ---")
    # Kiểm tra xem các rule mới có trong prompt không
    rules_check = "PHẢI xét" in prompt and "công nhận một phần" in prompt
    print(f"Rules inclusion check: {'✅ PASSED' if rules_check else '❌ FAILED'}")
    print("\n" + prompt[prompt.find("NGUYÊN TẮC BẮT BUỘC:"):prompt.find("YÊU CẦU:")])

if __name__ == "__main__":
    test_logic()
