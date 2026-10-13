import os
import json
import time
import sys
import random
from typing import List, Dict, Optional
from google import genai
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

class LawStandardizer:
    def __init__(self, api_key: str):
        if not api_key:
            print("❌ Error: GEMINI_API_KEY not found in environment.")
            sys.exit(1)
        self.client = genai.Client(api_key=api_key)
        self.model = "models/gemini-2.5-flash"
    def extract_title(self, content: str) -> str:
        """
        Extract title from the first line of content.
        Example: 'Điều 1. Phạm vi điều chỉnh' -> 'Phạm vi điều chỉnh'
        """
        lines = content.strip().split('\n')
        if not lines:
            return "Không có tiêu đề"
        first_line = lines[0]
        # Xử lý: 'Điều 1. Phạm vi điều chỉnh' hoặc 'Điều 1: Phạm vi...'
        for sep in ['.', ':']:
            if sep in first_line:
                parts = first_line.split(sep, 1)
                if len(parts) > 1:
                    # Bỏ phần 'Điều X'
                    return parts[1].strip()
        return first_line

    def get_enriched_metadata(self, law_id: str, content: str, retries: int = 3) -> Dict:
        """
        Sử dụng Gemini với prompt 'Chuyên gia pháp lý cao cấp' để chuẩn hóa dữ liệu RAG.
        """
        prompt = f"""
Bạn là một chuyên gia pháp lý cao cấp kiêm kỹ sư dữ liệu. Nhiệm vụ của bạn là phân tích điều luật dưới đây và chuyển đổi sang định dạng JSON chuẩn để nạp vào hệ thống RAG (Retrieval-Augmented Generation).

--- 
### 🔴 YÊU CẦU ĐỊNH DẠNG JSON (SCHEMA): 
Mỗi điều luật phải là một Object trong mảng JSON với các trường sau: 
1. "law_id": {law_id}
2. "title": Tiêu đề điều luật. 
3. "content": Nội dung đầy đủ của điều luật. 
4. "category": Định danh snake_case chuyên sâu (Ví dụ: hop_dong_vay, dai_dien_phap_nhan, thoi_hieu_khoi_kien). 
5. "chapter": Tên chương chính xác trong Bộ luật Dân sự (Ví dụ: Những quy định chung, Pháp nhân, Hợp đồng). 
6. "section": Tên mục hoặc tên nhóm quy định cụ thể (Nếu không có mục rõ ràng, hãy lấy chính Tiêu đề điều luật). 
7. "keywords": Mảng 7-10 từ khóa "Semantic Legal" (Ví dụ: "nghĩa vụ trả tiền", "lãi suất chậm trả", "bên vay tài sản"). 
8. "tags": Mảng 5-7 thẻ lý luận pháp lý để tìm kiếm nâng cao (Ví dụ: "vo_hieu_giao_dich", "trach_nhiem_dan_su", "vi_pham_nghia_vu"). 
9. "type": Phân loại quy phạm (Chọn 1 trong: nghia_vu, hieu_luc, trach_nhiem, vo_hieu, thu_tuc). 
10. "priority": Trọng số ưu tiên (Từ 0.1 đến 1.0. Quy định chung: 0.5, Quy định trọng tâm: 1.0). 

--- 
### ⚠️ QUI TẮC QUAN TRỌNG: 
- TUYỆT ĐỐI KHÔNG được trả về giá trị "unknown". Nếu nội dung ngắn, hãy dùng kiến thức luật dân sự để suy luận Chapter/Category phù hợp nhất. 
- Ngôn ngữ: Tiếng Việt. 
- Chỉ trả về duy nhất khối lượng JSON, không giải thích gì thêm.

NỘI DUNG ĐIỀU LUẬT CẦN PHÂN TÍCH:
{content}
"""
        
        for attempt in range(retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={'temperature': 0.1, 'response_mime_type': 'application/json'}
                )
                if response and response.text:
                    text = response.text.replace("```json", "").replace("```", "").strip()
                    result = json.loads(text)
                    
                    # Đảm bảo law_id và content luôn chính xác từ nguồn gốc
                    result["law_id"] = law_id
                    result["content"] = content
                    
                    # Force conversion of types if needed
                    if isinstance(result.get("priority"), str):
                        try: result["priority"] = float(result["priority"])
                        except: result["priority"] = 0.5
                    
                    return result
                else:
                    raise Exception("Empty response from Gemini")
            except Exception as e:
                if "429" in str(e):
                    wait_time = 15 + (attempt * 10)
                    print(f"🛑 Rate limit (429). Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    time.sleep(2)
                
                if attempt == retries - 1:
                    print(f"❌ Failed after {retries} retries: {e}")
                    return None
        return None

    def process_file(self, input_path: str, output_path: str, start_idx: int = 0, end_idx: int = None):
        print(f"[*] Reading input file: {input_path}")
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ Error reading input file: {e}")
            return

        if end_idx is None:
            end_idx = len(data)
            
        data_chunk = data[start_idx:end_idx]
        print(f"[*] Processing items {start_idx} to {end_idx}...")

        results = []
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                    print(f"[*] Loaded {len(results)} existing items.")
            except:
                pass

        processed_ids = {str(item['law_id']) for item in results}
        remaining_data = [item for item in data_chunk if str(item['law_id']) not in processed_ids]
        
        if not remaining_data:
            print(f"✅ All items in this range already processed.")
            return

        print(f"[*] Standardizing {len(remaining_data)} items with NEW High-Level Prompt...")
        
        try:
            for item in tqdm(remaining_data, desc="Expert Standardization"):
                # Respect rate limits for free tier if applicable
                time.sleep(2.0) 
                
                law_id = str(item['law_id'])
                content = item['content']
                
                standardized_item = self.get_enriched_metadata(law_id, content)
                
                if standardized_item:
                    # Đảm bảo chỉ lấy 10 trường như yêu cầu
                    ordered_fields = ["law_id", "title", "content", "category", "chapter", "section", "keywords", "tags", "type", "priority"]
                    final_item = {field: standardized_item.get(field) for field in ordered_fields}
                    
                    results.append(final_item)
                    
                    if len(results) % 5 == 0:
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(results, f, ensure_ascii=False, indent=2)
                else:
                    print(f"⚠️ Failed to enrich ID {law_id}")
                
                # Small delay to respect API limits
                time.sleep(0.1) 

        except KeyboardInterrupt:
            print("\n[!] Process interrupted by user. Saving progress...")
        finally:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"✅ Progress saved to {output_path}")

if __name__ == "__main__":
    # Define paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_FILE = os.path.join(BASE_DIR, "data", "cleaned_law_articles.json")
    OUTPUT_FILE = os.path.join(BASE_DIR, "data", "standardized_law_articles.json")
    
    API_KEY = os.getenv("GEMINI_API_KEY")
    
    standardizer = LawStandardizer(API_KEY)
    
    # standardizer.clean_existing_data(OUTPUT_FILE)

    # Check if range and output path were passed: python script.py start end [output_file]
    start_idx = 0
    end_idx = None
    output_file = OUTPUT_FILE
    
    if len(sys.argv) > 3:
        start_idx = int(sys.argv[1])
        end_idx = int(sys.argv[2])
        output_file = sys.argv[3]
        print(f"[*] Range: {start_idx} to {end_idx}, Output: {output_file}")
    elif len(sys.argv) > 2:
        start_idx = int(sys.argv[1])
        end_idx = int(sys.argv[2])
        print(f"[*] Range: {start_idx} to {end_idx}")
    elif len(sys.argv) > 1 and sys.argv[1].isdigit():
        end_idx = int(sys.argv[1])
        print(f"[*] Limit: {end_idx}")

    standardizer.process_file(INPUT_FILE, output_file, start_idx=start_idx, end_idx=end_idx)
