import os
import json
import time
import sys
from typing import Dict, Optional
from google import genai
from google.genai.errors import APIError
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_LAW_DIR = os.path.join(BASE_DIR, "dataLaw")
CONFIG_PATH = os.path.join(DATA_LAW_DIR, "laws_config.json")

class LawTransformer:
    def __init__(self, api_key: str):
        if not api_key:
            print("❌ Lỗi: Không tìm thấy GEMINI_API_KEY trong file .env")
            sys.exit(1)
        self.client = genai.Client(api_key=api_key)
        self.model = "models/gemini-2.5-flash"

    def get_enriched_metadata(self, item: Dict, retries: int = 3) -> Optional[Dict]:
        law_id = item['law_id']
        content = item['content']
        law_name = item['law_name']

        prompt = f"""
Bạn là chuyên gia pháp lý kiêm kỹ sư dữ liệu. Nhiệm vụ: Phân tích điều luật dưới đây từ "{law_name}" và xuất ra JSON chuẩn Schema V2.

[SCHEMA BẮT BUỘC]:
{{
  "law_id": "{law_id}",
  "summary": "<Tóm tắt 1-2 câu>",
  "normalized_content": "<Viết lại nội dung bằng ngôn ngữ dễ hiểu, liền mạch>",
  "lexical_keywords": ["<từ khóa tìm kiếm 1>", "<từ khóa tìm kiếm 2>"],
  "legal_concepts": ["<khái niệm cốt lõi dạng snake_case. VD: thu_hoi_dat, boi_thuong>"],
  "keywords": ["<các từ khóa chung>"],
  "tags": ["<thẻ phân loại>"],
  "category": "<nhóm nội dung snake_case>",
  "chapter": "<Tên chương chứa điều luật này (suy luận từ nội dung nếu cần)>",
  "section": "<Tên mục chứa điều luật này (nếu không có, lấy Tiêu đề điều luật)>",
  "type": "<1 trong các loại: nghia_vu, hieu_luc, trach_nhiem, thu_tuc, quy_dinh_chung>",
  "priority": 0.5,
  "law_name": "{law_name}",
  "title": "<Tiêu đề điều luật (sau chữ Điều X.)>",
  "content": {json.dumps(content)}
}}

[NỘI DUNG GỐC]:
{content}

Chỉ trả về JSON thuần hợp lệ, không giải thích gì thêm.
"""

        for attempt in range(retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={'temperature': 0.1, 'response_mime_type': 'application/json'}
                )
                if response and response.text:
                    result = json.loads(response.text)
                    result["law_id"] = law_id
                    result["content"] = content
                    result["law_name"] = law_name
                    if "priority" in result:
                        try: result["priority"] = float(result["priority"])
                        except: result["priority"] = 0.5
                    return result
            except Exception as e:
                if "503" in str(e) or "429" in str(e):
                    wait = 10 * (attempt + 1)
                    print(f"\n⚠️ Rate limit hoặc quá tải. Đang chờ {wait}s...")
                    time.sleep(wait)
                else:
                    time.sleep(2)
        return None

    def process_file(self, input_name: str, output_name: str):
        in_path = os.path.join(DATA_DIR, input_name)
        out_path = os.path.join(DATA_DIR, output_name)
        
        if not os.path.exists(in_path):
            print(f"⏭️ Bỏ qua {input_name} (Không tìm thấy file raw thô).")
            return False
            
        with open(in_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
            
        results = []
        if os.path.exists(out_path):
            try:
                with open(out_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                print(f"[*] Đã load {len(results)} điều luật đã xử lý trước đó từ {output_name}.")
            except:
                pass
                
        processed_ids = {str(item.get('law_id')) for item in results}
        remaining = [a for a in articles if str(a.get('law_id')) not in processed_ids]
        
        if not remaining:
            print(f"✅ Đã xử lý xong 100% file {input_name} -> {output_name}.")
            return True
            
        print(f"[*] Bắt đầu Transform {len(remaining)}/{len(articles)} điều luật của {input_name}...")
        
        try:
            for item in tqdm(remaining, desc=f"Transforming {input_name}"):
                standardized = self.get_enriched_metadata(item)
                if standardized:
                    results.append(standardized)
                    # Tự động lưu checkpoint sau mỗi 5 điều luật
                    if len(results) % 5 == 0:
                        with open(out_path, 'w', encoding='utf-8') as f:
                            json.dump(results, f, ensure_ascii=False, indent=2)
                time.sleep(1) # Tránh Rate Limit của Gemini free tier
        except KeyboardInterrupt:
            print("\n[!] Bị ngắt bởi người dùng. Đang lưu Checkpoint...")
            return False
        finally:
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"💾 Đã lưu dữ liệu Transform vào {output_name}\n")
            
        return True

def load_config() -> list:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Không thể đọc file config: {e}")
    return []

def main():
    print("=== BƯỚC 2: TRANSFORM LÀM GIÀU METADATA BẰNG GEMINI (TỔNG QUÁT) ===")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Lỗi: GEMINI_API_KEY chưa được thiết lập trong .env")
        sys.exit(1)
        
    config = load_config()
    if not config:
        print("⚠️ Không có cấu hình bộ luật nào trong laws_config.json.")
        return
        
    # Phân tích tham số dòng lệnh
    target_key = None
    if len(sys.argv) > 1:
        # Nếu người dùng truyền key cụ thể, ví dụ: "land_law_2024"
        arg = sys.argv[1]
        # Loại bỏ các flag nếu người dùng vô tình truyền
        if not arg.startswith("-"):
            target_key = arg
            
    transformer = LawTransformer(api_key)
    
    # Lọc danh sách bộ luật cần chạy
    if target_key:
        target_entries = [entry for entry in config if entry["key"] == target_key]
        if not target_entries:
            print(f"❌ Không tìm thấy bộ luật nào có key='{target_key}' trong laws_config.json")
            print("Các key hợp lệ:")
            for entry in config:
                print(f"  - {entry['key']} ({entry['law_name']})")
            return
    else:
        target_entries = config
        print(f"[*] Sẽ xử lý lần lượt {len(target_entries)} bộ luật từ cấu hình...")
        
    # Tiến hành transform
    success_count = 0
    for entry in target_entries:
        key = entry["key"]
        law_name = entry["law_name"]
        raw_file = f"raw_{key}.json"
        standardized_file = f"standardized_{key}.json"
        
        print(f"=== Đang chuẩn hóa: {law_name} ===")
        if transformer.process_file(raw_file, standardized_file):
            success_count += 1
            
    print(f"✅ HOÀN TẤT BƯỚC 2! Đã xử lý {success_count}/{len(target_entries)} bộ luật mục tiêu.")

if __name__ == "__main__":
    main()
