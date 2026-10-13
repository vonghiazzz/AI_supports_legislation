import json
from pathlib import Path

def clean_duplicates():
    BASE_DIR = Path(__file__).parent.parent.parent
    eval_file = BASE_DIR / "data" / "judgments_evaluation.json"
    
    if not eval_file.exists():
        print(f"❌ Không tìm thấy file: {eval_file}")
        return
        
    with open(eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Lọc bỏ các bản ghi chứa dấu '/' trong judgment_name
    cleaned_data = [item for item in data if "/" not in item.get("judgment_name", "")]
    
    # Sắp xếp lại theo id
    for idx, item in enumerate(cleaned_data, 1):
        item["id"] = idx
        
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
        
    print(f"✅ Hoàn tất dọn dẹp!")
    print(f"   - Số bản ghi ban đầu: {len(data)}")
    print(f"   - Số bản ghi sau khi dọn dẹp: {len(cleaned_data)}")

if __name__ == "__main__":
    clean_duplicates()
