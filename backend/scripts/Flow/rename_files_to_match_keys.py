import os
import json

def main():
    base_dir = "/Users/tannghiavo/Documents/AllProjects/AI/AI_supports_legislation"
    data_dir = os.path.join(base_dir, "backend/data")
    config_path = os.path.join(base_dir, "backend/dataLaw/laws_config.json")

    if not os.path.exists(config_path):
        print("❌ Lỗi: laws_config.json không tồn tại!")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Các fallback ánh xạ giữa key Tiếng Việt mới và key Tiếng Anh cũ
    key_fallbacks = {
        "bo_luat_dan_su_2015": ["civil_law", "law_articles", "91"],
        "bo_luat_to_tung_dan_su_2015": ["civil_procedure_law"],
        "luat_dat_dai_2024": ["land_law"],
        "luat_hon_nhan_gia_dinh_2014": ["marriage_law", "luat_hon_nhan_va_gia_dinh_2014"],
        "test_law": ["luat_thu_nghiem"]
    }

    print("=== ĐỒNG BỘ TÊN FILE DỮ LIỆU VỚI KEY CẤU HÌNH ===")
    for entry in config:
        key = entry["key"]
        law_name = entry["law_name"]
        
        raw_name = f"raw_{key}.json"
        std_name = f"standardized_{key}.json"
        
        raw_path = os.path.join(data_dir, raw_name)
        std_path = os.path.join(data_dir, std_name)
        
        # 1. Đổi tên file raw
        if not os.path.exists(raw_path):
            found = False
            for fallback_key in key_fallbacks.get(key, []):
                fallback_raw = os.path.join(data_dir, f"raw_{fallback_key}.json")
                if os.path.exists(fallback_raw):
                    os.rename(fallback_raw, raw_path)
                    print(f"✅ Đã đổi tên file raw: raw_{fallback_key}.json -> {raw_name}")
                    found = True
                    break
            if not found:
                print(f"ℹ️ Không thấy file raw cũ cho: {law_name} (chờ bóc tách mới)")
                
        # 2. Đổi tên file standardized
        if not os.path.exists(std_path):
            found = False
            for fallback_key in key_fallbacks.get(key, []):
                fallback_std = os.path.join(data_dir, f"standardized_{fallback_key}.json")
                if os.path.exists(fallback_std):
                    os.rename(fallback_std, std_path)
                    print(f"✅ Đã đổi tên file chuẩn hóa: standardized_{fallback_key}.json -> {std_name}")
                    
                    # Đổi tên cả file copy (nếu có)
                    fallback_std_copy = os.path.join(data_dir, f"standardized_{fallback_key} copy.json")
                    std_copy_path = os.path.join(data_dir, f"standardized_{key} copy.json")
                    if os.path.exists(fallback_std_copy):
                        os.rename(fallback_std_copy, std_copy_path)
                        print(f"✅ Đã đổi tên file copy: standardized_{fallback_key} copy.json -> standardized_{key} copy.json")
                        
                    found = True
                    break
            if not found:
                print(f"⚠️ Không thấy file chuẩn hóa cũ cho: {law_name} ({std_name})")

    print("\n=== DỌN DẸP CÁC FILE RÁC KHÔNG CÒN TRONG CẤU HÌNH ===")
    all_keys = {entry["key"] for entry in config}
    # Thêm các key fallback vào danh sách bỏ qua khi dọn dẹp (nếu cần giữ làm backup)
    for filename in os.listdir(data_dir):
        if filename.startswith("raw_") and filename.endswith(".json"):
            k = filename[4:-5]
            if k not in all_keys and k not in ["law_articles", "91"]: # Giữ các file raw gốc chính làm backup
                path = os.path.join(data_dir, filename)
                os.remove(path)
                print(f"🗑️ Đã xóa file raw thừa: {filename}")
        elif filename.startswith("standardized_") and filename.endswith(".json"):
            clean_name = filename.replace(" copy.json", ".json")
            k = clean_name[13:-5]
            if k not in all_keys:
                path = os.path.join(data_dir, filename)
                os.remove(path)
                print(f"🗑️ Đã xóa file chuẩn hóa thừa: {filename}")

if __name__ == "__main__":
    main()
