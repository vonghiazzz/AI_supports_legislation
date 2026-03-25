import json
import re
import os

def clean_data():
    input_file = "data/law_articles.json"
    output_file = "data/cleaned_law_articles.json"
    
    print(f"[*] Đang tải dữ liệu từ {input_file}...")
    if not os.path.exists(input_file):
        print(f"Lỗi: Không tìm thấy {input_file}")
        return
        
    with open(input_file, "r", encoding="utf-8") as f:
        articles = json.load(f)
        
    cleaned_articles = []
    
    for item in articles:
        content = item.get("content", "")
        
        # 1. Loại bỏ các ký tự lạ và chuỗi rác OCR phổ biến
        junk_patterns = [
            r'AS\*+', r'7?»', r'\^V', r'\^\^', r'⁄', r'ả`', r'\?zv', r'h6', r'4S', r'®&', r'Ä', r'ø\)', r'©\)', r'£', r'ỹ', r'AS\*+', r'⁄'
        ]
        for pattern in junk_patterns:
            content = re.sub(pattern, '', content)
            
        content = content.replace('\f', '')
        
        # 2. Loại bỏ số trang (số đứng độc lập giữa các đoạn hoặc dòng)
        content = re.sub(r'\n\s*\d+\s*\n', '\n', content)
        content = re.sub(r'^\s*\d+\s*\n', '', content)
        content = re.sub(r'\n\s*\d+\s*$', '', content)
        
        # 3. Sửa lỗi chính tả
        spelling_fixes = {
            r'bình đăng': 'bình đẳng',
            r'Hiên pháp': 'Hiến pháp',
            r'Luật Nhân Sự': 'Bộ luật Dân sự',
            r'thâm quyền': 'thẩm quyền',
            r'thẩm quyên': 'thẩm quyền',
            r'quyên': 'quyền'
        }
        
        for wrong, correct in spelling_fixes.items():
            content = re.sub(wrong, correct, content, flags=re.IGNORECASE)
            
        # 4. Loại bỏ các tiêu đề chương/mục/phần nếu nó nằm ở cuối nội dung điều luật
        # Vì khi tách bằng 'Điều', các tiêu đề này thường bị dính vào cuối điều trước đó.
        structural_headers = [r'CHƯƠNG\s+[IVXLCDM]+.*', r'Mục\s+\d+.*', r'PHẦN\s+THỨ\s+\w+.*']
        for header in structural_headers:
            content = re.sub(header, '', content, flags=re.IGNORECASE | re.MULTILINE)

        # 5. Xử lý khoảng trắng và xuống dòng (thay thế nhiều bằng một)
        content = re.sub(r' +', ' ', content)
        content = re.sub(r'\n+', '\n', content)
        
        # 6. Chuẩn hóa cấu trúc: Bắt đầu bằng 'Điều [X].'
        match = re.search(r'Điều\s+\d+[\.,\s]', content)
        if match:
            content = content[match.start():]
            
        # Chỉ giữ lại law_id, content, source
        cleaned_item = {
            "law_id": item.get("law_id"),
            "content": content.strip(),
            "source": item.get("source")
        }
        cleaned_articles.append(cleaned_item)
        
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_articles, f, ensure_ascii=False, indent=2)
        
    print(f"✅ Đã dọn dẹp xong! Kết quả lưu tại {output_file} ({len(cleaned_articles)} điều luật)")

if __name__ == "__main__":
    clean_data()
