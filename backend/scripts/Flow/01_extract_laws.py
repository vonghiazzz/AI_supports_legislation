import os
import json
import re
import sys
import subprocess

try:
    import pdfplumber
except ImportError:
    print("⚠️ Thiếu thư viện pdfplumber. Chạy lệnh: pip install pdfplumber")
    sys.exit(1)

# Thiết lập đường dẫn thư mục
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_LAW_DIR = os.path.join(BASE_DIR, "dataLaw")
CONFIG_PATH = os.path.join(DATA_LAW_DIR, "laws_config.json")

def slugify(text: str) -> str:
    """Chuyển đổi text có dấu thành không dấu và dạng slug_case"""
    import unicodedata
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = text.lower()
    text = re.sub(r'[^a-z0-9_]+', '_', text)
    return text.strip('_')

def get_suggested_law_name(filename: str) -> str:
    """Tạo tên gợi ý từ tên file"""
    name_without_ext = os.path.splitext(filename)[0]
    if name_without_ext.lower().endswith('.aspx'):
        name_without_ext = name_without_ext[:-5]
    # Thay thế ký tự đặc biệt bằng khoảng trắng
    name = re.sub(r'[-_]+', ' ', name_without_ext)
    # Viết hoa chữ cái đầu tiên của mỗi từ
    return " ".join([w.capitalize() for w in name.split()])

def clean_text(content: str) -> str:
    """Làm sạch văn bản thô, dọn dẹp ký tự rác từ OCR và sửa lỗi chính tả phổ biến"""
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
    
    # 2.5. Loại bỏ thông tin in ấn trang web (URL, ngày tháng, số trang, giờ)
    # - Loại bỏ các URL bắt đầu bằng http://, https://, hoặc www. (kèm tên trình duyệt nếu có)
    content = re.sub(r'(?i)\b(Firefox|Chrome|Safari|Edge|Opera)?\s*(https?://|www\.)[^\s]+', '', content)
    # - Loại bỏ các tên miền luật phổ biến có kèm đường dẫn (ví dụ: thuvienphapluat.vn/...)
    content = re.sub(r'(?i)\b(thuvienphapluat|luatvietnam|thukyluat)\.vn\S*', '', content)
    
    # - Loại bỏ thông tin phân trang (ví dụ: "105 trong 106", "Trang 5", "Page 10 of 20", "105/106")
    content = re.sub(r'(?i)\b\d+\s+(trong|of|/)\s+\d+\b', '', content)
    content = re.sub(r'(?i)\b(Trang|Page)\s*\d+(\s*(trong|of|/)\s*\d+)?\b', '', content)
    
    # - Loại bỏ dấu mốc ngày tháng và giờ in ấn (ví dụ: "28/05/2026, 15:32", "28-05-2026", "15:32:00 PM")
    content = re.sub(r'\b\d{1,2}[-./]\d{1,2}[-./]\d{2,4}(,\s*|\s+)\d{1,2}:\d{2}(:\d{2})?(\s*[A-Z]{2})?\b', '', content)
    content = re.sub(r'\b\d{1,2}[-./]\d{1,2}[-./]\d{2,4}\b', '', content)
    content = re.sub(r'\b\d{1,2}:\d{2}(:\d{2})?(\s*[A-Z]{2})?\b', '', content)
    
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
    structural_headers = [r'CHƯƠNG\s+[IVXLCDM]+.*', r'Mục\s+\d+.*', r'PHẦN\s+THỨ\s+\w+.*']
    for header in structural_headers:
        content = re.sub(header, '', content, flags=re.IGNORECASE | re.MULTILINE)

    # 5. Xử lý khoảng trắng và xuống dòng (thay thế nhiều bằng một)
    content = re.sub(r' +', ' ', content)
    content = re.sub(r'\n+', '\n', content)
    
    return content.strip()

def extract_from_doc(file_path: str) -> str:
    print(f"[*] Extracting text from DOC: {os.path.basename(file_path)}")
    try:
        result = subprocess.run(
            ["textutil", "-convert", "txt", file_path, "-stdout"],
            capture_output=True, text=True, check=True
        )
        return result.stdout
    except Exception as e:
        print(f"❌ Lỗi đọc DOC: {e}")
        return ""

def extract_from_docx(file_path: str) -> str:
    print(f"[*] Extracting text from DOCX: {os.path.basename(file_path)}")
    try:
        from docx import Document
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        print(f"❌ Lỗi đọc DOCX: {e}")
        return ""

def extract_from_pdf_ocr(file_path: str) -> str:
    """Fallback thực hiện OCR nếu file PDF là dạng ảnh quét"""
    try:
        from app.config import settings
        tesseract_path = settings.TESSERACT_PATH
        poppler_path = settings.POPPLER_PATH
    except Exception:
        tesseract_path = "/Users/tannghiavo/miniconda3/bin/tesseract"
        poppler_path = "/Users/tannghiavo/miniconda3/bin"
        
    print(f"[*] Khởi động bộ máy OCR: Tesseract={tesseract_path}, Poppler={poppler_path}")
    
    try:
        import pytesseract
        from pdf2image import convert_from_path
        
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        print("[*] Đang chuyển đổi các trang PDF sang hình ảnh (DPI: 150 để tối ưu tốc độ)...")
        # Chuyển đổi PDF sang hình ảnh
        images = convert_from_path(
            file_path, 
            dpi=150,
            poppler_path=poppler_path
        )
        
        if not images:
            print("❌ Không thể chuyển đổi file PDF thành hình ảnh.")
            return ""
            
        full_text = []
        for i, img in enumerate(images):
            print(f"    - Đang chạy OCR trang {i+1}/{len(images)}...")
            page_text = pytesseract.image_to_string(img, lang='vie')
            full_text.append(page_text)
            
        return "\n".join(full_text)
    except ImportError:
        print("⚠️ Thiếu thư viện pytesseract hoặc pdf2image để chạy OCR. Vui lòng cài đặt: pip install pytesseract pdf2image")
        return ""
    except Exception as e:
        print(f"❌ Lỗi trong quá trình chạy OCR: {e}")
        return ""

def extract_from_pdf(file_path: str) -> str:
    print(f"[*] Extracting text from PDF: {os.path.basename(file_path)}")
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"❌ Lỗi đọc PDF: {e}")
        
    # Nếu text trích xuất được quá ngắn, khả năng cao là file scan hình ảnh -> Chạy OCR
    if len(text.strip()) < 100:
        print(f"⚠️ Phát hiện file PDF scan hoặc thiếu text. Đang chuyển sang chế độ OCR...")
        text = extract_from_pdf_ocr(file_path)
        
    return text

def extract_from_txt(file_path: str) -> str:
    print(f"[*] Extracting text from TXT: {os.path.basename(file_path)}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"❌ Lỗi đọc TXT: {e}")
        return ""

def split_into_articles(text: str, law_name: str) -> list:
    print(f"[*] Splitting text into articles for {law_name}...")
    
    # 1. Sửa các lỗi OCR đặc thù:
    # - Thay thế dấu § (thường là lỗi nhận diện của số 8) thành số 8
    text = text.replace('§', '8')
    
    # - Sửa lỗi đặc thù của Điều 595 bị nhận diện thành Điều 5
    text = re.sub(r'Điều 5\.\s+B[ằồ]i\s+[^t\n]*?thường\s+thiệt\s+hại\s+trong\s+trường\s+hợp\s+vượt\s+quá\s+yêu\s+cầu', 'Điều 595. Bồi thường thiệt hại trong trường hợp vượt quá yêu cầu', text)
    
    # - Đồng bộ các lỗi chính tả/OCR của từ "Điều" (như Điền, Diều, Biều, Diêu, Diểu, Đieu, etc. đứng trước một số)
    text = re.sub(r'\n[^\w\n]*?(Điều|Điền|Diều|Biều|Diêu|Diểu|Đieu|Điễu|Điệu|Diêu|ĐIều|ĐIỂU)\s+(\d+)\b', r'\nĐiều \2', '\n' + text, flags=re.IGNORECASE)
    
    # 2. Tách văn bản tại mỗi chữ "Điều "
    blocks = re.split(r'\n(?=Điều\s+)', text, flags=re.IGNORECASE)
    
    articles_dict = {}
    last_id = 0
    
    for block in blocks:
        block = block.strip()
        if not block:
            continue
            
        is_new_article = False
        law_id = None
        
        # Kiểm tra xem block này có phải là câu tham chiếu thông thường không (ví dụ: "Điều X của Luật này...")
        # Nếu là câu tham chiếu (được nhận diện qua các từ bổ trợ viết thường ngay sau số Điều) thì không coi là điều luật mới
        ref_pattern = r'^Điều\s+[a-zA-Z0-9]+[.\s:]+\s*(của|tại|trong|này|được|khoản|bị|đến|theo|và|với|nhưng|luật|bộ|quy\s+định|để|như|trên|dưới|khác|áp\s+dụng|sau)\b'
        is_reference = re.match(ref_pattern, block)
        
        if not is_reference:
            # Kiểm tra trùng khớp strict: "Điều X." hoặc "Điều X:" hoặc "Điều X "
            match_strict = re.match(r'^Điều\s+(\d+)\b', block, re.IGNORECASE)
            if match_strict:
                law_id = int(match_strict.group(1))
                is_new_article = True
            else:
                # Kiểm tra trùng khớp loose cho OCR lỗi hoặc chữ thay thế số
                match_loose = re.match(r'^Điều\s+([A-Za-z0-9]+)\b', block, re.IGNORECASE)
                if match_loose:
                    token = match_loose.group(1)
                    if token.isdigit():
                        law_id = int(token)
                        is_new_article = True
                    elif token[0].isupper():
                        first_line = block.split('\n')[0]
                        # Loại trừ các câu tham chiếu bình thường
                        if "của Bộ luật" not in first_line and "khoản" not in first_line:
                            is_new_article = True
                            law_id = last_id + 1
                        
        if is_new_article and law_id is not None:
            # Làm sạch dữ liệu của block
            cleaned_block = clean_text(block)
            
            # Chuẩn hóa định dạng mở đầu "Điều X. "
            cleaned_block = re.sub(r'^Điều\s+[A-Za-z0-9]+\b[^\w]*', f'Điều {law_id}. ', cleaned_block, count=1, flags=re.IGNORECASE)
            
            articles_dict[law_id] = {
                "law_id": str(law_id),
                "content": cleaned_block,
                "law_name": law_name
            }
            last_id = law_id
        else:
            # Nối tiếp vào điều luật trước đó nếu không phải điều luật mới
            if last_id in articles_dict:
                articles_dict[last_id]["content"] += "\n\n" + clean_text(block)
                
    # Chuyển đổi và sắp xếp theo thứ tự điều luật tăng dần
    sorted_keys = sorted(articles_dict.keys())
    articles = [articles_dict[k] for k in sorted_keys]
    
    print(f"✅ Bóc tách thành công {len(articles)} Điều luật duy nhất.")
    return articles

def get_base_name_and_suffix(filename: str):
    """Tách tên file thành phần tên gốc và hậu tố chỉ phần (nếu có)"""
    name_without_ext = os.path.splitext(filename)[0]
    if name_without_ext.lower().endswith('.aspx'):
        name_without_ext = name_without_ext[:-5]
    # Tìm mẫu chỉ phần ở cuối tên file, ví dụ _part1, -part-2, _phan3, _p4, _vol5, _1, -2
    suffix_pattern = r'[-_](?:part|phan|p|vol)?[-_]?\d+$'
    match = re.search(suffix_pattern, name_without_ext, re.IGNORECASE)
    if match:
        base_name = name_without_ext[:match.start()]
        suffix = match.group(0)
        return base_name, suffix
    return name_without_ext, None

def load_config() -> list:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Không thể đọc file config: {e}")
    return []

def save_config(config: list):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"💾 Đã lưu cấu hình mới vào {os.path.basename(CONFIG_PATH)}")
    except Exception as e:
        print(f"❌ Không thể lưu file config: {e}")

def sync_config_with_files(config: list) -> list:
    # Tập hợp các file đã được định cấu hình từ trước
    existing_files = set()
    for item in config:
        fn = item.get("file_name")
        if isinstance(fn, list):
            for f in fn:
                existing_files.add(f)
        elif isinstance(fn, str):
            existing_files.add(fn)
            
    updated = False
    
    if not os.path.exists(DATA_LAW_DIR):
        os.makedirs(DATA_LAW_DIR, exist_ok=True)
        return config
        
    # Lấy danh sách tất cả các file hợp lệ trên đĩa
    disk_files = []
    for filename in os.listdir(DATA_LAW_DIR):
        if filename == "laws_config.json" or filename.startswith("."):
            continue
        file_path = os.path.join(DATA_LAW_DIR, filename)
        if os.path.isdir(file_path):
            continue
        ext = os.path.splitext(filename)[1].lower()
        if ext in [".pdf", ".doc", ".docx", ".txt"]:
            disk_files.append(filename)
            
    processed_new_files = set()
    
    for filename in disk_files:
        if filename in existing_files or filename in processed_new_files:
            continue
            
        # Tìm tất cả các file có chung base name trên đĩa
        base_name, suffix = get_base_name_and_suffix(filename)
        
        # Lọc ra các file có cùng base_name
        all_parts = []
        for df in disk_files:
            df_base, _ = get_base_name_and_suffix(df)
            if df_base == base_name:
                all_parts.append(df)
                
        # Sắp xếp theo bảng chữ cái để khớp thứ tự phần 1, 2, ...
        all_parts = sorted(all_parts)
        
        # Kiểm tra xem có file nào trong nhóm này đã nằm trong config hay chưa
        existing_entry = None
        for part in all_parts:
            for item in config:
                fn = item.get("file_name")
                if (isinstance(fn, list) and part in fn) or (isinstance(fn, str) and fn == part):
                    existing_entry = item
                    break
            if existing_entry:
                break
                
        if existing_entry:
            # Nếu đã có cấu hình cũ, cập nhật danh sách file gộp mới
            old_fn = existing_entry.get("file_name")
            new_fn = all_parts if len(all_parts) > 1 else all_parts[0]
            if old_fn != new_fn:
                existing_entry["file_name"] = new_fn
                print(f"🔄 Đã cập nhật nhóm file cho bộ luật '{existing_entry['law_name']}': {new_fn}")
                updated = True
            for part in all_parts:
                processed_new_files.add(part)
        else:
            # Nếu chưa có cấu hình nào, tạo mới hoàn toàn
            law_name = get_suggested_law_name(base_name)
            key = slugify(base_name)
            
            # Tránh trùng lặp key
            existing_keys = {item["key"] for item in config}
            base_key = key
            counter = 1
            while key in existing_keys:
                key = f"{base_key}_{counter}"
                counter += 1
                
            new_entry = {
                "file_name": all_parts if len(all_parts) > 1 else all_parts[0],
                "law_name": law_name,
                "key": key
            }
            config.append(new_entry)
            print(f"✨ Phát hiện nhóm file luật mới: {all_parts}. Tự động tạo key: '{key}' và tên gợi ý: '{law_name}'")
            updated = True
            for part in all_parts:
                processed_new_files.add(part)
            
    if updated:
        save_config(config)
        
    return config

def process_law_entry(entry: dict):
    file_name = entry["file_name"]
    law_name = entry["law_name"]
    key = entry["key"]
    output_name = f"raw_{key}.json"
    
    # Hỗ trợ cả trường hợp file_name là danh sách hoặc chuỗi đơn
    file_names = file_name if isinstance(file_name, list) else [file_name]
    
    raw_texts = []
    for fn in file_names:
        file_path = os.path.join(DATA_LAW_DIR, fn)
        if not os.path.exists(file_path):
            print(f"⚠️ Bỏ qua {fn} (Không tìm thấy file trên đĩa tại {file_path})")
            continue

        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.doc':
            raw_text = extract_from_doc(file_path)
        elif ext == '.docx':
            raw_text = extract_from_docx(file_path)
        elif ext == '.pdf':
            raw_text = extract_from_pdf(file_path)
        elif ext == '.txt':
            raw_text = extract_from_txt(file_path)
        else:
            print(f"⚠️ Định dạng {ext} không được hỗ trợ cho {fn}.")
            continue
            
        if raw_text:
            raw_texts.append(raw_text)
            
    if not raw_texts:
        print(f"❌ Không trích xuất được văn bản từ bất kỳ file nào của {law_name}.")
        return False
        
    # Nối tất cả các phần lại
    combined_raw_text = "\n\n".join(raw_texts)
    articles = split_into_articles(combined_raw_text, law_name)
    
    if articles:
        out_path = os.path.join(DATA_DIR, output_name)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        print(f"💾 Đã lưu dữ liệu thô vào: {output_name}\n")
        return True
    else:
        print(f"⚠️ Không bóc tách được điều khoản nào từ {law_name}.")
        return False

def main():
    print("=== BƯỚC 1: EXTRACT DỮ LIỆU THÔ (TỔNG QUÁT TÍCH HỢP OCR & DỌN DẸP) ===")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 1. Load và đồng bộ cấu hình
    config = load_config()
    config = sync_config_with_files(config)
    
    if not config:
        print("⚠️ Không có cấu hình bộ luật nào để xử lý. Vui lòng thêm file vào dataLaw/")
        return
        
    # 2. Xử lý bóc tách từng bộ luật
    success_count = 0
    for entry in config:
        files_str = ", ".join(entry['file_name']) if isinstance(entry['file_name'], list) else entry['file_name']
        print(f"[*] Đang xử lý: {entry['law_name']} ({files_str})")
        if process_law_entry(entry):
            success_count += 1
            
    print(f"✅ HOÀN TẤT BƯỚC 1! Đã bóc tách thành công {success_count}/{len(config)} bộ luật.")

if __name__ == "__main__":
    main()
