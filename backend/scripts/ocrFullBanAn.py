import os
import sys
import re
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# Thêm thư mục gốc vào sys.path để import cấu hình app
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR / "backend"))

# Load configuration and environment variables
load_dotenv(dotenv_path=BASE_DIR / "backend" / ".env")

try:
    from app.config import settings
    TESSERACT_PATH = settings.TESSERACT_PATH
    POPPLER_PATH = settings.POPPLER_PATH
except ImportError:
    TESSERACT_PATH = os.getenv("TESSERACT_PATH", "/Users/tannghiavo/miniconda3/bin/tesseract")
    POPPLER_PATH = os.getenv("POPPLER_PATH", "/Users/tannghiavo/miniconda3/bin")

# Cấu hình đường dẫn Tesseract cho pytesseract
import pytesseract
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def clean_page_text(text: str) -> str:
    """
    Hàm làm sạch và chuẩn hóa văn bản của từng trang sau khi trích xuất hoặc OCR.
    """
    if not text:
        return ""
        
    # 1. Chuẩn hóa Unicode (dựng sẵn)
    import unicodedata
    text = unicodedata.normalize('NFC', text)
    
    # 2. Loại bỏ các ký tự rác OCR phổ biến
    junk_patterns = [
        r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', # Non-printable ASCII
        r'AS\*+', r'7?»', r'\^V', r'\^\^', r'⁄', r'ả`', r'\?zv', r'®&', r'Ä', r'ø\)', r'©\)', r'£', r'ỹ'
    ]
    for pattern in junk_patterns:
        text = re.sub(pattern, '', text)
        
    text = text.replace('\f', '')
    
    # 3. Loại bỏ thông tin in ấn của các trang web/trình duyệt
    text = re.sub(r'(?i)\b(Firefox|Chrome|Safari|Edge|Opera)?\s*(https?://|www\.)[^\s]+', '', text)
    text = re.sub(r'(?i)\b(thuvienphapluat|luatvietnam|thukyluat|congbao)\.vn\S*', '', text)
    
    # 4. Thay đổi nhiều khoảng trắng thừa bằng 1 khoảng trắng
    text = re.sub(r'[ \t]+', ' ', text)
    
    # 5. Loại bỏ khoảng trắng ở đầu và cuối dòng
    text = '\n'.join([line.strip() for line in text.split('\n')])
    
    return text.strip()

def strip_accents(text: str) -> str:
    """
    Chuyển văn bản có dấu thành không dấu viết thường để phục vụ so khớp Regex chịu lỗi OCR.
    """
    import unicodedata
    nfkd_form = unicodedata.normalize('NFKD', text)
    only_ascii = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    # Thay thế chữ đ/Đ
    only_ascii = only_ascii.replace('đ', 'd').replace('Đ', 'd')
    only_ascii = only_ascii.replace('đ̣', 'd').replace('Đ̣', 'd')
    return only_ascii.lower()

def extract_full_pages(pdf_path: Path, force_ocr: bool = False) -> list:
    """
    Trích xuất toàn bộ các trang của file PDF.
    Ưu tiên dùng pdfplumber trước. Nếu phát hiện trang scan (text quá ngắn), 
    sẽ tự động fallback sang chạy OCR trang đó bằng pypdfium2 và pytesseract.
    
    Trả về danh sách dạng: [{"page": 1, "text": "..."}, ...]
    """
    import pdfplumber
    import pypdfium2 as pdfium
    
    pages_data = []
    
    print(f"[*] Đang đọc file: {pdf_path.name}")
    
    # Mở tài liệu bằng cả pdfplumber và pypdfium2 để sẵn sàng trích xuất lai
    try:
        pdf_plumber = pdfplumber.open(pdf_path)
        pdf_ium = pdfium.PdfDocument(str(pdf_path))
        total_pages = len(pdf_plumber.pages)
        
        for idx in tqdm(range(total_pages), desc="Trích xuất trang"):
            page_num = idx + 1
            page_text = ""
            
            # 1. Thử trích xuất trực tiếp bằng pdfplumber (nếu không force OCR)
            if not force_ocr:
                plumber_page = pdf_plumber.pages[idx]
                page_text = plumber_page.extract_text() or ""
                
            # 2. Fallback sang OCR nếu text trực tiếp quá ngắn (ví dụ < 100 ký tự) hoặc force OCR
            if force_ocr or len(page_text.strip()) < 100:
                # print(f"    - Trang {page_num}: Chạy OCR...")
                ium_page = pdf_ium[idx]
                # Render trang với scale=2 (DPI ~144) để tối ưu giữa chất lượng và tốc độ
                bitmap = ium_page.render(scale=2)
                img = bitmap.to_pil()
                
                # Thực hiện OCR tiếng Việt
                page_text = pytesseract.image_to_string(img, lang='vie')
                
            # Làm sạch dữ liệu trang
            cleaned_text = clean_page_text(page_text)
            
            pages_data.append({
                "page": page_num,
                "text": cleaned_text
            })
            
        pdf_plumber.close()
        return pages_data
        
    except Exception as e:
        print(f"❌ Lỗi trong quá trình trích xuất PDF: {e}")
        return []

def find_section_indices(full_text: str) -> tuple:
    """
    Tìm vị trí của các tiêu đề phần chính trong bản án bằng cách so khớp không dấu
    để chịu lỗi OCR sai chính tả, sai dấu.
    """
    import unicodedata
    # Chuẩn hóa văn bản sang NFC trước
    full_text_nfc = unicodedata.normalize('NFC', full_text)
    # Loại bỏ dấu tiếng Việt để so khớp chính xác
    stripped_text = strip_accents(full_text_nfc)
    
    # 1. Định vị phần "NỘI DUNG VỤ ÁN" (Facts)
    idx_facts = None
    facts_patterns = [
        # Match ở đầu dòng (khuyên dùng cho tiêu đề chính)
        r'(?im)(?:^|\n)\s*(?:noi\s+dung\s+vu\s+an|noi\s+dung\s+vu\s+viec|nhan\s+thay|toa\s+an\s+nhan\s+thay)\b',
        # Fallback tìm kiếm tự do trong toàn văn bản
        r'(?i)\b(?:noi\s+dung\s+vu\s+an|noi\s+dung\s+vu\s+viec)\b'
    ]
    for pat in facts_patterns:
        match = re.search(pat, stripped_text)
        if match:
            idx_facts = match.start()
            break
            
    # 2. Định vị phần "NHẬN ĐỊNH CỦA TÒA ÁN" (Reasoning)
    idx_reasoning = None
    start_search_r = idx_facts if idx_facts is not None else 0
    reasoning_patterns = [
        r'(?im)(?:^|\n)\s*(?:nhan\s+dinh\s+cua\s+toa\s+an|nhan\s+dinh\s+cua\s+hoi\s+dong\s+xet\s+xu|xet\s+thay)\b',
        r'(?im)(?:^|\n)\s*(?:nhan\s+dinh|xet\s+thay)\b',
        r'(?i)\b(?:nhan\s+dinh\s+cua\s+toa\s+an|nhan\s+dinh\s+cua\s+hoi\s+dong\s+xet\s+xu)\b'
    ]
    for pat in reasoning_patterns:
        matches = list(re.finditer(pat, stripped_text))
        for m in matches:
            if m.start() > start_search_r:
                idx_reasoning = m.start()
                break
        if idx_reasoning is not None:
            break
            
    # 3. Định vị phần "QUYẾT ĐỊNH" (Decision)
    idx_decision = None
    start_search_d = idx_reasoning if idx_reasoning is not None else (idx_facts if idx_facts is not None else 0)
    decision_patterns = [
        r'(?im)(?:^|\n)\s*(?:quyet\s+dinh)\b',
        r'(?i)\b(?:quyet\s+dinh)\b'
    ]
    for pat in decision_patterns:
        matches = list(re.finditer(pat, stripped_text))
        for m in matches:
            if m.start() > start_search_d:
                idx_decision = m.start()
                break
        if idx_decision is not None:
            break
            
    return idx_facts, idx_reasoning, idx_decision

def split_sections(pages: list) -> dict:
    """
    Phân đoạn nội dung bản án thành các phần: header, facts, reasoning, decision.
    Ghép các trang thành chuỗi duy nhất, tìm vị trí các phần và tiến hành cắt chuỗi.
    """
    full_text = "\n".join([p["text"] for p in pages])
    
    idx_facts, idx_reasoning, idx_decision = find_section_indices(full_text)
    
    sections = {
        "header": "",
        "facts": "",
        "reasoning": "",
        "decision": ""
    }
    
    # Gom các điểm cắt hợp lệ
    splits = []
    if idx_facts is not None:
        splits.append(("facts", idx_facts))
    if idx_reasoning is not None:
        splits.append(("reasoning", idx_reasoning))
    if idx_decision is not None:
        splits.append(("decision", idx_decision))
        
    # Sắp xếp các điểm cắt tăng dần theo index
    splits.sort(key=lambda x: x[1])
    
    current_idx = 0
    current_section = "header"
    
    for next_section, next_idx in splits:
        sections[current_section] = full_text[current_idx:next_idx].strip()
        current_idx = next_idx
        current_section = next_section
        
    sections[current_section] = full_text[current_idx:].strip()
    
    return sections

def save_json(output_data: dict, output_path: Path):
    """
    Lưu dữ liệu kết quả thành file JSON chuẩn mã hóa UTF-8.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"💾 Đã lưu cấu trúc JSON thành công vào: {output_path.relative_to(BASE_DIR) if output_path.is_relative_to(BASE_DIR) else output_path}")
    except Exception as e:
        print(f"❌ Không thể ghi file JSON: {e}")

def process_judgment_file(pdf_path: Path, output_dir: Path, force_ocr: bool = False) -> bool:
    """
    Xử lý trọn gói 1 file bản án PDF: Trích xuất trang, chia section và ghi JSON.
    """
    print(f"\n=======================================================")
    print(f"[*] ĐANG XỬ LÝ: {pdf_path.name}")
    print(f"=======================================================")
    
    # 1. Trích xuất toàn bộ các trang
    pages = extract_full_pages(pdf_path, force_ocr=force_ocr)
    if not pages:
        print(f"❌ Thất bại khi trích xuất các trang của file {pdf_path.name}")
        return False
        
    # 2. Phân đoạn nội dung bản án
    print("[*] Đang phân tích cấu trúc bản án (Header, Facts, Reasoning, Decision)...")
    sections = split_sections(pages)
    
    # 3. Gom cấu trúc dữ liệu JSON
    output_data = {
        "judgment_name": pdf_path.name,
        "page_count": len(pages),
        "pages": pages,
        "sections": sections
    }
    
    # 4. Lưu file JSON đầu ra
    output_filename = f"{pdf_path.stem}_structured.json"
    output_path = output_dir / output_filename
    save_json(output_data, output_path)
    return True

def main():
    parser = argparse.ArgumentParser(description="Tool OCR & Cấu trúc hóa bản án Việt Nam không sử dụng LLM.")
    parser.add_argument("input", help="Đường dẫn đến file PDF bản án hoặc thư mục chứa các bản án.")
    parser.add_argument("-o", "--output", help="Thư mục lưu kết quả JSON đầu ra. Mặc định lưu vào backend/data/structured_judgments/.")
    parser.add_argument("--force-ocr", action="store_true", help="Bắt buộc chạy OCR Tesseract cho tất cả các trang thay vì trích xuất trực tiếp.")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # Xác định thư mục đầu ra
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = BASE_DIR / "backend" / "data" / "structured_judgments"
        
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            print(f"❌ Chỉ hỗ trợ định dạng file PDF. Bỏ qua file: {input_path.name}")
            sys.exit(1)
        success = process_judgment_file(input_path, output_dir, args.force_ocr)
        if success:
            print("\n🎉 Hoàn thành xử lý file bản án!")
        else:
            sys.exit(1)
            
    elif input_path.is_dir():
        pdf_files = sorted(list(input_path.glob("*.pdf")))
        if not pdf_files:
            print(f"⚠️ Không tìm thấy bất kỳ file PDF nào trong thư mục: {input_path}")
            sys.exit(0)
            
        print(f"🚀 Bắt đầu xử lý hàng loạt (Batch Processing) {len(pdf_files)} file PDF bản án...")
        success_count = 0
        for pdf_file in pdf_files:
            try:
                if process_judgment_file(pdf_file, output_dir, args.force_ocr):
                    success_count += 1
            except Exception as e:
                print(f"❌ Gặp lỗi đột ngột khi xử lý {pdf_file.name}: {e}")
                
        print(f"\n🎉 Hoàn tất hàng loạt! Đã cấu trúc hóa thành công {success_count}/{len(pdf_files)} bản án.")
    else:
        print(f"❌ Đường dẫn đầu vào không hợp lệ: {args.input}")
        sys.exit(1)

if __name__ == "__main__":
    main()
