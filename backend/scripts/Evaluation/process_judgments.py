import os
import sys
import json
import time
import argparse
from pathlib import Path
from pdf2image import convert_from_path
import pytesseract

# Thêm thư mục gốc vào sys.path để có thể import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.config import settings
from app.services.llm_service import llm_service

# Thiết lập đường dẫn Tesseract từ cấu hình hệ thống
pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_PATH

# =====================================================================
# I. CÁC HÀM TIỆN ÍCH TRÍCH XUẤT (OCR & GEMINI)
# =====================================================================

def extract_first_pages_for_intent(file_path, num_pages=5):
    """OCR một số trang đầu để lấy nội dung tóm tắt vụ án dùng pypdfium2."""
    import pypdfium2 as pdfium
    try:
        doc = pdfium.PdfDocument(str(file_path))
        total_pages = len(doc)
        end_page = min(num_pages, total_pages)
        text_parts = []
        for idx in range(end_page):
            page = doc[idx]
            bitmap = page.render(scale=2)
            img = bitmap.to_pil()
            page_text = pytesseract.image_to_string(img, lang='vie')
            text_parts.append(f"--- TRANG {idx+1} ---\n{page_text}")
        return "\n".join(text_parts)
    except Exception as e:
        print(f"   ❌ Lỗi OCR trang đầu: {e}")
        return ""

def extract_text_hybrid(file_path, num_pages=5):
    """
    Trích xuất văn bản lai (Hybrid):
    1. Cố gắng trích xuất văn bản trực tiếp bằng pdfplumber.
    2. Nếu văn bản lấy ra quá ngắn (< 100 ký tự), coi như PDF là dạng ảnh quét (scanned)
       và chuyển sang OCR bằng pypdfium2 + pytesseract.
    """
    import pdfplumber
    try:
        direct_text_parts = []
        with pdfplumber.open(file_path) as pdf:
            pages_to_read = pdf.pages[:num_pages]
            for i, page in enumerate(pages_to_read):
                text = page.extract_text()
                if text:
                    direct_text_parts.append(f"--- TRANG {i+1} ---\n{text}")
        
        direct_text = "\n".join(direct_text_parts)
        if len(direct_text.strip()) >= 100:
            print(f"   ⚡ Trích xuất text trực tiếp thành công bằng pdfplumber ({len(direct_text)} ký tự).")
            return direct_text
    except Exception as e:
        print(f"   ⚠️ Lỗi khi trích xuất bằng pdfplumber: {e}. Chuyển sang OCR...")

    # Fallback to OCR if direct text is empty or too short
    print("   ⚠️ PDF không có selectable text hoặc text quá ngắn. Đang chạy OCR...")
    return extract_first_pages_for_intent(file_path, num_pages)

def extract_combined_pages_for_eval(file_path):
    """OCR kết hợp 5 trang đầu và 3 trang cuối sử dụng pypdfium2."""
    import pypdfium2 as pdfium
    try:
        doc = pdfium.PdfDocument(str(file_path))
        total_pages = len(doc)
        text_parts = []
        
        # 5 trang đầu
        end_first = min(5, total_pages)
        for idx in range(end_first):
            page = doc[idx]
            bitmap = page.render(scale=2)
            img = bitmap.to_pil()
            page_text = pytesseract.image_to_string(img, lang='vie')
            text_parts.append(f"--- TRANG {idx+1} ---\n{page_text}")
            
        # 3 trang cuối
        if total_pages > 5:
            text_parts.append("\n... [LƯỢC BỎ PHẦN GIỮA] ...\n")
            start_last = max(end_first, total_pages - 3)
            for idx in range(start_last, total_pages):
                page = doc[idx]
                bitmap = page.render(scale=2)
                img = bitmap.to_pil()
                page_text = pytesseract.image_to_string(img, lang='vie')
                text_parts.append(f"--- TRANG {idx+1} ---\n{page_text}")
                
        return "\n".join(text_parts)
    except Exception as e:
        print(f"   ❌ Lỗi OCR gom trang: {e}")
        return ""

def extract_legal_intent_with_gemini(text_content):
    """Gọi Gemini tối ưu câu hỏi truy vấn pháp lý (Legal Query) cho RAG."""
    if not text_content or len(text_content.strip()) < 20:
        return ""

    prompt = f"""
Bạn là chuyên gia pháp lý Việt Nam kiêm kỹ sư RAG. Nhiệm vụ của bạn là chuyển đổi nội dung tóm tắt vụ án dưới đây thành một câu hỏi truy vấn pháp lý (Legal Query) cô đọng, tối ưu nhất để tìm kiếm các điều luật liên quan trong cơ sở dữ liệu luật.

YÊU CẦU TỐI ƯU HÓA CÂU TRUY VẤN (QUERY):
1. KHÔNG chứa tên riêng của người (ví dụ: ông H, bà T, ông Phan Đình M). Thay thế bằng vai trò tố tụng (nguyên đơn, bị đơn, người có quyền lợi nghĩa vụ liên quan, bên bán, bên mua, bên vay, bên cho vay...).
2. KHÔNG chứa số tiền cụ thể, ngày tháng cụ thể hoặc tên công ty/địa danh cụ thể (ví dụ: thay vì "16 tỷ đồng" hoặc "biên bản ngày 20/10/2018" -> thay bằng "tiền nợ gốc và lãi", "biên bản thỏa thuận").
3. Tập trung vào vấn đề pháp lý cốt lõi (Core Legal Issue), các hành vi pháp lý và đối tượng tranh chấp (ví dụ: tranh chấp hợp đồng thế chấp quyền sử dụng đất, bồi thường thiệt hại ngoài hợp đồng do lỗi của công chứng viên, hợp đồng chuyển nhượng quyền sử dụng đất vô hiệu do giả tạo...).
4. Độ dài tối đa: 1-2 câu ngắn gọn, súc tích. Sử dụng thuật ngữ pháp lý chuẩn xác của Việt Nam.

NỘI DUNG VỤ ÁN CẦN PHÂN TÍCH:
{text_content}

YÊU CẦU ĐẦU RA:
- Chỉ trả về duy nhất câu truy vấn pháp lý đã tối ưu.
- KHÔNG giải thích, KHÔNG có phần mở đầu, KHÔNG dùng ký tự đặc biệt hay Markdown.
"""
    try:
        response = llm_service.call_llm(prompt, llm_service.primary_model)
        return response.strip()
    except Exception as e:
        print(f"   ⚠️ Lỗi Gemini (Intent): {e}")
        return ""

def extract_full_evaluation_with_gemini(text_content, judgment_name):
    """Gọi Gemini trích xuất cấu trúc JSON đầy đủ của bản án bao gồm Metadata và Ground Truth."""
    if not text_content or len(text_content.strip()) < 50:
        return None

    prompt = f"""
Bạn là chuyên gia phân tích dữ liệu pháp lý Việt Nam và xây dựng benchmark cho hệ thống Legal RAG.

Nhiệm vụ của bạn là đọc nội dung bản án OCR và trích xuất dữ liệu thành JSON chuẩn phục vụ Retrieval-Augmented Generation (RAG) cho pháp luật Việt Nam.

MỤC TIÊU:
* Tách biệt rõ:
  * luật nội dung (substantive law)
  * luật tố tụng (procedural law)
* Hỗ trợ:
  * nhiều bộ luật khác nhau
  * nhiều phiên bản luật theo năm
  * mở rộng benchmark về sau
  * retrieval đa luật (multi-law retrieval)
* KHÔNG chỉ lấy điều luật xuất hiện trong phần quyết định. Phải đọc toàn bộ:
  * phần nhận định
  * nội dung tranh chấp
  * lập luận của tòa
  * quan hệ pháp luật
  để suy luận các điều luật nội dung liên quan.

VĂN BẢN CẦN PHÂN TÍCH:
Tên bản án (PDF): {judgment_name}
Nội dung OCR:
{text_content}

---
## QUY TẮC PHÂN LOẠI

1. PROCEDURAL LAW
Là các điều luật thuộc:
* Bộ luật Tố tụng dân sự
* thủ tục giám đốc thẩm, tái thẩm, phúc thẩm, sơ thẩm, thẩm quyền, hủy án, đình chỉ, kháng nghị, xét xử lại...
Nguồn ưu tiên: phần quyết định, phần nhận định tố tụng

2. SUBSTANTIVE LAW
Là các điều luật điều chỉnh bản chất tranh chấp:
* hợp đồng, vay tài sản, bồi thường thiệt hại, đất đai, hôn nhân gia đình, thừa kế...
Nguồn: nội dung vụ án, nhận định của tòa, quan hệ pháp luật phát sinh.
Ngay cả khi điều luật KHÔNG xuất hiện trực tiếp trong bản án, hãy suy luận các điều luật phù hợp dựa trên nội dung tranh chấp.

---
## QUY TẮC source_type
Nếu điều luật xuất hiện trực tiếp trong bản án: "source_type": "explicit"
Nếu AI suy luận từ nội dung vụ án: "source_type": "inferred"

---
## CẤU TRÚC JSON BẮT BUỘC
{{
  "metadata": {{
    "case_type": "<loại tranh chấp>",
    "procedure_stage": "<so_tham|phuc_tham|giam_doc_tham|tai_tham>",
    "court_level": "<toa_an_nhan_dan|toa_an_cap_cao|toi_cao>",
    "year": <year>,
    "domain": [
      "<dan_su>",
      "<hon_nhan_gia_dinh>",
      "<dat_dai>"
    ]
  }},
  "query": "<tóm tắt vụ án sạch, dễ retrieval>",
  "query_intent": "<citation|procedural|semantic|mixed>",
  "ground_truth": {{
    "substantive": [
      {{
        "law_code": "<bo_luat_dan_su_2015>",
        "article": "<Điều 466>",
        "source_type": "<explicit|inferred>"
      }}
    ],
    "procedural": [
      {{
        "law_code": "<bo_luat_to_tung_dan_su_2015>",
        "article": "<Điều 337>",
        "source_type": "<explicit>"
      }}
    ]
  }}
}}

---
## QUY TẮC law_code
Sử dụng format snake_case chuẩn hóa. Ví dụ:
* bo_luat_dan_su_2015
* bo_luat_to_tung_dan_su_2015
* luat_dat_dai_2013

---
## QUY TẮC QUERY
Query phải loại bỏ OCR noise, dễ retrieval.
Giữ lại: hành vi pháp lý, bản chất tranh chấp, procedural stage, yêu cầu khởi kiện, quyết định của tòa.
KHÔNG copy nguyên văn toàn bộ OCR.

---
## QUY TẮC QUERY_INTENT
* citation: query chứa điều luật cụ thể
* procedural: query tập trung thủ tục tố tụng
* semantic: query là mô tả tranh chấp ngữ nghĩa
* mixed: vừa procedural vừa substantive

---
## OUTPUT
Chỉ trả về JSON hợp lệ.
Không giải thích.
Không markdown.
Không bổ sung văn bản khác.
"""
    try:
        response = llm_service.call_llm(prompt, llm_service.primary_model, is_json=True)
        clean_json = response.strip()
        if clean_json.startswith("```json"):
            clean_json = clean_json[7:]
        if clean_json.startswith("```"):
            clean_json = clean_json[3:]
        if clean_json.endswith("```"):
            clean_json = clean_json[:-3]
        
        parsed = json.loads(clean_json.strip())
        return parsed
    except Exception as e:
        print(f"   ⚠️ Lỗi Gemini hoặc Parse JSON: {e}")
        return None

# =====================================================================
# II. CÁC LUỒNG XỬ LÝ CHÍNH
# =====================================================================

def process_judgment_file(pdf_path, pdf_filename, current_id):
    """Hàm xử lý một file bản án lẻ bằng OCR + Gemini."""
    try:
        print("   [*] Đang OCR gom trang (5 đầu + 3 cuối)...")
        pages_text = extract_combined_pages_for_eval(pdf_path)
        
        if not pages_text.strip():
            print("   ⚠️ File bị lỗi ảnh/trắng tinh, không thể OCR.")
            return None
            
        print("   [*] Đang gọi Gemini trích xuất Full JSON (Quy tắc 1-8)...")
        gemini_result = None
        for attempt in range(3):
            gemini_result = extract_full_evaluation_with_gemini(pages_text, pdf_filename)
            if gemini_result:
                break
            print(f"   ⚠️ Lỗi gọi Gemini hoặc parse JSON. Thử lại lần {attempt + 1}/3 sau 10 giây...")
            time.sleep(10)
            
        if gemini_result:
            return {
                "id": current_id,
                "judgment_name": pdf_filename,
                "is_updated": True,
                "metadata": gemini_result.get("metadata", {}),
                "query": gemini_result.get("query", ""),
                "query_intent": gemini_result.get("query_intent", "mixed"),
                "ground_truth": gemini_result.get("ground_truth", {"substantive": [], "procedural": []})
            }
    except Exception as e:
        print(f"   ⚠️ Lỗi bất ngờ khi xử lý {pdf_filename}: {e}")
    return None

# Chế độ 1: Cập nhật thông minh (Incremental Update) / Rebuild toàn bộ
def run_dataset_generation(rebuild=False):
    BASE_DIR = Path(__file__).parent.parent.parent
    eval_file = BASE_DIR / "data" / "judgments_evaluation.json"
    pdf_dir = BASE_DIR / "dataLaw" / "ban_an"
    
    if not pdf_dir.exists():
        pdf_dir = BASE_DIR.parent / "dataLaw" / "ban_an"
        
    if not pdf_dir.exists():
        print(f"❌ Không tìm thấy thư mục bản án tại: {pdf_dir}")
        return

    existing_data = []
    if not rebuild and eval_file.exists():
        print(f"[*] Đọc danh sách PDF hiện tại từ {eval_file.name}")
        with open(eval_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            
    existing_filenames = set()
    max_id = 0
    for item in existing_data:
        name = item.get("judgment_name")
        if name:
            existing_filenames.add(name)
            existing_filenames.add(name.replace('/', ':'))
            existing_filenames.add(name.replace(':', '/'))
        item_id = item.get("id", 0)
        if item_id > max_id:
            max_id = item_id

    print(f"[*] Quét thư mục bản án: {pdf_dir}")
    all_pdfs = [f.name for f in pdf_dir.iterdir() if f.is_file() and f.name.endswith('.pdf')]
    
    # Lọc file mới
    target_pdfs = all_pdfs if rebuild else [f for f in all_pdfs if f not in existing_filenames]
    
    if not target_pdfs:
        print("✅ Bộ dữ liệu đã hoàn toàn đồng bộ với thư mục bản án. Không cần cập nhật thêm!")
        return

    print(f"🚀 Bắt đầu xử lý {len(target_pdfs)} file bản án...")
    
    new_data = [] if rebuild else existing_data
    success_count = 0
    
    for idx, filename in enumerate(target_pdfs):
        print(f"\n({idx+1}/{len(target_pdfs)}) Xử lý: {filename}")
        pdf_path = pdf_dir / filename
        
        # Đồng bộ ID tự tăng
        item_id = (idx + 1) if rebuild else (max_id + 1)
        
        result = process_judgment_file(pdf_path, filename, item_id)
        if result:
            new_data.append(result)
            success_count += 1
            if not rebuild:
                max_id = item_id
            
            # Lưu checkpoint thời gian thực tránh mất dữ liệu
            with open(eval_file, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, ensure_ascii=False, indent=2)
            print(f"   💾 Checkpoint: Đã cập nhật dataset tại {eval_file.name}")
            
        # Tránh rate limit của Gemini API
        time.sleep(2)
        if (idx + 1) % 10 == 0:
            print("   💤 Đang tạm nghỉ 15 giây để ổn định API và tránh Rate Limit...")
            time.sleep(15)
            
    print(f"\n🎉 HOÀN TẤT! Đã bổ sung thành công {success_count}/{len(target_pdfs)} bản ghi vào dataset.")

def run_batch_update(limit=None):
    BASE_DIR = Path(__file__).parent.parent.parent
    eval_file = BASE_DIR / "data" / "judgments_evaluation.json"
    pdf_dir = BASE_DIR / "dataLaw" / "ban_an"
    
    if not pdf_dir.exists():
        pdf_dir = BASE_DIR.parent / "dataLaw" / "ban_an"
        
    if not pdf_dir.exists():
        print(f"❌ Không tìm thấy thư mục bản án tại: {pdf_dir}")
        return

    # Load existing data
    existing_data = []
    if eval_file.exists():
        print(f"[*] Đọc danh sách PDF hiện tại từ {eval_file.name}")
        with open(eval_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            
    # Lọc ra các bản ghi chưa được cập nhật (không có is_updated=True)
    data_map = {item.get("judgment_name"): item for item in existing_data}
    
    print(f"[*] Quét thư mục bản án: {pdf_dir}")
    all_pdfs = sorted([f.name for f in pdf_dir.iterdir() if f.is_file() and f.name.endswith('.pdf')])
    
    # Target PDFs to process: files that don't have is_updated in data_map
    target_pdfs = []
    for filename in all_pdfs:
        record = data_map.get(filename)
        if not record or record.get("is_updated") != True:
            target_pdfs.append(filename)
            
    if not target_pdfs:
        print("✅ Tất cả các bản án đã được cập nhật thành công bằng bộ OCR mới!")
        return

    # Áp dụng giới hạn limit nếu có
    if limit is not None:
        target_pdfs = target_pdfs[:limit]
        print(f"[*] Chạy chế độ cuốn chiếu: Chỉ xử lý tối đa {limit} bản án trong đợt này.")

    print(f"🚀 Bắt đầu cập nhật {len(target_pdfs)} file bản án...")
    
    success_count = 0
    for idx, filename in enumerate(target_pdfs):
        print(f"\n({idx+1}/{len(target_pdfs)}) Cập nhật bản án: {filename}")
        pdf_path = pdf_dir / filename
        
        # Lấy ID hiện tại hoặc tự tăng nếu là file mới
        existing_record = data_map.get(filename)
        if existing_record:
            item_id = existing_record.get("id")
        else:
            max_id = max([item.get("id", 0) for item in existing_data]) if existing_data else 0
            item_id = max_id + 1
            
        result = process_judgment_file(pdf_path, filename, item_id)
        if result:
            result["is_updated"] = True
            
            # Cập nhật đè (Upsert) vào existing_data
            if existing_record:
                for i, item in enumerate(existing_data):
                    if item.get("judgment_name") == filename:
                        existing_data[i] = result
                        break
            else:
                existing_data.append(result)
                
            success_count += 1
            
            # Sắp xếp lại theo ID để giữ trật tự đẹp mắt
            existing_data.sort(key=lambda x: x.get("id", 0))
            
            # Lưu checkpoint
            with open(eval_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
            print(f"   💾 Checkpoint: Đã lưu cập nhật cho {filename} vào {eval_file.name}")
            
        # Tránh rate limit của Gemini API
        time.sleep(2)
        if (idx + 1) % 10 == 0:
            print("   💤 Đang tạm nghỉ 15 giây để ổn định API và tránh Rate Limit...")
            time.sleep(15)
            
    print(f"\n🎉 HOÀN TẤT ĐỢT NÀY! Đã cập nhật thành công {success_count}/{len(target_pdfs)} bản án.")

# Chế độ 2: Làm sạch các câu truy vấn cũ (Query Cleaner)
def clean_queries(force=False):
    BASE_DIR = Path(__file__).parent.parent.parent
    eval_file = BASE_DIR / "data" / "judgments_evaluation.json"
    pdf_dir = BASE_DIR / "dataLaw" / "ban_an"
    
    if not pdf_dir.exists():
        pdf_dir = BASE_DIR.parent / "dataLaw" / "ban_an"
        
    if not eval_file.exists():
        print(f"❌ Không tìm thấy file bộ dữ liệu để làm sạch: {eval_file}")
        return
        
    print(f"[*] Đang đọc file: {eval_file}")
    with open(eval_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if force:
        print(f"[*] Đang BẮT BUỘC tối ưu hóa Legal Intent cho toàn bộ {len(data)} câu hỏi bằng Gemini...")
    else:
        print(f"[*] Đang sử dụng Gemini làm sạch Legal Intent cho các câu hỏi chưa chuẩn hóa...")
    
    # Chuỗi câu hỏi trùng lặp bị lỗi do poppler render trắng trang trước đây
    dup_query = "Tranh chấp hợp đồng vay tài sản và hợp đồng thế chấp quyền sử dụng đất của bên thứ ba, tập trung vào hiệu lực của hợp đồng vay do lãi suất vượt quá quy định pháp luật và hệ quả pháp lý đối với hợp đồng thế chấp."

    updated_count = 0
    for idx, item in enumerate(data):
        pdf_filename = item.get('judgment_name')
        if not pdf_filename:
            continue
            
        pdf_path = pdf_dir / pdf_filename
        if not pdf_path.exists() and '/' in pdf_filename:
            pdf_path = pdf_dir / pdf_filename.replace('/', ':')
            
        if not pdf_path.exists():
            continue
            
        current_query = item.get('query', '')
        
        # Các dấu hiệu cho thấy query thô, chưa được chuẩn hóa
        is_raw = "NHÂN DANH" in current_query or "CỘNG HÒA XÃ HÒA" in current_query or "TÒA ÁN" in current_query
        is_failed = "không có nội dung bản án" in current_query.lower() or not current_query.strip()
        is_duplicate_fail = current_query.strip() == dup_query
        
        if not force and not is_raw and not is_failed and not is_duplicate_fail and ("tranh chấp" in current_query.lower() or "yêu cầu" in current_query.lower()):
            continue
            
        print(f"\n({idx+1}/{len(data)}) Tối ưu hóa Query cho file: {pdf_filename}")
        
        # Sử dụng trích xuất văn bản lai (Hybrid) nhanh chóng
        pages_text = extract_text_hybrid(pdf_path, num_pages=10 if is_failed or is_duplicate_fail else 5)
        
        if not pages_text.strip():
            continue
            
        intent_summary = extract_legal_intent_with_gemini(pages_text)
        if intent_summary:
            item['query'] = intent_summary
            updated_count += 1
            print(f"   ✅ Đã tối ưu query -> {intent_summary[:80]}...")
            
            if updated_count % 5 == 0:
                with open(eval_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print("   💾 Lưu checkpoint làm sạch.")
                
        # Tránh rate limit của Gemini API
        time.sleep(2)
        if (idx + 1) % 10 == 0:
            print("   💤 Đang tạm nghỉ 15 giây để ổn định API và tránh Rate Limit...")
            time.sleep(15)
                
    with open(eval_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n[*] HOÀN TẤT! Đã làm sạch/tối ưu hóa thành công {updated_count} câu hỏi.")

# =====================================================================
# III. ĐIỂM KHỞI CHẠY (MAIN)
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Quản lý và xử lý bộ dữ liệu đánh giá RAG bản án.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--rebuild", action="store_true", help="Xây dựng lại bộ dữ liệu từ đầu (ghi đè dữ liệu cũ)")
    group.add_argument("--clean-queries", action="store_true", help="Chỉ chạy làm sạch/tối ưu các câu hỏi thô trong bộ dữ liệu cũ")
    group.add_argument("--force-clean-queries", action="store_true", help="Buộc làm sạch và tối ưu lại toàn bộ câu hỏi (kể cả câu đã làm sạch)")
    group.add_argument("--update-batch", action="store_true", help="Cập nhật cuốn chiếu từng đợt thông tin Ground Truth pháp lý của bản án bằng OCR mới")
    
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số lượng bản án xử lý trong lượt này (dùng với --update-batch)")
    
    args = parser.parse_args()
    
    if args.rebuild:
        print("⚠️ CẢNH BÁO: Bạn chọn chế độ --rebuild. Toàn bộ dataset cũ sẽ bị xóa để quét lại từ đầu.")
        confirm = input("Bạn có chắc chắn muốn tiếp tục không? (y/n): ")
        if confirm.lower() == 'y':
            run_dataset_generation(rebuild=True)
        else:
            print("Đã hủy thao tác.")
    elif args.clean_queries:
        clean_queries(force=False)
    elif args.force_clean_queries:
        clean_queries(force=True)
    elif args.update_batch:
        run_batch_update(limit=args.limit)
    else:
        # Chạy mặc định: Tự phát hiện và cập nhật bản án mới (Incremental Update)
        print("[*] Chạy chế độ mặc định: Cập nhật tăng cường bản án mới...")
        run_dataset_generation(rebuild=False)

if __name__ == "__main__":
    main()
