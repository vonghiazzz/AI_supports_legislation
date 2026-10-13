import os
import json
import glob
import numpy as np

# Giảm cảnh báo từ tokenizers và tránh lỗi đa luồng trên Mac/Linux
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_LAW_DIR = os.path.join(BASE_DIR, "dataLaw")
CONFIG_PATH = os.path.join(DATA_LAW_DIR, "laws_config.json")
OUTPUT_INDEX = os.path.join(DATA_DIR, "law_index.faiss")
OUTPUT_METADATA = os.path.join(DATA_DIR, "law_metadata.json")

def load_all_standardized_data() -> list:
    """Tải dữ liệu chuẩn hóa của các bộ luật dựa trên cấu hình, có fallback tự quét thư mục."""
    all_articles = []
    
    # 1. Thử load theo file cấu hình
    config = []
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception as e:
            print(f"⚠️ Không đọc được file config: {e}. Sẽ chuyển sang quét tự động.")

    if config:
        print(f"[*] Tìm thấy cấu hình cho {len(config)} bộ luật trong laws_config.json:")
        for entry in config:
            key = entry["key"]
            law_name = entry["law_name"]
            filename = f"standardized_{key}.json"
            f_path = os.path.join(DATA_DIR, filename)
            
            if os.path.exists(f_path):
                try:
                    with open(f_path, "r", encoding="utf-8") as f:
                        articles = json.load(f)
                        if isinstance(articles, list):
                            all_articles.extend(articles)
                            print(f"    - {law_name} ({filename}): {len(articles)} điều luật.")
                except Exception as e:
                    print(f"    ❌ Lỗi đọc {filename}: {e}")
            else:
                print(f"    ⚠️ Chưa có file chuẩn hóa: {law_name} ({filename})")
    else:
        # 2. Fallback quét thư mục nếu không có file cấu hình
        print("[*] Quét toàn bộ thư mục data/ tìm các file standardized_*.json...")
        search_pattern = os.path.join(DATA_DIR, "standardized_*.json")
        files = glob.glob(search_pattern)
        
        if not files:
            print("⚠️ Không tìm thấy file JSON đã chuẩn hóa nào trong data/.")
            return []
            
        for f_path in files:
            filename = os.path.basename(f_path)
            # Tránh nạp đè chính file metadata tổng hợp hoặc file nháp
            if filename in [os.path.basename(OUTPUT_METADATA)]:
                continue
            try:
                with open(f_path, "r", encoding="utf-8") as f:
                    articles = json.load(f)
                    if isinstance(articles, list):
                        all_articles.extend(articles)
                        print(f"    - {filename}: {len(articles)} điều luật.")
            except Exception as e:
                print(f"    ❌ Lỗi đọc {filename}: {e}")
                
    print(f"✅ Tổng cộng: {len(all_articles)} điều luật sẵn sàng nạp vào Vector Store.")
    return all_articles

def build_vector_store():
    articles = load_all_standardized_data()
    if not articles:
        print("⚠️ Không có điều luật nào để nạp vector store.")
        return
        
    model_name = "keepitreal/vietnamese-sbert" # Mô hình mặc định (hoặc lấy từ biến môi trường)
    print(f"\n[*] Đang khởi tạo model Embedding: {model_name}...")
    try:
        model = SentenceTransformer(model_name, device='cpu')
    except Exception as e:
        print(f"❌ Lỗi khởi tạo model: {e}")
        return
        
    texts = []
    for item in articles:
        title = item.get('title', '')
        chapter = item.get('chapter', '')
        section = item.get('section', '')
        content = item.get('content', '')
        keywords = ", ".join(item.get('keywords', []))
        tags = ", ".join(item.get('tags', []))
        
        # Kỹ thuật Semantic Chunking: Đưa thật nhiều Metadata vào chuỗi để Vector giàu ngữ nghĩa
        combined_text = f"Tiêu đề: {title}. Chương: {chapter}. Mục: {section}. Nội dung: {content}. Từ khóa: {keywords}. Thẻ: {tags}"
        texts.append(combined_text)
        
    print(f"[*] Đang encode {len(texts)} điều luật (quá trình này có thể mất vài phút)...")
    embeddings = model.encode(
        texts,
        batch_size=4,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True, # Bắt buộc True để dùng Cosine Similarity (IndexFlatIP)
    )
    embeddings = np.array(embeddings).astype('float32')
    print(f"[*] Encode xong. Ma trận Vector có kích thước: {embeddings.shape}")

    # Lazy import FAISS
    import faiss

    # Khởi tạo FAISS Index với Cosine Similarity (vì vector đã được normalize=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    # Ghi dữ liệu xuống đĩa
    print(f"[*] Đang lưu FAISS Index vào {os.path.basename(OUTPUT_INDEX)}...")
    faiss.write_index(index, OUTPUT_INDEX)
    
    print(f"[*] Đang lưu Metadata Database vào {os.path.basename(OUTPUT_METADATA)}...")
    with open(OUTPUT_METADATA, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
        
    print(f"\n🎉 HOÀN TẤT BƯỚC 3! Hệ thống đã có {len(articles)} điều luật trong Kho Vector.")

if __name__ == "__main__":
    print("=== BƯỚC 3: LOAD VECTOR STORE (FAISS) (TỔNG QUÁT) ===")
    build_vector_store()
