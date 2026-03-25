import sys
import os
import json
import numpy as np

# Phải đặt trước khi import sentence_transformers/torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from sentence_transformers import SentenceTransformer

# Thêm thư mục gốc vào sys.path để import được app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings

def ingest_data():
    input_file = settings.DATA_PATH
    output_index = settings.INDEX_PATH
    output_metadata = settings.METADATA_PATH
    model_name = settings.EMBEDDING_MODEL
    
    print(f"[*] Đang tải dữ liệu từ {input_file}...")
    if not os.path.exists(input_file):
        print(f"Lỗi: Không tìm thấy {input_file}")
        return
        
    with open(input_file, "r", encoding="utf-8") as f:
        articles = json.load(f)
        
    print(f"[*] Đang khởi tạo model: {model_name}...")
    model = SentenceTransformer(model_name, device='cpu')
    
    # Tiền xử lý: Kết hợp law_id và content
    texts = []
    for item in articles:
        combined_text = f"Điều {item['law_id']}: {item['content']}"
        texts.append(combined_text)
        
    print(f"[*] Đang encode {len(texts)} điều luật (có thể mất vài phút)...")
    embeddings = model.encode(
        texts,
        batch_size=4,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings).astype('float32')
    
    print(f"[*] Encode xong. Shape: {embeddings.shape}")

    # Lazy import FAISS sau khi encode xong để tránh segfault do xung đột với torch
    import faiss

    # Dùng IndexFlatIP với vector đã normalize = Cosine Similarity
    # (normalize_embeddings=True đã xử lý ở bước encode, không cần normalize_L2 nữa)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    # Lưu Index và Metadata
    print(f"[*] Đang lưu index vào {output_index}...")
    faiss.write_index(index, output_index)
    
    print(f"[*] Đang lưu metadata vào {output_metadata}...")
    with open(output_metadata, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
        
    print(f"✅ Hoàn tất! Đã index {len(articles)} điều luật.")

if __name__ == "__main__":
    ingest_data()
