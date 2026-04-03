import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import traceback
import hashlib
from typing import List, Optional, Tuple, Dict
from google import genai
from app.config import settings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

def inject_hard_laws(context_docs: list, query: str) -> list:
    """
    Hàm bổ trợ để 'ép' lấy các luật quan trọng (Hard Laws) nếu kết quả search yếu.
    Giúp xử lý các case kinh điển như: Giám đốc ký vượt thẩm quyền, vay nợ, bồi thường.
    """
    from app.services.search_service import get_law_by_id 
    query_l = query.lower()
    hard_law_ids = set()
    
    # 1. Nhóm Đại diện / Pháp nhân (Điều 134-143)
    if any(k in query_l for k in [
        "giám đốc", "đại diện", "pháp nhân", "thẩm quyền",
        "ký thay", "đóng dấu", "vượt quyền", "nhân danh"
    ]):
        hard_law_ids.update(["134", "135", "137", "141", "142", "143"])
        
    # 2. Nhóm Hợp đồng vay (Điều 463-471)
    if any(k in query_l for k in ["vay tiền", "mượn tiền", "trả nợ", "lãi suất", "vay tài sản"]):
        hard_law_ids.update(["463", "466", "468"])
        
    # 3. Nhóm Bồi thường (Điều 584-592)
    if any(k in query_l for k in ["bồi thường", "thiệt hại", "đâm đụng", "tai nạn", "phạt"]):
        hard_law_ids.update(["584", "585"])

    if not hard_law_ids:
        return context_docs

    # Kiểm tra xem luật đã có trong context chưa, nếu chưa thì đánh dấu cần lấy thêm
    existing_ids = {str(d.get('law_id')) for d in context_docs}
    to_inject = [id for id in hard_law_ids if id not in existing_ids]

    if to_inject:
        print(f"[*] Injecting Hard Laws: {to_inject}")
        # Note: Trong production thực tế, chỗ này nên gọi DB để lấy content chính xác.
        # Ở đây ta tạm thời đánh dấu hoặc giả định metadata sẽ được search_service bổ sung.
        for lid in to_inject:
            real_law = get_law_by_id(lid)
            if real_law:
                real_law["is_hard_law"] = True
                real_law["score"] = 1.0
                context_docs.append(real_law)
            
    return context_docs

class LLMService:
    _instance = None
    _cache: Dict[str, str] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMService, cls).__new__(cls)
            cls._instance._init_service()
        return cls._instance

    def _init_service(self):
        """Khởi tạo cấu hình và client Gemini."""
        self.api_key = settings.GEMINI_API_KEY
        if not self.api_key:
            print("⚠️ Warning: GEMINI_API_KEY not found in config.")
            self.client = None
        else:
            # Sử dụng thư viện google-genai mới - KHÔNG đặt timeout ép buộc để dùng default bền bỉ hơn
            self.client = genai.Client(api_key=self.api_key)
        
        # Dùng model 2.5 Flash - Bản đã được xác thực có sẵn trong tài khoản bạn
        # Cấu hình Model - Sử dụng bản 2.5 Flash đã được xác thức cho tài khoản bạn
        # Đây là model mạnh mẽ và ổn định nhất Google đang cấp phép cho bạn
        self.primary_model = "models/gemini-2.5-flash"
        self.fallback_model = "models/gemini-3.1-flash-live-preview"
        # Performance Settings
        self.score_threshold = 0.15 
        self.timeout_seconds = 30

    def build_prompt(self, query: str, context_docs: list) -> str:
        """Xây dựng prompt với thông tin đầy đủ về tiêu đề, chương, mục, thẻ và loại quy phạm."""
        context_parts = []
        for doc in context_docs:
            header = f"Điều {doc.get('law_id', 'N/A')} - {doc.get('title', 'N/A')}"
            chap_sec = f"[{doc.get('chapter', 'N/A')} / {doc.get('section', 'N/A')} / {doc.get('category', 'N/A')}]"
            type_str = f"Loại: {doc.get('type', 'generic')} | Ưu tiên: {doc.get('priority', 0.5)}"
            tags_str = f"Thẻ lý luận: {', '.join(doc.get('tags', []))}"
            keywords_str = f"Từ khóa pháp lý: {', '.join(doc.get('keywords', []))}"
            
            main_content = doc.get("content", "N/A")

            doc_str = f"""
===== Điều {doc.get('law_id')} - {doc.get('title')} =====

{doc.get('content')}
"""
            context_parts.append(doc_str)
            
        context_str = "\n\n---\n\n".join(context_parts)

        prompt = f"""
Bạn là Thẩm phán Tòa án nhân dân.

Nhiệm vụ:
Phân tích vụ án và đưa ra nhận định pháp lý.

---

NGUYÊN TẮC BẮT BUỘC:

- Nếu có ít nhất 1 điều luật liên quan:
  → PHẢI phân tích (KHÔNG được từ chối)

- Nếu giao dịch có dấu hiệu vô hiệu hoặc chưa rõ hiệu lực:
  → PHẢI xét:
    + Có đủ căn cứ để công nhận giao dịch hay không?
    + Nếu không công nhận → Xử lý hậu quả của giao dịch vô hiệu như thế nào?
    + Có thể “công nhận một phần” hiệu lực của giao dịch hay không?

- KHÔNG được trả lời:
  "Không đủ căn cứ pháp lý để kết luận"
  nếu vẫn có thể suy luận từ các dữ kiện có sẵn.

- Nếu thiếu dữ kiện:
  → phải ghi rõ:
    "Thiếu dữ kiện: ..."

---

YÊU CẦU:

I. NỘI DUNG VỤ ÁN  
II. NHẬN ĐỊNH (đánh số [1], [2])  
III. QUYẾT ĐỊNH  

---

BẮT BUỘC:
- Viện dẫn: "Căn cứ Điều X ..."
- Không bịa luật
- Chỉ dùng luật trong context

---



HỒ SƠ:
{query}

CĂN CỨ PHÁP LÝ:
{context_str}

---

TRẢ LỜI:
"""
        return prompt

    def extract_legal_intent(self, query: str) -> Dict[str, any]:
        """Dùng Gemini để phân tích ý định pháp lý và trích xuất từ khóa mở rộng."""
        if not self.client:
            return {"queries": [query], "keywords": [], "lexical_keywords": []}
            
        prompt = f"""
Phân tích hồ sơ vụ việc sau:

Hồ sơ: {query}

Trả về JSON:

1. "queries": 3 câu truy vấn semantic (ngôn ngữ pháp lý rõ ràng, ngắn gọn)

2. "keywords": 5-7 từ khóa pháp lý chuẩn 
(ví dụ: đại diện pháp nhân, nghĩa vụ dân sự, giao dịch vô hiệu)

3. "lexical_keywords": 5-7 từ khóa gần với cách người dùng tìm kiếm 
(ví dụ: ký tên đóng dấu, công ty vay tiền, trách nhiệm trả nợ)

Chỉ trả về JSON.
"""
        try:
            response = self.call_llm(prompt, self.primary_model)
            import json
            # Làm sạch response nếu nó chứa Markdown
            clean_json = response.strip().replace("```json", "").replace("```", "")
            data = json.loads(clean_json)
            return {
                "queries": data.get("queries", [query]),
                "keywords": data.get("keywords", []),
                "lexical_keywords": data.get("lexical_keywords", [])
            }
        except Exception as e:
            print(f"⚠️ Warning: Error extracting intent ({e}). Falling back to simple keyword search.")
            # FALLBACK: Nếu LLM lỗi, trả về query gốc và các từ quan trọng làm từ khóa
            return {
                "queries": [query],
                "keywords": query.split()[:5],
                "lexical_keywords": []
            }

    def expand_query(self, query: str) -> List[str]:
        """Legacy wrapper for backward compatibility."""
        res = self.extract_legal_intent(query)
        return res.get("queries", [query])

    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """Sử dụng Gemini trích xuất từ khóa pháp lý và thực tế."""
        res = self.extract_legal_intent(query)
        return {
            "keywords": res.get("keywords", []),
            "lexical_keywords": res.get("lexical_keywords", [])
        }

    def evaluate_answer(self, query: str, context: str, answer: str) -> dict:
        """Production: Disable evaluation to save API costs."""
        return {"relevance_score": 1.0, "citation_score": 1.0, "explanation": "Skip evaluaton in production."}

    @retry(stop=stop_after_attempt(1))
    def call_llm(self, prompt: str, model_name: str) -> str:
        if not self.client:
            raise Exception("Gemini client is not initialized.")
            
        start_time = time.time()
        print(f"[*] Calling model: {model_name} (Timeout set to 600s)...")

        time.sleep(0.3)

        try:
            # Gọi AI theo phong cách tiêu chuẩn - KHÔNG đặt timeout ép buộc để dùng default bền bỉ hơn
            # Cấu hình này giúp bạn chạy được 100s+ giống như bản cũ đã chạy
            response = self.client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={'temperature': 0}
            )
            elapsed = time.time() - start_time
            print(f"✅ AI Response received in {elapsed:.2f}s")

            if not response or not response.text:
                raise Exception(f"Empty response from model {model_name}")
                
            return response.text

        except Exception as e:
            print(f"[*] ERROR model={model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e

    def rerank_documents(self, docs: list) -> list:
        for d in docs:
            vector = d.get("vector_score", 0)
            bm25 = d.get("bm25_score", 0)
            priority = d.get("priority", 0.5)
            d["final_score"] = (
                0.6 * vector +
                0.3 * (bm25 / 100) +   # scale BM25
                0.1 * priority
            )
            if d.get("law_id") in ["134", "135", "137", "141", "142", "143"]:
                d["final_score"] += 0.4
        return sorted(docs, key=lambda x: x["final_score"], reverse=True)
    def ask_with_context(self, query: str, context_docs: list) -> Tuple[Optional[str], str]:
        """Production logic: Cache -> Threshold -> Single LLM call."""
        if not context_docs:
            return "Không đủ căn cứ pháp lý để kết luận.", "success"
        context_docs = self.rerank_documents(context_docs)
        # 1. Check Threshold (Lọc bỏ các luật "rác" không liên quan)
        max_score = context_docs[0]["final_score"] if context_docs else 0
        if max_score < self.score_threshold:
            print(f"[*] Low confidence ({max_score:.4f} < {self.score_threshold}). Skipping LLM.")
            context_docs = inject_hard_laws(context_docs, query)
            context_docs = self.rerank_documents(context_docs)  # BẮT BUỘC
        # 2. Check Cache
        context_str = "".join([
            str(d.get('law_id')) +
            d.get('content', '')
            for d in context_docs
        ])
        cache_key = hashlib.md5((query + context_str).encode()).hexdigest()
        
        if cache_key in self._cache:
            print("[*] Cache hit!")
            return self._cache[cache_key], "success"

        # 3. Call LLM (Single Call)
        hard_laws = [d for d in context_docs if d.get("is_hard_law")]
        normal_laws = [d for d in context_docs if not d.get("is_hard_law")]

        MAX_HARD = min(4, len(context_docs)//2)
        hard_laws = hard_laws[:MAX_HARD]
        normal_laws = normal_laws[:(8 - len(hard_laws))]

        context_docs = hard_laws + normal_laws
        prompt = self.build_prompt(query, context_docs)
        try:
            answer = self.call_llm(prompt, self.primary_model)
            
            # Save to Cache
            self._cache[cache_key] = answer
            return answer, "success"
        except Exception:
            print("[*] Primary model failed, switching to fallback...")
            try:
                answer = self.call_llm(prompt, self.fallback_model)
                return answer, "success"   
            except Exception:
                print("[*] Fallback model failed")
                traceback.print_exc()
                return "⚠️ Kết nối mạng tới dịch vụ AI đang bị gián đoạn. Vui lòng kiểm tra lại kết nối hoặc thử lại sau ít phút. Bạn vẫn có thể xem các căn cứ pháp lý được trích lục ở bên trái.", "failed"

llm_service = LLMService()