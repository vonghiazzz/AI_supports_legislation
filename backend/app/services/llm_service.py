import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import traceback
from typing import List, Optional, Tuple
from google import genai
from app.config import settings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class LLMService:
    _instance = None

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
            # Sử dụng thư viện google-genai mới
            self.client = genai.Client(api_key=self.api_key)
        
        # Cấu hình Model - Sử dụng tên chính xác từ list_models
        self.primary_model = "models/gemini-2.0-flash"
        self.fallback_model = "models/gemini-flash-latest" # Gemini 1.5 Flash (fallback ổn định)
        
        # Performance Settings
        self.score_threshold = 0.0  # Giảm threshold xuống 0.0 theo yêu cầu
        self.timeout_seconds = 20    # Timeout tối đa cho mỗi call

    def build_prompt(self, query: str, context_docs: list) -> str:
        """Xây dựng prompt theo vai trò Thẩm phán - Mẫu số 52."""
        context_str = "\n".join([
            f"Điều {doc['law_id']}: {doc['content']}" 
            for doc in context_docs
        ])

        prompt = f"""
Bạn là Thẩm phán Tòa án nhân dân.

Nhiệm vụ: Phân tích và giải quyết vụ án dựa trên:
- Hồ sơ vụ án
- Căn cứ pháp lý được cung cấp

KHÔNG được trả lời chung chung.
KHÔNG được từ chối phân tích nếu vẫn có dữ kiện.

---

I. NỘI DUNG VỤ ÁN
- Tóm tắt:
  + Quan hệ tranh chấp
  + Yêu cầu nguyên đơn
  + Lập luận bị đơn

---

II. NHẬN ĐỊNH CỦA TÒA ÁN

YÊU CẦU BẮT BUỘC:
- Phân tích theo các bước:
  1. Hợp đồng có hợp pháp không?
  2. Có vi phạm nghĩa vụ không?
  3. Bên nào có lỗi?
  4. Có căn cứ buộc tiếp tục thực hiện hợp đồng không?

- Luôn viện dẫn:
  "Căn cứ Điều X ..."

- Nếu thiếu dữ kiện:
  → nêu rõ thiếu gì (KHÔNG được kết luận ngay)

---

III. QUYẾT ĐỊNH
- Chấp nhận / bác yêu cầu
- Nghĩa vụ các bên

---

RÀNG BUỘC:
- Không được bịa điều luật
- Chỉ dùng điều luật trong context

---

HỒ SƠ:
{query}

CĂN CỨ PHÁP LÝ:
{context_str}

TRẢ LỜI:
"""
        return prompt

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception)
    )
    def call_llm(self, prompt: str, model_name: str) -> str:
        """Thực hiện call API với retry logic và timeout."""
        if not self.client:
            raise Exception("Gemini client is not initialized.")

        try:
            # Trong thư viện google-genai, config được truyền vào trực tiếp
            response = self.client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    'temperature': 0,
                    # 'timeout': self.timeout_seconds * 1000 # ms
                }
            )
            if not response or not response.text:
                raise Exception(f"Empty response from model {model_name}")
            return response.text
        except Exception as e:
            # Log lỗi chi tiết ở server
            print(f"[*] API Call Failed for {model_name}:")
            traceback.print_exc()
            raise e

    def ask_with_context(self, query: str, context_docs: list) -> Tuple[Optional[str], str]:
        """Orchestrator: Threshold check -> Primary model -> Fallback handler."""
        if not context_docs:
            return None, "failed"

        max_score = max([d.get('score', 0) for d in context_docs])

        if max_score < self.score_threshold:
            print(f"[*] Low confidence ({max_score:.4f}) but still calling LLM")

        # LUÔN build prompt
        prompt = self.build_prompt(query, context_docs)

        try:
            print(f"[*] Calling primary model: {self.primary_model}")
            answer = self.call_llm(prompt, self.primary_model)

            # Guard chống trả lời ngu
            if (
                "Không đủ căn cứ pháp lý" in answer
                and max_score > 0.05
            ):
                print("[*] Warning: weak answer despite good context")

            return answer, "success"

        except Exception:
            print(f"[*] FAILED primary. Falling back to: {self.fallback_model}")
            try:
                answer = self.call_llm(prompt, self.fallback_model)
                return answer, "success"
            except Exception:
                print("[*] CRITICAL: Both LLM models failed.")
                return None, "failed"

llm_service = LLMService()