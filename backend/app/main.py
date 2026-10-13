import os
# Fix Mac segfault/hang issues with PyTorch/FAISS/Tokenizers
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import shutil
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from app.config import settings
from app.models.schema import SearchRequest, SearchResult, AskRequest, AskResponse, OCRResponse
from app.services.search_service import search_service
from app.services.llm_service import llm_service
from app.services.ocr_service import ocr_service
from app.services.docx_export_service import build_legal_analysis_docx

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail fast if the frozen runtime retrieval contract cannot load.
    search_service.ensure_ready()
    yield


app = FastAPI(
    title=settings.APP_NAME,
    lifespan=lifespan,
)


# Cấu hình CORS để Frontend có thể gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production thực tế nên giới hạn domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    print(f"[*] {request.method} {request.url.path} - Time: {process_time:.4f}s")
    return response

@app.post("/ai/export-docx")
def export_docx(data: dict):
    content = str(data.get("content", "")).strip()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Nội dung báo cáo không được để trống",
        )

    try:
        file_stream, filename = build_legal_analysis_docx(content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        print(f"[ERROR][/ai/export-docx] {exc}")
        raise HTTPException(
            status_code=500,
            detail="Không thể tạo tệp DOCX",
        ) from exc

    return StreamingResponse(
        file_stream,
        media_type=(
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document"
        ),
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )


@app.post("/search", response_model=list[SearchResult])
async def search_api(req: SearchRequest):
    try:
        results = search_service.search(
            req.query,
            top_k=req.top_k,
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=AskResponse)
async def ask_api(req: AskRequest):
    try:
        start = time.time()

        # FAST PATH: Kiểm tra câu chào hỏi hoặc câu hỏi đơn giản
        greetings = ["xin chào", "chào", "hi", "hello", "chào bạn", "bắt đầu", "tư vấn giúp tôi"]
        if req.query.lower().strip() in greetings:
            return AskResponse(
                query=req.query,
                expanded_queries=[req.query],
                keywords={"keywords": [], "lexical_keywords": []},
                answer="Chào mừng bạn đến với Hệ thống Trợ lý Thẩm phán AI. Tôi đã sẵn sàng hỗ trợ bạn phân tích các hồ sơ vụ việc dựa trên Bộ luật Dân sự 2015. Bạn đang gặp tranh chấp hay thắc mắc pháp lý nào?",
                sources=[],
                llm_status="success",
                processing_time=0.001
            )

        # 1. Frozen runtime retrieval uses the original Q0 query only.
        #
        # Query expansion / LLM intent extraction is deliberately excluded
        # from runtime_retrieval_v1 so that /ask and /search share the same
        # deterministic baseline retrieval contract.
        expanded_queries = [req.query]
        keywords = {
            "keywords": [],
            "lexical_keywords": [],
        }

        # 2. Frozen Q0/D0 retrieval.
        context_results = search_service.search(
            req.query,
            top_k=req.top_k,
        )

        # 4. Generate Answer using context
        answer, status = llm_service.ask_with_context(req.query, context_results)

        # 5. Automated Evaluation of the answer
        evaluation = None
        if status == "success" and answer and context_results:
            context_text = "\n".join([f"Điều {r['law_id']}: {r['content']}" for r in context_results])
            evaluation = llm_service.evaluate_answer(req.query, context_text, answer)

        process_time = time.time() - start

        return AskResponse(
            query=req.query,
            expanded_queries=expanded_queries,
            keywords=keywords,
            answer=answer,
            sources=context_results,
            llm_status=status,
            evaluation=evaluation,
            processing_time=round(process_time, 4)
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[ERROR][/ask] {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/ocr", response_model=OCRResponse)
async def ocr_api(file: UploadFile = File(...)):
    try:
        # Save temp file
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Perform OCR
        text = ocr_service.extract_text(temp_path)
        
        # Cleanup
        os.remove(temp_path)
        
        return OCRResponse(filename=file.filename, content=text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/live")
async def health_live():
    return {
        "status": "alive",
    }


@app.get("/health/ready")
async def health_ready():
    try:
        search_service.ensure_ready()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "error": str(exc),
            },
        ) from exc

    return search_service.status()


@app.get("/health")
async def health_check():
    return await health_ready()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
