import os
# Fix Mac segfault/hang issues with PyTorch/FAISS/Tokenizers
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from app.config import settings
from app.models.schema import SearchRequest, SearchResult, AskRequest, AskResponse, OCRResponse
from app.services.search_service import search_service
from app.services.llm_service import llm_service
from app.services.ocr_service import ocr_service

app = FastAPI(title=settings.APP_NAME)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    print(f"[*] {request.method} {request.url.path} - Time: {process_time:.4f}s")
    return response

@app.post("/search", response_model=list[SearchResult])
async def search_api(req: SearchRequest):
    try:
        results = search_service.search(req.query, top_k=req.top_k)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=AskResponse)
async def ask_api(req: AskRequest):
    try:
        start = time.time()

        # 1. Search context
        context_results = search_service.search(req.query, top_k=req.top_k)

        # 2. Call LLM
        answer, status = llm_service.ask_with_context(req.query, context_results)

        process_time = time.time() - start

        return AskResponse(
            query=req.query,
            answer=answer,
            sources=context_results,
            llm_status=status,
            processing_time=round(process_time, 4)
        )

    except Exception as e:
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

@app.get("/health")
async def health_check():
    return {"status": "healthy", "articles_loaded": len(search_service.metadata)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
