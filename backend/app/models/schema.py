from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)

class SearchResult(BaseModel):
    standard_law_id: Optional[str] = None
    law_id: Optional[str] = None
    title: Optional[str] = None
    content: str
    normalized_content: Optional[str] = None
    summary: Optional[str] = None
    keywords: Optional[List[str]] = None
    lexical_keywords: Optional[List[str]] = None
    legal_concepts: Optional[List[str]] = None
    category: Optional[str] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    type: Optional[str] = None
    priority: Optional[float] = None
    tags: Optional[List[str]] = None
    score: float
    source: Optional[str] = None
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None

class AskRequest(BaseModel):
    query: str
    top_k: int = Field(default=8, ge=1, le=20)

class AskResponse(BaseModel):
    query: str
    expanded_queries: Optional[List[str]] = None
    keywords: Optional[Dict[str, List[str]]] = None
    answer: Optional[str] = None
    sources: List[SearchResult]
    llm_status: str  # "success" | "failed"
    evaluation: Optional[Dict[str, Any]] = None
    processing_time: float

class OCRResponse(BaseModel):
    filename: str
    content: str
