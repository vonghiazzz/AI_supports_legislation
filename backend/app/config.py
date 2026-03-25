import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Fix Mac segfault issues with PyTorch/Tokenizers
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

class Settings(BaseSettings):
    APP_NAME: str = "Law Search Engine API"
    DEBUG: bool = True
    
    # Paths (Absolute paths relative to this file's parent's parent)
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_PATH: str = os.path.join(BASE_DIR, "backend", "data", "cleaned_law_articles.json")
    INDEX_PATH: str = os.path.join(BASE_DIR, "backend", "data", "law_index.faiss")
    TESSERACT_PATH: str = os.getenv("TESSERACT_PATH", "/Users/tannghiavo/miniconda3/bin/tesseract")
    POPPLER_PATH: str = os.getenv("POPPLER_PATH", "/Users/tannghiavo/miniconda3/bin")
    
    # AI Models
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    RERANK_MODEL: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    # RERANK_MODEL: str = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    METADATA_PATH: str = os.path.join(BASE_DIR, "backend", "data", "law_metadata.json")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    class Config:
        env_file = ".env"

settings = Settings()
