import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


# Native-library safeguards for macOS/PyTorch/FAISS/Tokenizers.
# Must be configured before loading sentence-transformers / torch / FAISS.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
    )

    APP_NAME: str = "Law Search Engine API"
    DEBUG: bool = True

    BASE_DIR: str = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    )

    # ------------------------------------------------------------------
    # Legacy paths
    # ------------------------------------------------------------------
    # Retained only for superseded/legacy scripts.
    # Active FastAPI retrieval MUST NOT consume these artifacts.
    DATA_PATH: str = os.path.join(
        BASE_DIR,
        "backend",
        "data",
        "standardized_law_articles.json",
    )
    INDEX_PATH: str = os.path.join(
        BASE_DIR,
        "backend",
        "data",
        "law_index.faiss",
    )
    METADATA_PATH: str = os.path.join(
        BASE_DIR,
        "backend",
        "data",
        "law_metadata.json",
    )

    # ------------------------------------------------------------------
    # Frozen runtime retrieval contract v1
    # ------------------------------------------------------------------
    RUNTIME_INDEX_PATH: str = os.path.join(
        BASE_DIR,
        "backend",
        "data",
        "runtime_law_index.faiss",
    )
    RUNTIME_METADATA_PATH: str = os.path.join(
        BASE_DIR,
        "backend",
        "data",
        "runtime_law_metadata.json",
    )
    RUNTIME_MANIFEST_PATH: str = os.path.join(
        BASE_DIR,
        "backend",
        "data",
        "runtime_retrieval_manifest.json",
    )

    RUNTIME_CONTRACT_ID: str = "runtime_retrieval_v1"
    RUNTIME_PROTOCOL_ID: str = "baseline_d0_v2"

    EXPECTED_CORPUS_COUNT: int = 1811
    EXPECTED_EMBEDDING_DIMENSION: int = 768
    EXPECTED_CORPUS_FINGERPRINT: str = (
        "66fb0b20e0505993ed616f864ece99ec"
        "fad801de98b4ba3a37d20a3a7bf55e58"
    )

    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "keepitreal/vietnamese-sbert",
    )
    EMBEDDING_REVISION: str = os.getenv(
        "EMBEDDING_REVISION",
        "a9467ef2ef47caa6448edeabfd8e5e5ce0fa2a23",
    )
    EMBEDDING_LOCAL_FILES_ONLY: bool = True

    RUNTIME_POOL_K: int = 30
    RUNTIME_FUSION_METHOD: str = "minmax"
    RUNTIME_STATIC_W_BM25: float = 0.5

    # Retained for non-baseline experimental/legacy code only.
    RERANK_MODEL: str = os.getenv(
        "RERANK_MODEL",
        "BAAI/bge-reranker-v2-m3",
    )

    TESSERACT_PATH: str = os.getenv(
        "TESSERACT_PATH",
        "/Users/tannghiavo/miniconda3/bin/tesseract",
    )
    POPPLER_PATH: str = os.getenv(
        "POPPLER_PATH",
        "/Users/tannghiavo/miniconda3/bin",
    )

    GEMINI_API_KEY: str = os.getenv(
        "GEMINI_API_KEY",
        "",
    )

settings = Settings()
