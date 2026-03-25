from app.config import settings

class EmbeddingService:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
        return cls._instance

    def _get_model(self):
        if self._model is None:
            print(f"[*] Initializing Embedding Model: {settings.EMBEDDING_MODEL}")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(settings.EMBEDDING_MODEL, device='cpu')
        return self._model

    def encode(self, text: str):
        self._get_model()
        # Cắt query trước khi embed (tối đa 500 ký tự)
        short_text = text[:500]
        return self._model.encode([short_text], normalize_embeddings=True)[0]

embedding_service = EmbeddingService()
