from app.config import settings


class EmbeddingService:
    _instance = None
    _model = None
    _resolved_revision = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _get_model(self):
        if self._model is None:
            print(
                "[*] Initializing frozen embedding model: "
                f"{settings.EMBEDDING_MODEL}"
            )
            print(
                "[*] Embedding revision: "
                f"{settings.EMBEDDING_REVISION}"
            )

            # Import lazily so PyTorch is initialized before FAISS.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                settings.EMBEDDING_MODEL,
                revision=settings.EMBEDDING_REVISION,
                device="cpu",
                local_files_only=settings.EMBEDDING_LOCAL_FILES_ONLY,
            )

            dimension = int(
                self._model.get_sentence_embedding_dimension()
            )

            if dimension != settings.EXPECTED_EMBEDDING_DIMENSION:
                raise RuntimeError(
                    "Embedding dimension mismatch: "
                    f"expected "
                    f"{settings.EXPECTED_EMBEDDING_DIMENSION}, "
                    f"got {dimension}"
                )

            # Reuse the same revision-verification primitive used
            # by the frozen evaluation pipeline.
            from scripts.Evaluation.reporting.manifest import (
                resolved_model_revision,
                verify_model_revision,
            )

            self._resolved_revision = resolved_model_revision(
                self._model
            )

            verify_model_revision(
                settings.EMBEDDING_REVISION,
                self._resolved_revision,
            )

            print(
                "✅ Frozen embedding model ready "
                f"(dim={dimension}, "
                f"revision={self._resolved_revision})"
            )

        return self._model

    def ensure_loaded(self):
        return self._get_model()

    @property
    def resolved_revision(self):
        return self._resolved_revision

    def encode(self, text: str):
        """
        Encode full Q0 query text.

        L2 normalization is deliberately performed by SearchService
        with FAISS, matching the frozen baseline implementation.
        """
        model = self._get_model()

        embedding = model.encode(
            [text],
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype("float32")

        if embedding.shape != (
            1,
            settings.EXPECTED_EMBEDDING_DIMENSION,
        ):
            raise RuntimeError(
                "Unexpected query embedding shape: "
                f"{embedding.shape}"
            )

        return embedding[0]


embedding_service = EmbeddingService()
