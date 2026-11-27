# 1127, LangChain AI Agent, kdm

import logging
import threading
from functools import lru_cache
from typing import Sequence
import torch
from sentence_transformers import SentenceTransformer
from app.core.settings import settings

logger = logging.getLogger(__name__)

# ============================================================
# Embedding Configuration
# ============================================================
EMBEDDING_DIMENSION = 384
QUERY_PREFIX = "query: "
DOCUMENT_PREFIX = "passage: "

# ============================================================
# E5 Embedding Service
# ============================================================
class E5EmbeddingService:
    """
    multilingual-e5-small 기반 임베딩 서비스
    Features:
        - CUDA / CPU 지원
        - Query / Document Prefix 자동 적용
        - L2 Normalization
        - Batch Embedding
        - Lazy Model Loading
        - Thread-safe Model Initialization
    """
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self.batch_size = settings.EMBEDDING_BATCH_SIZE
        self._model = None
        self._lock = threading.Lock()

        # Device 설정
        configured_device = settings.EMBEDDING_DEVICE.lower()
        if configured_device.startswith("cuda"):

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is configured but unavailable."
                )

            device_index = settings.CUDA_DEVICE

            if device_index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"Invalid CUDA device index: {device_index}"
                )

            self.device = f"cuda:{device_index}"

        else:
            self.device = "cpu"

        logger.info(
            "Embedding service initialized: model=%s, device=%s",
            self.model_name,
            self.device,
        )

    # --------------------------------------------------------
    # Model Loading
    # --------------------------------------------------------
    def load_model(self):
        """
        모델을 최초 호출 시 한 번만 로딩한다.
        """
        if self._model is not None:
            return self._model

        with self._lock:

            if self._model is not None:
                return self._model

            logger.info(
                "Loading embedding model: %s",
                self.model_name,
            )

            model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )

            dimension = model.get_sentence_embedding_dimension()
            if dimension != EMBEDDING_DIMENSION:
                raise RuntimeError(
                    f"Unexpected embedding dimension: {dimension}. "
                    f"Expected: {EMBEDDING_DIMENSION}"
                )

            model.eval()
            self._model = model
            logger.info(
                "Embedding model loaded successfully. "
                "Dimension=%s, Device=%s",
                dimension,
                self.device,
            )

        return self._model

    # --------------------------------------------------------
    # Input Validation
    # --------------------------------------------------------
    @staticmethod
    def _validate_text(text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("Embedding input must be a string.")

        text = text.strip()
        if not text:
            raise ValueError("Embedding input cannot be empty.")

        return text

    # --------------------------------------------------------
    # Query Embedding
    # --------------------------------------------------------
    def embed_query(self, query: str) -> list[float]:
        """
        검색 질문을 벡터로 변환한다.
        E5 Query Prefix:
            query:
        """
        query = self._validate_text(query)
        model = self.load_model()
        prefixed_query = QUERY_PREFIX + query
        with torch.inference_mode():
            embedding = model.encode(
                [prefixed_query],
                batch_size=1,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

        return embedding[0].astype(float).tolist()

    # --------------------------------------------------------
    # Document Embedding
    # --------------------------------------------------------
    def embed_documents(
        self,
        documents: Sequence[str],
    ) -> list[list[float]]:
        """
        문서 여러 개를 벡터로 변환한다.
        E5 Document Prefix:
            passage:
        """
        if isinstance(documents, (str, bytes)):
            raise TypeError(
                "documents must be a sequence of strings, "
                "not a single string."
            )

        if not documents:
            return []
        validated_documents = [
            self._validate_text(document)
            for document in documents
        ]

        model = self.load_model()
        prefixed_documents = [
            DOCUMENT_PREFIX + document
            for document in validated_documents
        ]

        with torch.inference_mode():
            embeddings = model.encode(
                prefixed_documents,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

        return [
            embedding.astype(float).tolist()
            for embedding in embeddings
        ]

    # --------------------------------------------------------
    # Embedding Information
    # --------------------------------------------------------
    def get_info(self) -> dict:
        """
        임베딩 모델 설정 및 상태 반환
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "dimension": EMBEDDING_DIMENSION,
            "batch_size": self.batch_size,
            "loaded": self._model is not None,
            "normalized": True,
        }

    # --------------------------------------------------------
    # Model Unloading
    # --------------------------------------------------------
    def unload_model(self):
        """
        GPU 메모리에서 모델 해제
        """
        with self._lock:
            if self._model is None:
                return

            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Embedding model unloaded.")

# ============================================================
# Global Embedding Service
# ============================================================
@lru_cache(maxsize=1)
def get_embedding_service() -> E5EmbeddingService:
    """
    Embedding Service Singleton
    """
    return E5EmbeddingService()

# ============================================================
# Convenience Functions
# ============================================================
def embed_query(query: str) -> list[float]:
    return get_embedding_service().embed_query(query)

def embed_documents(
    documents: Sequence[str],
) -> list[list[float]]:
    return get_embedding_service().embed_documents(documents)