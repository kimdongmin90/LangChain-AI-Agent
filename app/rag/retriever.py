# 1127, LangChain AI Agent, kdm

import json
import logging
import re
from typing import Any
from langchain_core.documents import Document
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from app.core.settings import settings
from app.db.connection import get_db_context
from app.rag.embeddings import (
    EMBEDDING_DIMENSION,
    embed_query,
)

logger = logging.getLogger(__name__)

# ============================================================
# Vector DB Configuration
# ============================================================
DEFAULT_TABLE = "documents"
ID_COLUMN = "id"
CONTENT_COLUMN = "content"
METADATA_COLUMN = "metadata"
VECTOR_COLUMN = "embedding"
MAX_TOP_K = 100

# ============================================================
# Identifier Validation
# ============================================================
def validate_identifier(identifier: str) -> str:
    """
    SQL 테이블명 및 컬럼명 검증
    SQL Parameter Binding은 테이블명이나 컬럼명에
    직접 적용할 수 없으므로 별도 검증한다.
    """
    if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", identifier):
        raise ValueError(
            f"Invalid SQL identifier: {identifier}"
        )

    return identifier

# ============================================================
# Metadata Converter
# ============================================================
def normalize_metadata(metadata: Any) -> dict:
    """
    PostgreSQL JSONB 또는 JSON 문자열을 Dictionary로 변환
    """
    if metadata is None:
        return {}

    if isinstance(metadata, dict):
        return metadata

    if isinstance(metadata, str):
        try:
            parsed = json.loads(metadata)

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError:
            logger.warning("Invalid document metadata JSON.")

    return {}

# ============================================================
# Vector Retriever
# ============================================================
class VectorRetriever:
    """
    PostgreSQL + pgvector 기반 유사도 검색
    Features:
        - E5 Query Embedding
        - Cosine Similarity
        - Top-K Retrieval
        - LangChain Document 반환
        - Metadata 지원
        - Parameterized SQL
    """
    def __init__(
        self,
        table_name: str = DEFAULT_TABLE,
        top_k: int | None = None,
    ):
        self.table_name = validate_identifier(table_name)
        self.id_column = validate_identifier(ID_COLUMN)
        self.content_column = validate_identifier(CONTENT_COLUMN)
        self.metadata_column = validate_identifier(METADATA_COLUMN)
        self.vector_column = validate_identifier(VECTOR_COLUMN)
        self.top_k = (
            top_k if top_k is not None else settings.RAG_TOP_K
        )

        self._validate_top_k(self.top_k)
        logger.info(
            "VectorRetriever initialized: table=%s, top_k=%s",
            self.table_name,
            self.top_k,
        )

    # --------------------------------------------------------
    # Top-K Validation
    # --------------------------------------------------------
    @staticmethod
    def _validate_top_k(top_k: int):
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise ValueError("top_k must be an integer.")

        if not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(
                f"top_k must be between 1 and {MAX_TOP_K}."
            )

    # --------------------------------------------------------
    # Vector Conversion
    # --------------------------------------------------------
    @staticmethod
    def _vector_to_string(vector: list[float]) -> str:
        """
        Python Vector를 PostgreSQL pgvector 형식으로 변환
        """
        if len(vector) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Invalid vector dimension: {len(vector)}. "
                f"Expected: {EMBEDDING_DIMENSION}"
            )

        return "[" + ",".join(
            str(float(value)) for value in vector
        ) + "]"

    # --------------------------------------------------------
    # SQL Generation
    # --------------------------------------------------------
    def _build_search_query(self) -> str:
        """
        Cosine Distance 기반 검색 SQL
        pgvector operator:
            <=> : Cosine Distance
        Similarity:
            1 - Cosine Distance
        """
        return f"""
            SELECT
                {self.id_column} AS document_id,
                {self.content_column} AS content,
                {self.metadata_column} AS metadata,
                {self.vector_column}
                    <=> CAST(:query_embedding AS vector({EMBEDDING_DIMENSION}))
                    AS distance
            FROM {self.table_name}
            WHERE {self.vector_column} IS NOT NULL
            ORDER BY distance ASC
            LIMIT :limit
        """

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------
    def search(
        self,
        query: str,
        k: int | None = None,
    ) -> list[Document]:
        """
        사용자 질문을 기반으로 유사 문서 검색
        Args:
            query: 검색 질문
            k: 검색할 문서 수
        Returns:
            list[Document]
        """
        search_k = k if k is not None else self.top_k
        self._validate_top_k(search_k)
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query cannot be empty.")

        # 1. Query Embedding
        query_vector = embed_query(query)
        vector_string = self._vector_to_string(query_vector)

        # 2. SQL Generation
        sql = text(self._build_search_query())

        # 3. PostgreSQL Vector Search
        try:
            with get_db_context() as db:
                result = db.execute(
                    sql,
                    {
                        "query_embedding": vector_string,
                        "limit": search_k,
                    },
                )
                rows = result.mappings().all()

            # 4. LangChain Document Conversion
            documents = []
            for row in rows:
                metadata = normalize_metadata(
                    row["metadata"]
                )
                distance = float(row["distance"])
                similarity = 1.0 - distance
                metadata.update({
                    "document_id": str(row["document_id"]),
                    "similarity": similarity,
                    "distance": distance,
                })
                document = Document(
                    page_content=str(row["content"]),
                    metadata=metadata,
                )
                documents.append(document)

            logger.info(
                "Vector search completed: query_length=%s, results=%s",
                len(query),
                len(documents),
            )

            return documents

        except SQLAlchemyError as e:
            logger.exception("Vector search database error.")
            raise RuntimeError(
                "Vector database search failed."
            ) from e

    # --------------------------------------------------------
    # Similarity Search With Score
    # --------------------------------------------------------
    def similarity_search_with_score(
        self,
        query: str,
        k: int | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Document와 Similarity Score를 함께 반환
        """
        documents = self.search(query, k)
        return [
            (
                document,
                float(document.metadata["similarity"]),
            )
            for document in documents
        ]

    # --------------------------------------------------------
    # Retriever Information
    # --------------------------------------------------------
    def get_info(self) -> dict:
        return {
            "table_name": self.table_name,
            "top_k": self.top_k,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "similarity_metric": "cosine",
            "result_type": "LangChain Document",
        }

# ============================================================
# Singleton
# ============================================================
def get_retriever() -> VectorRetriever:
    """
    기본 VectorRetriever 생성
    """
    return VectorRetriever()

# ============================================================
# Convenience Function
# ============================================================
def retrieve_documents(
    query: str,
    k: int | None = None,
) -> list[Document]:
    retriever = get_retriever()
    return retriever.search(
        query=query,
        k=k,
    )