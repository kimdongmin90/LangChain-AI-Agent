# 1127, LangChain AI Agent, kdm

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    프로젝트 전체 환경변수 관리 클래스
    """

    # --------------------------------------------------------
    # Application
    # --------------------------------------------------------
    APP_NAME: str = "LangChain AI Report Agent"
    APP_ENV: str = "development"
    DEBUG: bool = True

    # --------------------------------------------------------
    # Server
    # --------------------------------------------------------
    HOST: str = "0.0.0.0"
    PORT: int = 8001

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------
    OPENAI_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4.1"
    LLM_TEMPERATURE: float = 0.0

    # --------------------------------------------------------
    # Database
    # --------------------------------------------------------
    DATABASE_URL: str = ""
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_TIMEOUT: int = 30

    # --------------------------------------------------------
    # Embedding
    # --------------------------------------------------------
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"
    EMBEDDING_DEVICE: str = "cuda"
    EMBEDDING_BATCH_SIZE: int = 32

    # --------------------------------------------------------
    # Reranker
    # --------------------------------------------------------
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANKER_DEVICE: str = "cuda"

    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------
    RAG_TOP_K: int = 20
    RERANK_TOP_K: int = 5

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------
    CUDA_DEVICE: int = 0

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------
    LOG_LEVEL: str = "INFO"

    # --------------------------------------------------------
    # Pydantic Settings Configuration
    # --------------------------------------------------------
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

@lru_cache
def get_settings() -> Settings:
    """
    Settings 싱글턴 객체 반환
    최초 호출 시 환경변수를 읽고
    이후 호출에서는 캐시된 객체를 반환한다.
    """
    return Settings()

# 전역 설정 객체
settings = get_settings()