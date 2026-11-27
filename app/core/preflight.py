# 1127, LangChain AI Agent, kdm

import asyncio
import logging
import torch
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from app.core.settings import settings

logger = logging.getLogger(__name__)

# ============================================================
# GPU Preflight
# ============================================================
def check_gpu() -> dict:
    """
    CUDA 및 GPU 상태 확인
    """
    embedding_device = settings.EMBEDDING_DEVICE.lower()
    reranker_device = settings.RERANKER_DEVICE.lower()
    cuda_required = (
        embedding_device.startswith("cuda")
        or reranker_device.startswith("cuda")
    )

    if not cuda_required:
        logger.info("GPU check skipped: CPU mode configured.")
        return {
            "status": "skipped",
            "device": "cpu",
        }

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required but not available. "
            "Check NVIDIA driver, CUDA and PyTorch installation."
        )

    device_index = settings.CUDA_DEVICE

    if device_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Invalid CUDA device index: {device_index}. "
            f"Available GPU count: {torch.cuda.device_count()}"
        )

    device = torch.device(f"cuda:{device_index}")
    gpu_name = torch.cuda.get_device_name(device_index)
    gpu_properties = torch.cuda.get_device_properties(device_index)
    free_memory, total_memory = torch.cuda.mem_get_info(device)
    total_gb = total_memory / (1024 ** 3)
    free_gb = free_memory / (1024 ** 3)

    allocated_gb = (
        torch.cuda.memory_allocated(device) / (1024 ** 3)
    )

    reserved_gb = (
        torch.cuda.memory_reserved(device) / (1024 ** 3)
    )

    logger.info("CUDA available: True")
    logger.info("GPU Name: %s", gpu_name)
    logger.info("GPU Total Memory: %.2f GB", total_gb)
    logger.info("GPU Free Memory: %.2f GB", free_gb)
    logger.info("GPU Allocated Memory: %.2f GB", allocated_gb)
    logger.info("GPU Reserved Memory: %.2f GB", reserved_gb)

    return {
        "status": "healthy",
        "device": str(device),
        "gpu_name": gpu_name,
        "gpu_count": torch.cuda.device_count(),
        "total_memory_gb": round(total_gb, 2),
        "free_memory_gb": round(free_gb, 2),
        "allocated_memory_gb": round(allocated_gb, 2),
        "reserved_memory_gb": round(reserved_gb, 2),
        "compute_capability": (
            f"{gpu_properties.major}.{gpu_properties.minor}"
        ),
    }

# ============================================================
# Database Preflight
# ============================================================
def _check_database_sync() -> dict:
    """
    PostgreSQL 연결 및 pgvector 확장 확인
    SQLAlchemy 동기 연결을 사용한다.
    """
    if not settings.DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not configured. "
            "Check your .env file."
        )

    engine = None

    try:
        engine = create_engine(
            settings.DATABASE_URL,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 5},
        )
        with engine.connect() as connection:
            # PostgreSQL 연결 확인
            result = connection.execute(
                text("SELECT 1")
            ).scalar_one()

            if result != 1:
                raise RuntimeError(
                    "PostgreSQL connection test failed."
                )

            # PostgreSQL 버전 확인
            version = connection.execute(
                text("SELECT version()")
            ).scalar_one()

            # pgvector 확장 확인
            vector_version = connection.execute(
                text("""
                    SELECT extversion
                    FROM pg_extension
                    WHERE extname = 'vector'
                """)
            ).scalar_one_or_none()

            if vector_version is None:
                raise RuntimeError(
                    "pgvector extension is not installed. "
                    "Install it before starting the RAG service."
                )

        logger.info("PostgreSQL connection: OK")
        logger.info("pgvector version: %s", vector_version)

        return {
            "status": "healthy",
            "connection": "connected",
            "postgresql_version": version,
            "pgvector_version": vector_version,
        }

    except SQLAlchemyError as e:
        logger.error(
            "Database preflight failed: %s",
            type(e).__name__,
        )
        raise RuntimeError(
            "PostgreSQL connection or query failed. "
            "Check DATABASE_URL and database status."
        ) from e

    finally:
        if engine is not None:
            engine.dispose()

async def check_database() -> dict:
    """
    이벤트 루프를 차단하지 않고 DB 점검 수행
    """
    return await asyncio.to_thread(
        _check_database_sync
    )

# ============================================================
# Environment Preflight
# ============================================================
def check_environment() -> dict:
    """
    필수 환경변수 확인
    """
    if not settings.APP_NAME:
        raise RuntimeError("APP_NAME is not configured.")

    if settings.PORT < 1 or settings.PORT > 65535:
        raise RuntimeError("Invalid server PORT.")

    if not 0 <= settings.LLM_TEMPERATURE <= 2:
        raise RuntimeError("Invalid LLM_TEMPERATURE.")

    if settings.RAG_TOP_K < 1:
        raise RuntimeError("RAG_TOP_K must be greater than 0.")

    if settings.RERANK_TOP_K < 1:
        raise RuntimeError(
            "RERANK_TOP_K must be greater than 0."
        )

    if not settings.OPENAI_API_KEY:
        logger.warning(
            "OPENAI_API_KEY is empty. "
            "LLM API calls will not work until configured."
        )

    logger.info("Environment configuration: OK")

    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "environment": settings.APP_ENV,
        "port": settings.PORT,
        "embedding_model": settings.EMBEDDING_MODEL,
        "reranker_model": settings.RERANKER_MODEL,
    }

# ============================================================
# Main Preflight
# ============================================================
async def run_preflight() -> dict:
    """
    애플리케이션 시작 전 전체 시스템 점검
    1. Environment
    2. GPU / CUDA
    3. PostgreSQL
    4. pgvector
    하나라도 필수 점검에 실패하면 예외를 발생시킨다.
    """

    logger.info("=" * 60)
    logger.info("Starting system preflight checks")
    logger.info("=" * 60)
    results = {}

    try:
        # 1. Environment
        logger.info("[1/3] Checking environment...")
        results["environment"] = check_environment()

        # 2. GPU
        logger.info("[2/3] Checking GPU...")
        results["gpu"] = check_gpu()

        # 3. Database
        logger.info("[3/3] Checking PostgreSQL and pgvector...")
        results["database"] = await check_database()
        logger.info("=" * 60)
        logger.info("All preflight checks completed successfully.")
        logger.info("=" * 60)

        return {
            "status": "healthy",
            "checks": results,
        }

    except Exception:
        logger.exception(
            "System preflight failed. Application startup aborted."
        )
        raise