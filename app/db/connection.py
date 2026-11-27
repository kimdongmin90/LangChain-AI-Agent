# 1127, LangChain AI Agent, kdm

from contextlib import contextmanager
from functools import lru_cache
import logging
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from app.core.settings import settings

logger = logging.getLogger(__name__)

# ============================================================
# Database Engine
# ============================================================
@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """
    PostgreSQL SQLAlchemy Engine 생성
    - Connection Pool 관리
    - Pool Pre-Ping 활성화
    - DB 연결은 실제 쿼리 실행 시 수행
    """
    if not settings.DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not configured. "
            "Check your .env file."
        )

    try:
        engine = create_engine(
            settings.DATABASE_URL,

            # Connection Pool
            pool_size=settings.DB_POOL_SIZE,
            max_overflow=settings.DB_MAX_OVERFLOW,
            pool_timeout=settings.DB_POOL_TIMEOUT,

            # 연결 유효성 검사
            pool_pre_ping=True,

            # SQL 로그 출력 비활성화
            echo=False,
        )

        logger.info("SQLAlchemy engine initialized.")
        return engine

    except Exception:
        logger.exception("Failed to initialize database engine.")
        raise

# ============================================================
# Session Factory
# ============================================================
@lru_cache(maxsize=1)
def get_session_factory():
    """
    SQLAlchemy Session Factory 생성
    """
    engine = get_engine()
    return sessionmaker(
        bind=engine,
        class_=Session,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

# ============================================================
# FastAPI Database Dependency
# ============================================================
def get_db():
    """
    FastAPI Dependency Injection용 DB Session
    사용 예시:
        @router.get("/example")
        def example(db: Session = Depends(get_db)):
            ...
    """
    SessionFactory = get_session_factory()
    db = SessionFactory()
    try:
        yield db

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()

# ============================================================
# Database Session Context Manager
# ============================================================
@contextmanager
def get_db_context():
    """
    FastAPI 외부에서 사용하는 DB Session Context Manager
    사용 예시:
        with get_db_context() as db:
            result = db.execute(...)
    """
    SessionFactory = get_session_factory()
    db = SessionFactory()
    try:
        yield db

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()

# ============================================================
# Database Health Check
# ============================================================
def check_db_connection() -> bool:
    """
    PostgreSQL 연결 상태 확인
    """
    from sqlalchemy import text
    engine = get_engine()
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        logger.info("Database connection successful.")
        return True

    except Exception:
        logger.exception("Database connection check failed.")
        return False

# ============================================================
# Database Shutdown
# ============================================================
def close_database():
    """
    SQLAlchemy Engine 및 Connection Pool 종료
    """
    if get_engine.cache_info().currsize == 0:
        return

    try:
        engine = get_engine()
        engine.dispose()

        logger.info("Database connection pool disposed.")

    finally:
        get_session_factory.cache_clear()
        get_engine.cache_clear()