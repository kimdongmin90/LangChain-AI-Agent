# 1127, LangChain AI Agent, kdm

import logging
import re
from typing import Any
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from app.db.connection import get_db_context

logger = logging.getLogger(__name__)

# ============================================================
# Repository Exceptions
# ============================================================
class RepositoryError(Exception):
    """Repository 기본 예외"""
    pass

class DataNotFoundError(RepositoryError):
    """조회 결과가 없는 경우"""
    pass

class InvalidQueryError(RepositoryError):
    """허용되지 않은 SQL 쿼리"""
    pass

# ============================================================
# SQL Validation
# ============================================================
def validate_select_query(query: str) -> None:
    """
    SELECT 전용 쿼리 검증
    주의:
    - 단순한 읽기 전용 가드
    - SQL 권한을 대체하지 않음
    """
    if not isinstance(query, str) or not query.strip():
        raise InvalidQueryError("SQL query is empty.")

    normalized = query.strip()

    # 다중 SQL 실행 방지
    if ";" in normalized.rstrip(";"):
        raise InvalidQueryError(
            "Multiple SQL statements are not allowed."
        )

    # SELECT 또는 WITH만 허용
    if not re.match(
        r"^(SELECT|WITH)\b",
        normalized,
        flags=re.IGNORECASE,
    ):
        raise InvalidQueryError(
            "Only SELECT or WITH queries are allowed."
        )

    # 위험한 SQL 명령어 차단
    forbidden = re.compile(
        r"\b("
        r"INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|"
        r"TRUNCATE|GRANT|REVOKE|CALL|COPY|"
        r"EXEC|EXECUTE|"
        r"PG_SLEEP|SET_CONFIG|"
        r"NEXTVAL|SETVAL|"
        r"LO_IMPORT|LO_EXPORT"
        r")\b",
        flags=re.IGNORECASE,
    )

    if forbidden.search(normalized):
        raise InvalidQueryError(
            "Query contains a forbidden SQL operation."
        )

    # SQL 주석 차단
    if "--" in normalized or "/*" in normalized or "*/" in normalized:
        raise InvalidQueryError(
            "SQL comments are not allowed."
        )

# ============================================================
# Query Result Converter
# ============================================================
def convert_row_to_dict(row: Any) -> dict:
    """
    SQLAlchemy Row 객체를 Dictionary로 변환
    """
    return dict(row._mapping)

# ============================================================
# Repository
# ============================================================
class DatabaseRepository:
    """
    데이터베이스 조회 전용 Repository
    모든 조회는 SQL Parameter Binding을 사용한다.
    """
    def __init__(self, db: Session | None = None):
        self.db = db

    # --------------------------------------------------------
    # Internal Execution
    # --------------------------------------------------------
    def _execute(
        self,
        query: str,
        params: dict | None = None,
    ):
        """
        SELECT 쿼리 실행
        외부 Session이 있으면 재사용하고,
        없으면 새로운 Context Session을 생성한다.
        """
        validate_select_query(query)
        try:
            if self.db is not None:
                result = self.db.execute(
                    text(query),
                    params or {},
                )

                return [
                    convert_row_to_dict(row)
                    for row in result.fetchall()
                ]

            with get_db_context() as db:
                result = db.execute(
                    text(query),
                    params or {},
                )

                return [
                    convert_row_to_dict(row)
                    for row in result.fetchall()
                ]

        except SQLAlchemyError as e:
            logger.error(
                "Database query failed: %s",
                type(e).__name__,
            )

            raise RepositoryError(
                "Database query execution failed."
            ) from e

    # --------------------------------------------------------
    # Fetch All
    # --------------------------------------------------------
    def fetch_all(
        self,
        query: str,
        params: dict | None = None,
    ) -> list[dict]:
        """
        여러 행 조회
        반환:
            list[dict]
        """
        return self._execute(query, params)

    # --------------------------------------------------------
    # Fetch One
    # --------------------------------------------------------
    def fetch_one(
        self,
        query: str,
        params: dict | None = None,
    ) -> dict:
        """
        단일 행 조회
        결과가 없으면 DataNotFoundError 발생
        """
        results = self._execute(query, params)
        if not results:
            raise DataNotFoundError(
                "No matching record found."
            )

        return results[0]

    # --------------------------------------------------------
    # Fetch Optional
    # --------------------------------------------------------
    def fetch_optional(
        self,
        query: str,
        params: dict | None = None,
    ) -> dict | None:
        """
        단일 행 조회
        결과가 없으면 None 반환
        """
        results = self._execute(query, params)
        if not results:
            return None

        return results[0]

    # --------------------------------------------------------
    # Fetch Scalar
    # --------------------------------------------------------
    def fetch_scalar(
        self,
        query: str,
        params: dict | None = None,
    ) -> Any:
        """
        첫 번째 행의 첫 번째 컬럼 반환
        """
        result = self.fetch_optional(query, params)
        if result is None:
            return None

        return next(iter(result.values()), None)