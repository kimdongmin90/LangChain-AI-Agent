# 1127, LangChain AI Agent, kdm

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ============================================================
# Application Lifecycle
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 애플리케이션 시작 및 종료 관리
    startup:
        - 애플리케이션 초기화
        - 추후 GPU / DB Preflight 연결
        - 추후 Agent 초기화
    shutdown:
        - DB 연결 종료
        - GPU 및 모델 리소스 정리
    """
    print("[INFO] LangChain AI Agent starting...")
    # TODO: 추후 Preflight 연결
    # from app.core.preflight import run_preflight
    # await run_preflight()
    yield
    print("[INFO] LangChain AI Agent shutting down...")

# ============================================================
# FastAPI Application
# ============================================================
app = FastAPI(
    title="LangChain AI Report Agent",
    description=(
        "FastAPI 기반 LangChain 보고서 작성 AI Agent API"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ============================================================
# CORS Middleware
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# API Router
# ============================================================

# Agent API Router
try:
    from app.api.agent import router as agent_router
    app.include_router(
        agent_router,
        prefix="/api/v1/agent",
        tags=["Agent"],
    )

except ImportError as e:
    print(f"[WARNING] Agent router not available: {e}")

# ============================================================
# Health Check
# ============================================================
@app.get(
    "/",
    tags=["Health"],
    summary="Root Endpoint",
)
async def root():
    return {
        "status": "success",
        "message": "LangChain AI Report Agent is running",
        "version": app.version,
    }

@app.get(
    "/health",
    tags=["Health"],
    summary="Health Check",
)
async def health_check():
    return {
        "status": "healthy",
        "service": "langchain-ai-report-agent",
    }

# ============================================================
# Application Entry Point
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    )