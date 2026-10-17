"""
ChurnIQ — FastAPI Main Application

Entry point for the ChurnIQ API backend.
Configures:
- Lifespan startup (table initialization, ML bundle cache warm-up)
- CORS middleware restricted to FRONTEND_URL
- API routers for Health, Prediction, Model Metrics, and Dashboard
- Generic error handlers ensuring internal exceptions do not leak details (Rule 6)
"""

from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.app.config import PROJECT_ROOT, get_settings
from backend.app.models.database import init_db
from backend.app.services.ml_service import MLService
from backend.app.api.routes_health import router as health_router
from backend.app.api.routes_predict import router as predict_router
from backend.app.api.routes_model import router as model_router
from backend.app.api.routes_dashboard import router as dashboard_router
from backend.app.api.routes_ai import router as ai_router

# Configure server-side logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("churniq.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager: runs on startup and shutdown."""
    logger.info("Initializing ChurnIQ database tables...")
    init_db()

    logger.info("Warming up MLService model bundle and SHAP explainer cache...")
    ml_service = MLService.get_instance()
    if ml_service.model_ready:
        logger.info("MLService initialized successfully. Model ready for predictions.")
    else:
        logger.warning("ML model bundle not found. Run 'python setup_project.py' to train.")

    yield
    logger.info("ChurnIQ backend shutting down.")


settings = get_settings()

app = FastAPI(
    title="ChurnIQ API",
    description="Customer Churn Prediction, Explainability & AI-Driven Retention Platform",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS Middleware ──────────────────────────────────────────
# Allow FRONTEND_URL and local development variations
origins = [
    settings.FRONTEND_URL,
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Include Routers ──────────────────────────────────────────
app.include_router(health_router)
app.include_router(predict_router)
app.include_router(model_router)
app.include_router(dashboard_router)
app.include_router(ai_router)


# ── Exception Handlers (Rule 6) ──────────────────────────────
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return standard 422 for schema and validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()},
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Pass through known HTTP exceptions (e.g. 503, 404)."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler: logs exception internally and returns generic 500 without internals."""
    logger.error("Unhandled server exception on %s %s: %s", request.method, request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


# ── Mount Frontend Static Files ──────────────────────────────
# Enables direct in-browser access at http://127.0.0.1:8000/
frontend_dir = PROJECT_ROOT / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
