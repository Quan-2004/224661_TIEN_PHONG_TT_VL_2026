"""
FastAPI Main Application
=========================
Entry point cho backend mOC-iSVM.

Tính năng:
  - CORS config (cho phép React frontend kết nối)
  - SQLite Database (tự động tạo bảng khi khởi động)
  - Swagger UI tại /docs
  - ReDoc tại /redoc

Chạy:
    uvicorn backend.main:app --reload --port 8000
"""

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Thêm root directory vào sys.path để import mocsvm
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.routers import upload, train, models_router, predict
from backend.routers import upload_raw, auto_train, test_csv
from backend.schemas import ModelsListResponse
from mocsvm.core.manifest_manager import ManifestManager


# --------------------------------------------------------------------------
# Lifespan event – chạy khi khởi động / tắt server
# --------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Khởi tạo các thư mục cần thiết."""
    print("  [Server] 🚀 Đang khởi động mOC-iSVM Backend...")
    # Đảm bảo thư mục cần thiết tồn tại
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/uploads", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    print("  [Server] ✓ Server sẵn sàng!")
    yield
    print("  [Server] 🛑 Server đang tắt...")


# --------------------------------------------------------------------------
# Khởi tạo FastAPI app
# --------------------------------------------------------------------------

app = FastAPI(
    title       = "mOC-iSVM API",
    description = (
        "API cho hệ thống Modified One-Class Incremental SVM (mOC-iSVM).\n\n"
        "## Luồng sử dụng (Phase 0 → Train → Predict)\n"
        "0. **[MỚI] Upload CSV thô** → tiền xử lý tự động qua `/upload-raw`\n"
        "1. **Upload 3 CSV đã xử lý** qua `/upload`\n"
        "2. **Huấn luyện** model qua `/train`\n"
        "3. **Xem danh sách model** qua `/models`\n"
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)

# --------------------------------------------------------------------------
# CORS – Cho phép React frontend (localhost:5173 / localhost:3000) kết nối
# --------------------------------------------------------------------------

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
# Nếu có "*" trong danh sách, ta cho phép tất cả các domain
if "*" in CORS_ORIGINS:
    CORS_ORIGINS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins     = CORS_ORIGINS,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# --------------------------------------------------------------------------
# Đăng ký các routers
# --------------------------------------------------------------------------

app.include_router(upload_raw.router,     prefix="/upload-raw",  tags=["upload-raw"])
app.include_router(auto_train.router,     prefix="/auto-train",  tags=["auto-train"])
app.include_router(upload.router)
app.include_router(train.router)
app.include_router(models_router.router)
app.include_router(predict.router)
app.include_router(test_csv.router)


# --------------------------------------------------------------------------
# Root endpoint
# --------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def root():
    """Health check endpoint."""
    return {
        "status" : "ok",
        "service": "mOC-iSVM API",
        "version": "1.0.0",
        "docs"   : "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    """Kiểm tra trạng thái server và các module."""
    return JSONResponse(content={
        "status"  : "healthy",
        "models"  : os.path.exists("models/global_manifest.xml"),
    })


@app.get("/dashboard", response_model=ModelsListResponse, tags=["Dashboard"])
def dashboard():
    """
    GET /dashboard – Trả về trạng thái các phiên bản model hiện tại.
    Alias của GET /models để tuân thủ yêu cầu dự án.
    """
    manifest_path = os.getenv("MANIFEST_PATH", "models/global_manifest.xml")
    if not os.path.exists(manifest_path):
        return ModelsListResponse(total=0, last_updated="N/A", models=[])
    try:
        from backend.schemas import ModelInfo, ModelMetadata
        manager = ManifestManager(manifest_path)
        all_info = manager.list_all()
        models = []
        for item in all_info:
            meta_dict = item.get("metadata", {}) or {}
            models.append(ModelInfo(
                class_name = item.get("class_name", ""),
                version    = item.get("version", ""),
                pkl_path   = item.get("pkl_path"),
                metadata   = ModelMetadata(**{k: str(v) for k, v in meta_dict.items()}),
            ))
        return ModelsListResponse(
            total        = len(models),
            last_updated = manager.get_last_updated(),
            models       = models,
        )
    except Exception as e:
        from fastapi.responses import JSONResponse as JR
        return JR(status_code=500, content={"detail": str(e)})
