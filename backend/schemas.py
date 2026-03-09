"""
Pydantic Schemas
==================
Định nghĩa các schema Request/Response cho FastAPI.
Tách biệt hoàn toàn với ORM models (database.py).
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Upload Schemas
# --------------------------------------------------------------------------

class UploadValidationResult(BaseModel):
    valid         : bool
    n_samples     : int
    n_features    : int
    n_classes     : int
    unique_classes: List[str]
    errors        : List[str]
    saved_paths   : Optional[Dict[str, str]] = None


# --------------------------------------------------------------------------
# Training Schemas
# --------------------------------------------------------------------------

class TrainRequest(BaseModel):
    class_name  : str   = Field(..., description="Tên lớp cần huấn luyện (e.g., 'cat')")
    kernel      : str   = Field(default="rbf",   description="Loại kernel: 'rbf', 'linear', 'poly'")
    nu          : float = Field(default=0.1,  ge=0.0, le=1.0, description="Tham số nu (0-1)")
    gamma       : str   = Field(default="scale", description="Tham số gamma: 'scale', 'auto' hoặc float string")
    version_name: Optional[str] = Field(None,    description="Tên phiên bản (tự tạo nếu None)")
    retrain     : bool  = Field(default=False,   description="True = incremental retrain, False = train mới")
    samples_file: Optional[str] = Field(None,    description="Đường dẫn file samples đã upload")
    features_file: Optional[str] = Field(None,   description="Đường dẫn file features đã upload")
    classes_file : Optional[str] = Field(None,   description="Đường dẫn file classes đã upload")


class TrainResponse(BaseModel):
    success      : bool
    class_name   : str
    version_name : str
    n_samples    : int
    n_features   : int
    inlier_ratio : float
    trained_at   : str
    message      : str


# --------------------------------------------------------------------------
# Models Schemas
# --------------------------------------------------------------------------

class ModelMetadata(BaseModel):
    kernel    : Optional[str] = None
    gamma     : Optional[str] = None
    nu        : Optional[str] = None
    n_samples : Optional[str] = None
    trained_at: Optional[str] = None


class ModelInfo(BaseModel):
    class_name: str
    version   : str
    pkl_path  : Optional[str] = None
    metadata  : Optional[ModelMetadata] = None


class ModelsListResponse(BaseModel):
    total       : int
    last_updated: str
    models      : List[ModelInfo]


class ModelDetailResponse(BaseModel):
    class_name        : str
    version_name      : str
    is_trained        : bool
    kernel            : str
    nu                : float
    gamma             : str
    n_samples         : int
    n_features        : int
    support_vectors   : int = 0
    support_vectors_data : List[List[float]]
    total_rows        : int
    showing_rows      : int


