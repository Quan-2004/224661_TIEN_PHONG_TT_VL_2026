"""
Router: Models List
====================
Endpoints:
  GET /models              – Đọc global_manifest.xml, trả về danh sách tất cả model.
  GET /models/{class_name} – Đọc file .pkl, trả về metadata + training data.
"""

import os
import joblib
import numpy as np

from fastapi import APIRouter, HTTPException, Query

from backend.schemas import ModelsListResponse, ModelInfo, ModelMetadata, ModelDetailResponse
from mocsvm.core.manifest_manager import ManifestManager

router = APIRouter(prefix="/models", tags=["Models"])

MANIFEST_PATH = os.getenv("MANIFEST_PATH", "models/global_manifest.xml")
MODEL_DIR     = os.getenv("MODEL_DIR",     "models")


@router.get("", response_model=ModelsListResponse)
def list_models():
    """
    Lấy danh sách tất cả model từ global_manifest.xml.
    Không yêu cầu xác thực – endpoint public để Dashboard đọc.
    """
    if not os.path.exists(MANIFEST_PATH):
        # Manifest chưa có → trả về danh sách rỗng
        return ModelsListResponse(total=0, last_updated="N/A", models=[])

    try:
        manager = ManifestManager(MANIFEST_PATH)
        all_info = manager.list_all()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi đọc manifest: {e}")

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


@router.get("/{class_name}", response_model=ModelDetailResponse)
def get_model_detail(
    class_name: str,
    limit: int = Query(default=100, ge=1, le=1000, description="Số dòng training data tối đa trả về"),
):
    """
    Đọc file .pkl của model và trả về metadata + Support Vectors data.
    """
    # Tìm pkl_path từ manifest
    if not os.path.exists(MANIFEST_PATH):
        raise HTTPException(status_code=404, detail="Manifest chưa tồn tại.")

    try:
        manager = ManifestManager(MANIFEST_PATH)
        all_info = manager.list_all()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi đọc manifest: {e}")

    # Tìm model tương ứng
    pkl_path = None
    for item in all_info:
        if item.get("class_name", "").strip() == class_name.strip():
            pkl_path = item.get("pkl_path")
            break

    if pkl_path is None:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy model cho lớp '{class_name}'.")

    # Load file pkl
    full_path = os.path.abspath(pkl_path)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail=f"File pkl không tồn tại: {pkl_path}")

    try:
        data = joblib.load(full_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi đọc file pkl: {e}")

    # Trích xuất memory_data (Support Vectors)
    if "memory_data" in data:
        training_data_raw = data.get("memory_data", [])
    elif "support_vectors_cache" in data and data["support_vectors_cache"] is not None:
        training_data_raw = data["support_vectors_cache"]
    else:
        training_data_raw = data.get("training_data", [])
        
    total_rows = len(training_data_raw) if training_data_raw is not None else 0
    limited_data = training_data_raw[:limit] if training_data_raw is not None else []

    # Chuyển sang list[list[float]] – hỗ trợ cả numpy array và list thuần
    training_data_list = []
    for row in limited_data:
        if isinstance(row, np.ndarray):
            training_data_list.append([round(float(x), 6) for x in row])
        elif isinstance(row, (list, tuple)):
            training_data_list.append([round(float(x), 6) for x in row])
        else:
            training_data_list.append([float(row)])

    # Đếm support vectors nếu có model sklearn
    sv_count = 0
    sklearn_model = data.get("model")
    if sklearn_model is not None and hasattr(sklearn_model, "support_vectors_"):
        sv_count = len(sklearn_model.support_vectors_)

    return ModelDetailResponse(
        class_name      = data.get("class_name", class_name),
        version_name    = data.get("version_name", "unknown"),
        is_trained      = data.get("is_trained", False),
        kernel          = data.get("kernel", "rbf"),
        nu              = float(data.get("nu", 0.1)),
        gamma           = str(data.get("gamma", "scale")),
        n_samples       = total_rows,
        n_features      = len(training_data_list[0]) if training_data_list else 0,
        support_vectors = sv_count,
        support_vectors_data = training_data_list,
        total_rows      = total_rows,
        showing_rows    = len(training_data_list),
    )
