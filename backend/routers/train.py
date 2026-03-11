"""
Router: Training
=================
Endpoints:
  POST /train         – Kích hoạt workflow huấn luyện / retrain cho 1 lớp
  GET  /train/classes – Lấy danh sách lớp từ session upload CSV thô

Quy trình:
  1. Đọc 3 file CSV (samples/features/classes) từ đường dẫn đã upload.
  2. Load GlobalScaler đã đóng băng (TUYỆT ĐỐI KHÔNG fit lại).
  3. Transform X qua GlobalScaler.
  4. Lọc lấy mẫu thuộc class_name, huấn luyện / retrain model OC-SVM.
"""

import os
import numpy as np
from datetime import datetime
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from backend.schemas import TrainRequest, TrainResponse
from mocsvm.core.multiclass import MultiClassOCSVM
from mocsvm.utils.data_loader import load_and_validate_csv, split_by_class
from mocsvm.utils.global_scaler import GlobalScalerManager

router = APIRouter(prefix="/train", tags=["Training"])

MODEL_DIR     = os.getenv("MODEL_DIR",     "models")
MANIFEST_PATH = os.getenv("MANIFEST_PATH", "models/global_manifest.xml")

# Singleton manager – khởi tạo một lần khi import
_mc_manager: MultiClassOCSVM | None = None


def get_mc_manager() -> MultiClassOCSVM:
    """Trả về singleton MultiClassOCSVM, khởi tạo nếu chưa có."""
    global _mc_manager
    if _mc_manager is None:
        _mc_manager = MultiClassOCSVM(model_dir=MODEL_DIR, manifest_path=MANIFEST_PATH)
        _mc_manager.load_all_from_manifest()
    return _mc_manager


def _load_global_scaler() -> GlobalScalerManager:
    """Load GlobalScaler đã đóng băng từ disk."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    gsm = GlobalScalerManager(model_dir=os.path.join(base_dir, MODEL_DIR))
    loaded = gsm.load()
    if not loaded:
        raise HTTPException(
            status_code=400,
            detail=(
                "GlobalScaler chưa được khởi tạo. "
                "Hãy chạy /auto-train trước để fit GlobalScaler trên dữ liệu gốc."
            ),
        )
    return gsm


@router.post("", response_model=TrainResponse)
def train_model(payload: TrainRequest):
    """
    Khởi động workflow huấn luyện hoặc retrain cho một lớp.

    Sử dụng GlobalScaler đã đóng băng để chuẩn hóa dữ liệu.
    TUYỆT ĐỐI KHÔNG fit lại scaler – đảm bảo hệ quy chiếu nhất quán.
    """
    mc = get_mc_manager()

    samples_file  = payload.samples_file
    features_file = payload.features_file
    classes_file  = payload.classes_file

    if not all([samples_file, features_file, classes_file]):
        raise HTTPException(
            status_code=400,
            detail="Phải cung cấp samples_file, features_file, classes_file.",
        )

    # Load dữ liệu thô
    try:
        X_full, feature_names, class_labels = load_and_validate_csv(
            samples_file, features_file, classes_file
        )
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Load GlobalScaler đóng băng → transform (KHÔNG fit lại)
    gsm = _load_global_scaler()
    try:
        X_full_scaled = gsm.transform(X_full)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Tách theo lớp (đã scale)
    class_data = split_by_class(X_full_scaled, class_labels)
    target_class = payload.class_name.strip()
    matched_key  = next(
        (k for k in class_data if str(k).strip() == target_class), None
    )
    if matched_key is None:
        available = [str(k) for k in class_data.keys()]
        raise HTTPException(
            status_code=400,
            detail=f"Lớp '{payload.class_name}' không có trong dữ liệu. Có sẵn: {available}",
        )

    X_scaled     = class_data[matched_key]
    X_neg_list   = [v for k, v in class_data.items() if str(k).strip() != target_class]
    X_neg_scaled = np.vstack(X_neg_list) if X_neg_list else None

    # Thực hiện train / retrain
    info: dict = {}
    try:
        if payload.retrain:
            try:
                info = mc.retrain_class(
                    class_name   = payload.class_name,
                    X_new_scaled = X_scaled,
                    X_neg_scaled = X_neg_scaled,
                    new_version  = payload.version_name,
                )
            except ValueError as ve:
                raise HTTPException(status_code=400, detail=str(ve))
            except Exception as e:
                if any(kw in str(e).lower() for kw in ["chưa được train", "không tìm thấy", "không tồn tại"]):
                    info = mc.train_class(
                        class_name   = payload.class_name,
                        X_scaled     = X_scaled,
                        X_neg_scaled = X_neg_scaled,
                        version_name = payload.version_name,
                        nu=payload.nu, gamma=payload.gamma, kernel=payload.kernel,
                    )
                else:
                    raise e
        else:
            info = mc.train_class(
                class_name   = payload.class_name,
                X_scaled     = X_scaled,
                X_neg_scaled = X_neg_scaled,
                version_name = payload.version_name,
                nu=payload.nu, gamma=payload.gamma, kernel=payload.kernel,
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi huấn luyện: {e}")

    return TrainResponse(
        success      = True,
        class_name   = info["class_name"],
        version_name = info["version_name"],
        n_samples    = info["n_samples"],
        n_features   = info["n_features"],
        inlier_ratio = info["inlier_ratio"],
        trained_at   = info["trained_at"],
        message      = f"✓ {'Retrain' if payload.retrain else 'Train'} thành công cho lớp '{payload.class_name}'",
    )


@router.get("/classes", tags=["Training"])
def get_classes_from_session(
    session_id: str = Query(..., description="session_id trả về sau khi upload CSV thô"),
):
    """Trả về danh sách lớp từ file classes.csv của một session."""
    processed_dir = os.getenv("PROCESSED_DIR", "data/processed")
    base_dir      = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    classes_path  = os.path.join(base_dir, processed_dir, session_id, "classes.csv")

    if not os.path.exists(classes_path):
        raise HTTPException(
            status_code=404,
            detail=f"Không tìm thấy session '{session_id}'. Hãy upload CSV thô trước.",
        )

    try:
        import pandas as pd
        df           = pd.read_csv(classes_path, header=0)
        class_col    = "class" if "class" in df.columns else df.columns[0]
        unique_classes = sorted(df[class_col].astype(str).unique().tolist())
        class_counts   = df[class_col].astype(str).value_counts().to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi đọc classes.csv: {e}")

    return {
        "session_id"    : session_id,
        "unique_classes": unique_classes,
        "class_counts"  : class_counts,
        "n_classes"     : len(unique_classes),
    }
