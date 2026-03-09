"""
Router: Training
=================
Endpoints:
  POST /train         – Kích hoạt workflow huấn luyện / retrain
  GET  /train/history – Lấy lịch sử huấn luyện
  GET  /train/classes – Lấy danh sách lớp từ session upload CSV thô
"""

import os
import numpy as np
from datetime import datetime
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from backend.schemas import TrainRequest, TrainResponse
from mocsvm.core.multiclass import MultiClassOCSVM
from mocsvm.utils.data_loader import load_and_validate_csv, split_by_class

router = APIRouter(prefix="/train", tags=["Training"])

# Đường dẫn model và manifest toàn cục
MODEL_DIR     = os.getenv("MODEL_DIR",     "models")
MANIFEST_PATH = os.getenv("MANIFEST_PATH", "models/global_manifest.xml")

# Singleton manager – được khởi tạo một lần khi import
_mc_manager: MultiClassOCSVM | None = None


def get_mc_manager() -> MultiClassOCSVM:
    """Trả về singleton MultiClassOCSVM, khởi tạo nếu chưa có."""
    global _mc_manager
    if _mc_manager is None:
        _mc_manager = MultiClassOCSVM(model_dir=MODEL_DIR, manifest_path=MANIFEST_PATH)
        # Load tất cả model đã có trong manifest
        _mc_manager.load_all_from_manifest()
    return _mc_manager


@router.post("", response_model=TrainResponse)
def train_model(
    payload: TrainRequest,
):
    """
    Khởi động workflow huấn luyện hoặc retrain cho một lớp.

    Workflow:
    1. Đọc 3 file CSV (samples/features/classes) từ đường dẫn đã upload.
    2. Lọc lấy mẫu thuộc class_name.
    3. Huấn luyện (hoặc retrain) model OC-SVM.
    4. Ghi log vào SQLite.
    """
    mc = get_mc_manager()

    # Xác định đường dẫn file
    samples_file  = payload.samples_file
    features_file = payload.features_file
    classes_file  = payload.classes_file

    if not all([samples_file, features_file, classes_file]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Phải cung cấp samples_file, features_file, classes_file (từ kết quả /upload).",
        )

    # Load và validate dữ liệu
    try:
        X_full, feature_names, class_labels = load_and_validate_csv(
            samples_file, features_file, classes_file  # type: ignore[arg-type]
        )
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Lọc dữ liệu theo class_name
    class_data = split_by_class(X_full, class_labels)
    # Tìm class key tương ứng (chấp nhận lệch whitespace)
    target_class = payload.class_name.strip()
    matched_key = None
    for k in class_data.keys():
        if str(k).strip() == target_class:
            matched_key = k
            break

    if matched_key is None:
        available = [str(k) for k in class_data.keys()]
        raise HTTPException(
            status_code=400,
            detail=f"Lớp '{payload.class_name}' không có trong dữ liệu (so khớp bằng strip). Số lớp có sẵn: {len(available)}",
        )
    X = class_data[matched_key]

    # Thực hiện huấn luyện
    error_msg: str | None = None
    info: dict = {}
    try:
        if payload.retrain:
            try:
                info = mc.retrain_class(
                    class_name  = payload.class_name,
                    X_new       = X,
                    new_version = payload.version_name,
                )
            except ValueError as ve:
                # Lỗi số features không khớp → 400 (lỗi người dùng, không phải server)
                raise HTTPException(status_code=400, detail=str(ve))
            except Exception as e:
                # Nếu là lỗi do chưa train bao giờ, tự động fallback sang train_class thông thường
                if "chưa được train" in str(e).lower() or "không tìm thấy model" in str(e).lower() or "không tồn tại" in str(e).lower():
                    info = mc.train_class(
                        class_name   = payload.class_name,
                        X            = X,
                        version_name = payload.version_name,
                        nu           = payload.nu,
                        gamma        = payload.gamma,
                        kernel       = payload.kernel,
                    )
                else:
                    raise e
        else:
            info = mc.train_class(
                class_name   = payload.class_name,
                X            = X,
                version_name = payload.version_name,
                nu           = payload.nu,
                gamma        = payload.gamma,
                kernel       = payload.kernel,
            )
        log_status = "success"
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
    """
    Trả về danh sách lớp (unique) từ file classes.csv của một session đã xử lý.
    Frontend dùng để tự động điền dropdown lớp cần train.
    """
    processed_dir = os.getenv("PROCESSED_DIR", "data/processed")
    # Đảm bảo đường dẫn tuyệt đối để không bị lỗi 404 khi đổi thư mục
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    classes_path = os.path.join(base_dir, processed_dir, session_id, "classes.csv")

    if not os.path.exists(classes_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy session '{session_id}' tại {classes_path}. Hãy upload CSV thô trước.",
        )

    try:
        import pandas as pd
        df = pd.read_csv(classes_path, header=0)
        class_col = "class" if "class" in df.columns else df.columns[0]
        unique_classes = sorted(df[class_col].astype(str).unique().tolist())
        class_counts = df[class_col].astype(str).value_counts().to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi đọc classes.csv: {e}")

    return {
        "session_id"     : session_id,
        "unique_classes" : unique_classes,
        "class_counts"   : class_counts,
        "n_classes"      : len(unique_classes),
    }






