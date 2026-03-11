"""
backend/routers/auto_train.py
===============================
POST /auto-train
-----------------
Nhận 1 file CSV thô, chạy DataProcessor (tiền xử lý + kiểm tra alignment),
sau đó:
  1. Fit GlobalScaler trên toàn bộ X (Pha 0 – đảm bảo hệ quy chiếu chung).
  2. Transform X qua GlobalScaler.
  3. Lặp qua từng lớp để train/retrain (Pha 1/2).

Body (multipart/form-data):
    file         : UploadFile  – File CSV thô
    class_column : str         – Tên cột nhãn (mặc định: "state")
    id_columns   : str         – JSON array cột ID
    drop_columns : str         – JSON array cột bỏ qua
    kernel       : str         – (Mặc định 'rbf')
    nu           : float       – (Mặc định 0.05)
    gamma        : str         – (Mặc định 'scale')
    version_name : str         – Tên phiên bản (ví dụ 'v1.0')
    age_threshold: int         – Ngưỡng tuổi SV cho pruning (mặc định 5)
    error_threshold: float     – Ngưỡng accuracy error pruning (mặc định 0.5)
    retrain      : bool        – True = retrain nếu lớp đã tồn tại (mặc định True)
"""

import json
import os
import shutil
import uuid
import numpy as np
import traceback
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from mocsvm.utils.data_processor import DataProcessor
from mocsvm.utils.data_loader import load_and_validate_csv, split_by_class
from mocsvm.utils.global_scaler import GlobalScalerManager
from backend.routers.train import get_mc_manager

router = APIRouter(tags=["auto-train"])

UPLOAD_DIR    = os.getenv("UPLOAD_DIR",    "data/uploads")
PROCESSED_DIR = "data/processed"
MODEL_DIR     = os.getenv("MODEL_DIR",     "models")


def _get_global_scaler() -> GlobalScalerManager:
    """Trả về GlobalScalerManager đang dùng (load từ disk nếu có)."""
    gsm = GlobalScalerManager(model_dir=MODEL_DIR)
    gsm.load()
    return gsm


@router.post("", status_code=status.HTTP_200_OK)
async def auto_upload_and_train(
    file           : UploadFile = File(...,   description="File CSV thô"),
    class_column   : str        = Form("state", description="Tên cột nhãn"),
    id_columns     : str        = Form("[]",    description='JSON array cột ID'),
    drop_columns   : str        = Form("[]",    description='JSON array cột bỏ qua'),
    kernel         : str        = Form("rbf",   description="Kernel SVM"),
    nu             : float      = Form(0.15,    description="Tham số nu"),
    gamma          : str        = Form("1.0",   description="Tham số gamma"),
    version_name   : str        = Form("v1.0",  description="Tên phiên bản"),
    age_threshold  : int        = Form(5,       description="Ngưỡng tuổi SV (chu kỳ)"),
    error_threshold: float      = Form(0.5,     description="Ngưỡng accuracy error pruning"),
    retrain        : bool       = Form(True,    description="Retrain nếu lớp đã tồn tại?"),
):
    """
    **Tự động Upload và Huấn luyện (Auto Upload & Train)**

    Pipeline Phase 0 → 1/2:
    1. Nhận 1 file CSV thô.
    2. DataProcessor: Làm sạch NaN, Label Encode categorical.
    3. Kiểm tra alignment nghiêm ngặt.
    4. **Fit GlobalScaler trên toàn bộ X** (hệ quy chiếu chung duy nhất).
    5. Transform tất cả X qua GlobalScaler.
    6. Chia theo lớp, train/retrain từng lớp dengan SV Pruning.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Chỉ chấp nhận file .csv",
        )

    try:
        id_cols   = json.loads(id_columns)
        drop_cols = json.loads(drop_columns)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='id_columns và drop_columns phải là JSON array, ví dụ: ["ID","name"]',
        )

    session_id   = uuid.uuid4().hex[:12]
    raw_dir      = os.path.join(UPLOAD_DIR, f"raw_{session_id}")
    os.makedirs(raw_dir, exist_ok=True)
    raw_csv_path = os.path.join(raw_dir, file.filename)

    try:
        with open(raw_csv_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        await file.close()

    base_dir   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    output_dir = os.path.join(base_dir, PROCESSED_DIR, session_id)

    # ══════════════════════════════════════════════════════════════
    # PHA 0: Tiền Xử Lý (DataProcessor – KHÔNG scale nội bộ nữa)
    # ══════════════════════════════════════════════════════════════
    try:
        processor = DataProcessor(
            class_column = class_column,
            id_columns   = id_cols,
            drop_columns = drop_cols,
            scale        = False,   # Scale sẽ do GlobalScaler đảm nhận
        )
        report = processor.process(csv_path=raw_csv_path, output_dir=output_dir)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý file: {e}")

    samples_file  = report.saved_paths["samples"]
    features_file = report.saved_paths["features"]
    classes_file  = report.saved_paths["classes"]

    # Load toàn bộ X (chưa scale)
    try:
        X_full, feature_names, class_labels = load_and_validate_csv(
            samples_file, features_file, classes_file
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi đọc file sau xử lý: {e}")

    # ══════════════════════════════════════════════════════════════
    # PHA 0.5: Thiết lập Không gian Tọa độ Toàn cục (Global Scaler)
    # ══════════════════════════════════════════════════════════════
    try:
        gsm = GlobalScalerManager(model_dir=os.path.join(base_dir, MODEL_DIR))
        gsm.fit_and_save(X_full, feature_names=feature_names) # Fit trên TOÀN BỘ X (mọi lớp)
        X_full_scaled = gsm.transform(X_full) # Chuẩn hóa toàn bộ
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khởi tạo GlobalScaler: {e}")

    # ══════════════════════════════════════════════════════════════
    # PHA 1/2: Huấn luyện / Retrain từng lớp
    # ══════════════════════════════════════════════════════════════
    class_data        = split_by_class(X_full_scaled, class_labels)  # Chia theo lớp ĐÃ SCALE
    mc                = get_mc_manager()
    training_results  = []

    for class_key, X_cls_scaled in class_data.items():
        class_name = str(class_key).strip()

        # Negative data: dữ liệu của tất cả lớp khác (đã scale)
        X_neg_list   = [v for k, v in class_data.items() if str(k).strip() != class_name]
        X_neg_scaled = np.vstack(X_neg_list) if X_neg_list else None

        info = {}
        try:
            is_class_exists = (
                class_name in mc.models
                or class_name in mc.manifest.list_classes()
            )

            if is_class_exists and retrain:
                # Lớp đã tồn tại → Retrain với SV Pruning
                try:
                    info = mc.retrain_class(
                        class_name      = class_name,
                        X_new_scaled    = X_cls_scaled,
                        X_neg_scaled    = X_neg_scaled,
                        new_version     = None,
                        age_threshold   = age_threshold,
                        error_threshold = error_threshold,
                    )
                    info["action"] = "retrain"
                except Exception as re_err:
                    # Fallback: train mới nếu retrain thất bại
                    print(f"  [AutoTrain] Retrain lỗi: {re_err}. Fallback → train mới.")
                    info = mc.train_class(
                        class_name   = class_name,
                        X_scaled     = X_cls_scaled,
                        X_neg_scaled = X_neg_scaled,
                        version_name = f"{class_name}-{version_name}",
                        nu=nu, gamma=gamma, kernel=kernel,
                    )
                    info["action"] = "train_fallback"
            else:
                # Lớp mới → Train lần đầu
                info = mc.train_class(
                    class_name   = class_name,
                    X_scaled     = X_cls_scaled,
                    X_neg_scaled = X_neg_scaled,
                    version_name = f"{class_name}-{version_name}",
                    nu=nu, gamma=gamma, kernel=kernel,
                )
                info["action"] = "train"

        except Exception as e:
            info = {
                "class_name"  : class_name,
                "version_name": version_name,
                "n_samples"   : len(X_cls_scaled),
                "action"      : "failed",
                "error"       : str(e),
            }
            traceback.print_exc()

        training_results.append(info)

    return {
        "success"          : True,
        "session_id"       : session_id,
        "message"          : (
            f"Tự động xử lý {report.n_rows:,} dòng, "
            f"fit Global Scaler, "
            f"và huấn luyện {len(class_data)} lớp thành công."
        ),
        "global_scaler"    : gsm.get_info(),
        "alignment"        : {
            "n_rows"         : report.n_rows,
            "n_features"     : report.n_features,
            "n_classes"      : len(report.unique_classes),
            "unique_classes" : report.unique_classes,
            "class_counts"   : report.class_counts,
            "pipeline"       : {
                "encoded_columns" : report.encoded_columns,
                "columns_dropped" : report.columns_dropped,
            },
        },
        "training_results" : training_results,
    }
