"""
backend/routers/auto_train.py
===============================
POST /auto-train
-----------------
Nhận 1 file CSV thô, chạy DataProcessor (tiền xử lý + kiểm tra alignment
nghiêm ngặt), sau đó lặp qua từng lớp được tìm thấy để huấn luyện/retrain.

Body (multipart/form-data):
    file         : UploadFile  – File CSV thô
    class_column : str         – Tên cột nhãn (mặc định: "state")
    id_columns   : str         – JSON array cột ID, ví dụ: '["ID","name"]'
    drop_columns : str         – JSON array cột bỏ qua
    scale        : bool        – Có áp dụng StandardScaler? (mặc định: true)
    
    # Training params
    kernel       : str         - (Mặc định 'rbf')
    nu           : float       - (Mặc định 0.1)
    gamma        : str         - (Mặc định 'scale')
    version_name : str         - Phiên bản (ví dụ 'v1.0.0')
"""

import json
import os
import shutil
import uuid
import traceback
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from mocsvm.utils.data_processor import DataProcessor
from mocsvm.utils.data_loader import load_and_validate_csv, split_by_class
from backend.routers.train import get_mc_manager

router = APIRouter(tags=["auto-train"])

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "data/uploads")
PROCESSED_DIR = "data/processed"

@router.post("", status_code=status.HTTP_200_OK)
async def auto_upload_and_train(
    file        : UploadFile = File(..., description="File CSV thô"),
    class_column: str        = Form("state", description="Tên cột nhãn mục tiêu"),
    id_columns  : str        = Form("[]",   description='JSON array tên cột ID, ví dụ: ["ID","name"]'),
    drop_columns: str        = Form("[]",   description='JSON array tên cột bỏ qua'),
    scale       : bool       = Form(True,  description="Áp dụng StandardScaler"),
    
    kernel      : str        = Form("rbf", description="Kernel cho SVM"),
    nu          : float      = Form(0.1, description="Tham số nu cho SVM"),
    gamma       : str        = Form("scale", description="Tham số gamma cho SVM"),
    version_name: str        = Form("v1.0", description="Tên phiên bản huấn luyện"),
):
    """
    **Tự động Upload và Huấn luyện (Auto Upload & Train) **
    
    Pipeline:
    1. Nhận 1 file CSV thô
    2. DataProcessor: Làm sạch NaN, Label Encode categorical, StandardScaler số
    3. Kiểm tra alignment nghiêm ngặt
    4. Lưu 3 file: `samples.csv`, `features.csv`, `classes.csv`
    5. Đọc lại 3 file đã lưu và chia dữ liệu theo class
    6. Lặp qua các class: tự động retrain (nếu lớp đã tồn tại) hoặc train (nếu lớp mới).
    7. Ghi log và trả về báo cáo tổng hợp.
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
            detail="id_columns và drop_columns phải là JSON array hợp lệ, ví dụ: [\"ID\",\"name\"]",
        )

    session_id  = uuid.uuid4().hex[:12]
    raw_dir     = os.path.join(UPLOAD_DIR, f"raw_{session_id}")
    os.makedirs(raw_dir, exist_ok=True)
    raw_csv_path = os.path.join(raw_dir, file.filename)

    try:
        with open(raw_csv_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        await file.close()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    output_dir = os.path.join(base_dir, PROCESSED_DIR, session_id)

    # ==========================================
    # PHASE 0: Data Processing
    # ==========================================
    try:
        processor = DataProcessor(
            class_column = class_column,
            id_columns   = id_cols,
            drop_columns = drop_cols,
            scale        = scale,
        )
        report = processor.process(csv_path=raw_csv_path, output_dir=output_dir)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Lỗi xử lý file: {e}")

    samples_file  = report.saved_paths["samples"]
    features_file = report.saved_paths["features"]
    classes_file  = report.saved_paths["classes"]

    # ==========================================
    # PHASE 1: Auto Training
    # ==========================================
    try:
        X_full, feature_names, class_labels = load_and_validate_csv(
            samples_file, features_file, classes_file
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi đọc file sau khi xử lý: {str(e)}")

    class_data = split_by_class(X_full, class_labels)
    mc = get_mc_manager()
    
    training_results = []
    
    for class_key, X in class_data.items():
        class_name = str(class_key).strip()
        
        info = {}
        try:
            # Kiểm tra xem class đã có trong danh sách model hoặc manifest hay chưa
            is_class_exists = class_name in mc.models or class_name in mc.manifest.list_classes()

            try:
                if is_class_exists:
                    # Đã có -> retrain (Tự động tăng phiên bản, bỏ qua version_name từ UI)
                    info = mc.retrain_class(
                        class_name  = class_name,
                        X_new       = X,
                        new_version = None, 
                    )
                    info["action"] = "retrain"
                else:
                    # Chưa có -> train mới
                    # Ghép class_name vào version_name để tránh ghi đè file của nhau (vd: USD-v1.0)
                    info = mc.train_class(
                        class_name   = class_name,
                        X            = X,
                        version_name = f"{class_name}-{version_name}",
                        nu           = nu,
                        gamma        = gamma,
                        kernel       = kernel,
                    )
                    info["action"] = "train"
            except ValueError as ve:
                raise ve
            except Exception as e:
                # Fallback: Nếu retrain lỗi vì lý do nào đó (ví dụ file mất), thử train lại từ đầu
                if is_class_exists and ("chưa được train" in str(e).lower() or "không tìm thấy" in str(e).lower() or "không tồn tại" in str(e).lower()):
                    info = mc.train_class(
                        class_name   = class_name,
                        X            = X,
                        version_name = f"{class_name}-{version_name}",
                        nu           = nu,
                        gamma        = gamma,
                        kernel       = kernel,
                    )
                    info["action"] = "train"
                else:
                    raise e
        except Exception as e:
            info = {
                "class_name": class_name,
                "version_name": version_name,
                "n_samples": len(X),
                "action": "failed",
                "error": str(e)
            }
        
        training_results.append(info)

    return {
        "success"   : True,
        "session_id": session_id,
        "message"   : f"Tự động xử lý {report.n_rows:,} dòng và huấn luyện {len(class_data)} lớp thành công",
        "alignment" : {
            "samples_rows" : report.n_rows,
            "features_rows": report.n_rows,
            "classes_rows" : report.n_rows,
            "summary"      : {
                "n_rows"         : report.n_rows,
                "n_features"     : report.n_features,
                "n_classes"      : len(report.unique_classes),
                "unique_classes" : report.unique_classes,
                "class_counts"   : report.class_counts,
            },
            "pipeline": {
                "encoded_columns" : report.encoded_columns,
                "scaled_columns"  : report.scaled_columns,
                "columns_dropped" : report.columns_dropped,
            }
        },
        "training_results": training_results,
    }
