"""
backend/routers/upload_raw.py
===============================
POST /upload-raw
-----------------
Nhận 1 file CSV thô, chạy DataProcessor (tiền xử lý + kiểm tra alignment
nghiêm ngặt), lưu 3 file đồng bộ vào data/processed/<session_id>/.

Body (multipart/form-data):
    file         : UploadFile  – File CSV thô
    class_column : str         – Tên cột nhãn (mặc định: "state")
    id_columns   : str         – JSON array cột ID, ví dụ: '["ID","name"]'
    drop_columns : str         – JSON array cột bỏ qua
    scale        : bool        – Có áp dụng StandardScaler? (mặc định: true)
"""

import json
import os
import shutil
import uuid
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from mocsvm.utils.data_processor import DataProcessor

router = APIRouter(tags=["upload-raw"])

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "data/uploads")
PROCESSED_DIR = "data/processed"


@router.post("/preview", status_code=status.HTTP_200_OK)
async def preview_csv_columns(
    file: UploadFile = File(..., description="File CSV thô cần xem trước cột"),
):
    """
    **Preview CSV** – Đọc nhanh file CSV và trả về danh sách cột + mẫu dữ liệu.
    Dùng để auto-detect cột trước khi cấu hình xử lý.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file .csv")

    # Lưu tạm để đọc
    tmp_path = os.path.join(UPLOAD_DIR, f"preview_{uuid.uuid4().hex[:8]}.csv")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        await file.close()

    try:
        import pandas as pd
        try:
            df = pd.read_csv(tmp_path, encoding="utf-8", low_memory=False, nrows=5)
        except UnicodeDecodeError:
            df = pd.read_csv(tmp_path, encoding="latin-1", low_memory=False, nrows=5)

        # Strip tên cột
        df.columns = df.columns.str.strip()

        # Phân loại cột
        columns_info = []
        for col in df.columns:
            is_unnamed = str(col).startswith("Unnamed")
            dtype = str(df[col].dtype)
            sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            columns_info.append({
                "name"      : col,
                "dtype"     : dtype,
                "sample"    : str(sample) if sample is not None else "",
                "unnamed"   : is_unnamed,
                "all_nan"   : bool(df[col].isna().all()),
            })

        return {
            "filename"      : file.filename,
            "n_columns"     : len(df.columns),
            "columns"       : columns_info,
            "suggested": {
                "class_column"  : "",   # user tự chọn
                "id_columns"    : [c["name"] for c in columns_info if c["dtype"] in ("object", "int64") and not c["unnamed"] and c["name"] not in ("state",)][:2],
                "drop_columns"  : [c["name"] for c in columns_info if c["unnamed"] or c["all_nan"]],
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi đọc file: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)




@router.post("", status_code=status.HTTP_200_OK)
async def upload_raw_csv(
    file        : UploadFile = File(..., description="File CSV thô"),
    class_column: str        = Form("state", description="Tên cột nhãn mục tiêu"),
    id_columns  : str        = Form("[]",   description='JSON array tên cột ID, ví dụ: ["ID","name"]'),
    drop_columns: str        = Form("[]",   description='JSON array tên cột bỏ qua'),
    scale       : bool       = Form(True,  description="Áp dụng StandardScaler"),
):
    """
    **Phase 0 – Upload CSV thô & Tiền xử lý**

    Pipeline:
    1. Nhận 1 file CSV thô
    2. Làm sạch NaN, Label Encode categorical, StandardScaler số
    3. **Kiểm tra alignment nghiêm ngặt** (samples == features == classes rows)
    4. Lưu 3 file: `samples.csv`, `features.csv`, `classes.csv`
    5. Trả về báo cáo đầy đủ

    Trả về message: **"Đã tách thành công X dòng dữ liệu đồng bộ"**
    """
    # Validate file type
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Chỉ chấp nhận file .csv",
        )

    # Parse JSON arrays từ form
    try:
        id_cols   = json.loads(id_columns)
        drop_cols = json.loads(drop_columns)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="id_columns và drop_columns phải là JSON array hợp lệ, ví dụ: [\"ID\",\"name\"]",
        )

    # Lưu file tạm
    session_id  = uuid.uuid4().hex[:12]
    raw_dir     = os.path.join(UPLOAD_DIR, f"raw_{session_id}")
    os.makedirs(raw_dir, exist_ok=True)
    raw_csv_path = os.path.join(raw_dir, file.filename)

    try:
        with open(raw_csv_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        await file.close()

    # Thư mục output cho 3 file đã xử lý
    # Dùng đường dẫn tuyệt đối quay về gốc dự án mOC-isvm2/ để đồng bộ với train.py
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    output_dir = os.path.join(base_dir, PROCESSED_DIR, session_id)

    # Chạy DataProcessor
    try:
        processor = DataProcessor(
            class_column = class_column,
            id_columns   = id_cols,
            drop_columns = drop_cols,
            scale        = scale,
        )
        report = processor.process(csv_path=raw_csv_path, output_dir=output_dir)

    except ValueError as e:
        # Alignment check thất bại hoặc lỗi cấu hình
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi xử lý file: {e}",
        )

    return {
        "success"   : True,
        "session_id": session_id,
        "message"   : f"Đã tách thành công {report.n_rows:,} dòng dữ liệu đồng bộ",
        "summary"   : {
            "n_rows"         : report.n_rows,
            "n_features"     : report.n_features,
            "n_classes"      : len(report.unique_classes),
            "unique_classes" : report.unique_classes,
            "class_counts"   : report.class_counts,
        },
        "pipeline": {
            "columns_used"    : report.columns_used,
            "columns_dropped" : report.columns_dropped,
            "encoded_columns" : report.encoded_columns,
            "scaled_columns"  : report.scaled_columns,
            "scaling_applied" : scale,
        },
        "alignment": {
            "verified": True,
            "samples_rows" : report.n_rows,
            "features_rows": report.n_rows,
            "classes_rows" : report.n_rows,
        },
        "saved_paths": report.saved_paths,
    }
