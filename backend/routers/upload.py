"""
Router: Upload CSV Files
=========================
Endpoint:
  POST /upload – Nhận 3 file CSV (samples, features, classes),
                 validate tính nhất quán, lưu vào data/uploads/.
"""

import os
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from backend.schemas import UploadValidationResult
from mocsvm.utils.data_loader import validate_csv_consistency

router = APIRouter(prefix="/upload", tags=["Upload"])

# Thư mục lưu file upload
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "data/uploads")


def _save_upload(file: UploadFile, dest_dir: str, filename: str) -> str:
    """Lưu UploadFile vào đĩa và trả về đường dẫn."""
    os.makedirs(dest_dir, exist_ok=True)
    filepath = os.path.join(dest_dir, filename)
    content = file.file.read()
    with open(filepath, "wb") as f:
        f.write(content)
    return filepath


@router.post("", response_model=UploadValidationResult)
async def upload_csv(
    samples : UploadFile = File(..., description="File CSV chứa ma trận mẫu dữ liệu (n_samples × n_features)"),
    features: UploadFile = File(..., description="File CSV chứa tên features (1 hàng)"),
    classes : UploadFile = File(..., description="File CSV chứa nhãn lớp (n_samples × 1)"),
):
    """
    Nhận 3 file CSV và validate:
    1. Số dòng của samples == số dòng của classes
    2. Số cột của samples == số features trong features.csv

    File được lưu vào `data/uploads/<session_id>/`.
    """
    # Validate file type (chỉ chấp nhận CSV)
    for upload_file in (samples, features, classes):
        if not upload_file.filename.endswith(".csv"):  # type: ignore[union-attr]
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File '{upload_file.filename}' không phải CSV.",
            )

    # Tạo session folder duy nhất để tránh conflict
    session_id = str(uuid.uuid4())[:8]
    session_dir = os.path.join(UPLOAD_DIR, session_id)

    # Lưu file
    samples_path  = _save_upload(samples,  session_dir, "samples.csv")
    features_path = _save_upload(features, session_dir, "features.csv")
    classes_path  = _save_upload(classes,  session_dir, "classes.csv")

    # Validate tính nhất quán
    report = validate_csv_consistency(samples_path, features_path, classes_path)

    if not report["valid"]:
        # Xoá file nếu không hợp lệ
        import shutil
        shutil.rmtree(session_dir, ignore_errors=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"errors": report["errors"]},
        )

    # Trả kết quả kèm đường dẫn file
    return UploadValidationResult(
        valid          = True,
        n_samples      = report["n_samples"],
        n_features     = report["n_features"],
        n_classes      = report["n_classes"],
        unique_classes = report["unique_classes"],
        errors         = [],
        saved_paths    = {
            "samples" : samples_path,
            "features": features_path,
            "classes" : classes_path,
            "session" : session_id,
        },
    )
