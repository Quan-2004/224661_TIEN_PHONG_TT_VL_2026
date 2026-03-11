"""
Router: Test CSV
================
Endpoint:
  POST /test-csv/upload – Nhận file CSV, chạy test bằng MultiClassOCSVM (OvR)
                          và trả về kết quả dự đoán, thống kê số lượng.
"""

import os
import uuid
from typing import Dict, Any

import pandas as pd
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sklearn.preprocessing import StandardScaler

from mocsvm.core.multiclass import MultiClassOCSVM

router = APIRouter(prefix="/test-csv", tags=["Test CSV"])

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "data/uploads")
MODEL_DIR = os.getenv("MODEL_DIR", "models")
MANIFEST_PATH = os.getenv("MANIFEST_PATH", "models/global_manifest.xml")

def _save_upload(file: UploadFile, dest_dir: str, filename: str) -> str:
    """Lưu UploadFile vào đĩa và trả về đường dẫn."""
    os.makedirs(dest_dir, exist_ok=True)
    filepath = os.path.join(dest_dir, filename)
    content = file.file.read()
    with open(filepath, "wb") as f:
        f.write(content)
    return filepath

@router.post("/upload")
async def upload_and_test_csv(
    csv_file: UploadFile = File(..., description="File CSV chứa dữ liệu cần kiểm thử")
) -> Dict[str, Any]:
    """
    Nhận file CSV, thực hiện test OvR bằng các model đã lưu, và trả về report JSON.
    Cũng tạo ra file CSV chi tiết để tải xuống sau đó.
    """
    if not csv_file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File '{csv_file.filename}' không phải CSV.",
        )

    # 1. Tạo session folder duy nhất để lưu file CSV tạm / output
    session_id = str(uuid.uuid4())[:8]
    session_dir = os.path.join(UPLOAD_DIR, "test_csv", session_id)
    os.makedirs(session_dir, exist_ok=True)

    # Lưu file input
    input_filepath = _save_upload(csv_file, session_dir, "input.csv")

    try:
        df_test = pd.read_csv(input_filepath)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Lỗi đọc file CSV: {e}"
        )

    # 2. Khởi tạo và nạp tất cả model đã train từ Manifest
    manager = MultiClassOCSVM(model_dir=MODEL_DIR, manifest_path=MANIFEST_PATH)
    try:
        manager.load_all_from_manifest()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi load model: {e}"
        )

    active_classes = manager.list_classes()
    if not active_classes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Không có model nào được nạp. Vui lòng huấn luyện model trước!"
        )

    # 3. Tiền xử lý
    df_features = df_test.select_dtypes(include=[np.number])
    if df_features.empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File CSV không chứa cột dữ liệu số nào để phân tích."
        )
    
    X_test_raw = df_features.values
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test_raw)

    # 4. Thực hiện dự đoán hàng loạt (OvR)
    predictions, score_matrix = manager.predict_multi(X_test_scaled, return_scores=True)

    # Thống kê
    results_summary = pd.Series(predictions).value_counts().to_dict()

    # 5. Lưu file kết quả chi tiết
    output_filename = "test_results_detailed.csv"
    output_filepath = os.path.join(session_dir, output_filename)
    
    df_test['Dự_đoán'] = predictions
    for i, class_name in enumerate(active_classes):
        df_test[f'Score_{class_name}'] = score_matrix[:, i]
        
    df_test.to_csv(output_filepath, index=False)

    return {
        "success": True,
        "session_id": session_id,
        "total_samples": len(X_test_scaled),
        "results_summary": results_summary,
        "download_url": f"/test-csv/download/{session_id}"
    }

@router.get("/download/{session_id}")
async def download_test_result(session_id: str):
    """
    Tải về file test_results_detailed.csv từ một phiên test.
    """
    file_path = os.path.join(UPLOAD_DIR, "test_csv", session_id, "test_results_detailed.csv")
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy file kết quả cho phiên test này."
        )
    return FileResponse(path=file_path, filename="test_results_detailed.csv", media_type="text/csv")
