"""
Router: Predict
================
Endpoints:
  POST /predict         – Dự đoán nhãn cho mảng dữ liệu đầu vào (đã scale)
  POST /predict/csv     – Upload test.csv → tự động scale → dự đoán → xuất báo cáo
  GET  /predict/info    – Thông tin các model đang active
  POST /predict/reload  – Reload tất cả model từ manifest
  GET  /predict/sample  – Lấy mẫu ngẫu nhiên từ session (để test nhanh)

Quy trình Phase 3:
  X_test_raw  →  GlobalScaler.transform()  →  predict_multi()  →  ŷ
"""

import io
import os
import random
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sklearn.decomposition import PCA

from mocsvm.core.multiclass import MultiClassOCSVM
from mocsvm.utils.global_scaler import GlobalScalerManager
from mocsvm.utils.encoder_manager import CategoricalEncoderManager

router = APIRouter(prefix="/predict", tags=["Predict"])

MODEL_DIR     = os.getenv("MODEL_DIR",     "models")
MANIFEST_PATH = os.getenv("MANIFEST_PATH", "models/global_manifest.xml")
PROCESSED_DIR = os.getenv("PROCESSED_DIR", "data/processed")

_mc_manager: MultiClassOCSVM | None = None


def get_mc_manager() -> MultiClassOCSVM:
    global _mc_manager
    if _mc_manager is None:
        _mc_manager = MultiClassOCSVM(model_dir=MODEL_DIR, manifest_path=MANIFEST_PATH)
        _mc_manager.load_all_from_manifest()
    return _mc_manager


def _get_model_dir() -> str:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(base_dir, MODEL_DIR)

def _load_global_scaler() -> GlobalScalerManager:
    """Load GlobalScaler đóng băng – dùng cho mọi inference."""
    gsm = GlobalScalerManager(model_dir=_get_model_dir())
    loaded = gsm.load()
    if not loaded:
        raise HTTPException(
            status_code=400,
            detail=(
                "GlobalScaler chưa được khởi tạo. "
                "Hãy chạy POST /auto-train trước để fit GlobalScaler."
            ),
        )
    return gsm


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    data:           List[List[float]] = Field(..., description="Ma trận (n_samples, n_features) – dữ liệu thô CHƯA scale")
    return_scores:  bool  = Field(True,  description="Trả kèm decision scores từng lớp")
    min_margin:     float = Field(0.0,   description="Biên độ tối thiểu giữa lớp nhất và nhì (0 = tắt)")
    return_plot_data: bool = Field(False, description="Trả về data PCA 2D cho Scatter Plot")
    true_labels:    Optional[List[str]] = Field(None, description="Nhãn thật (để visualize)")


class PredictSingleResult(BaseModel):
    predicted_class:   str
    confidence:        float
    margin:            float
    is_low_confidence: bool
    all_scores:        dict


class PredictResponse(BaseModel):
    n_samples:      int
    n_classes:      int
    active_classes: List[str]
    results:        List[PredictSingleResult]
    plot_data:      Optional[dict] = None


# ---------------------------------------------------------------------------
# POST /predict  (dữ liệu thô từ JSON)
# ---------------------------------------------------------------------------

@router.post("", response_model=PredictResponse)
def predict(payload: PredictRequest):
    """
    Dự đoán nhãn cho từng mẫu.

    Dữ liệu đầu vào là dữ liệu CHƯA scale.
    GlobalScaler sẽ tự động transform trước khi đưa vào các model.
    """
    mc = get_mc_manager()
    if not mc.models:
        raise HTTPException(
            status_code=400,
            detail="Chưa có model nào. Hãy train ít nhất một lớp trước.",
        )

    try:
        X_raw = np.array(payload.data, dtype=float)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Dữ liệu không hợp lệ: {e}")

    if X_raw.ndim != 2:
        raise HTTPException(status_code=400, detail="data phải là ma trận 2D.")

    # Transform qua GlobalScaler
    gsm = _load_global_scaler()
    try:
        X_scaled = gsm.transform(X_raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        results_raw = mc.predict_with_confidence(X_scaled, min_margin=payload.min_margin)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi dự đoán: {e}")

    results = [
        PredictSingleResult(
            predicted_class   = r["predicted_class"],
            confidence        = r["confidence"],
            margin            = r["margin"],
            is_low_confidence = r["is_low_confidence"],
            all_scores        = {k: round(float(v), 6) for k, v in r["all_scores"].items()},
        )
        for r in results_raw
    ]

    plot_data = None
    if payload.return_plot_data:
        try:
            all_sv, sv_labels = [], []
            for cname, model in mc.models.items():
                sv = getattr(model, "memory_data", None)
                if sv is not None and len(sv) > 0:
                    all_sv.append(sv)
                    sv_labels.extend([cname] * len(sv))

            if all_sv:
                stacked  = np.vstack(all_sv)
                combined = np.vstack([stacked, X_scaled])
                pca      = PCA(n_components=2)
                c2d      = pca.fit_transform(combined)
                n_sv     = len(stacked)
                sv_2d    = c2d[:n_sv]
                test_2d  = c2d[n_sv:]

                plot_data = {
                    "support_vectors": [
                        {"class": lbl, "x": float(sv_2d[i][0]), "y": float(sv_2d[i][1])}
                        for i, lbl in enumerate(sv_labels)
                    ],
                    "test_points": [
                        {
                            "idx"            : i,
                            "true_class"     : payload.true_labels[i] if payload.true_labels else "Unknown",
                            "predicted_class": results[i].predicted_class,
                            "is_unknown"     : results[i].predicted_class == "unknown",
                            "x"              : float(test_2d[i][0]),
                            "y"              : float(test_2d[i][1]),
                        }
                        for i in range(len(test_2d))
                    ],
                }
        except Exception as e:
            print(f"[PCA Plot] Lỗi tạo plot_data: {e}")
            plot_data = None

    return PredictResponse(
        n_samples      = len(X_raw),
        n_classes      = len(mc.models),
        active_classes = mc.list_classes(),
        results        = results,
        plot_data      = plot_data,
    )


# ---------------------------------------------------------------------------
# POST /predict/csv  (Phase 3 – test.csv → nhãn dự đoán)
# ---------------------------------------------------------------------------

@router.post("/csv")
async def predict_csv(
    file        : UploadFile = File(..., description="File test.csv (chứa features, không cần cột label)"),
    class_column: Optional[str] = Form(None,    description="Tên cột nhãn thật (nếu có, để tính accuracy)"),
    min_margin  : float         = Form(0.0,     description="Biên độ tối thiểu (0 = tắt)"),
    return_csv  : bool          = Form(False,   description="Trả về CSV kết quả thay vì JSON"),
):
    """
    **Phase 3 – Test Workflow**

    1. Đọc file test.csv thô.
    2. Transform qua GlobalScaler đóng băng.
    3. Đưa qua tất cả models song song.
    4. Phân xử bằng Euclidean Nearest-SV Tie-break.
    5. Xuất nhãn dự đoán ŷ kèm báo cáo.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file .csv")

    # Đọc CSV
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content), low_memory=False)
        df.columns = df.columns.str.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi đọc file CSV: {e}")

    # Tách nhãn thật (nếu có)
    true_labels = None
    if class_column and class_column in df.columns:
        true_labels = df[class_column].astype(str).tolist()
        df_features = df.drop(columns=[class_column])
    else:
        df_features = df.copy()

    # Load Label Encoders và áp dụng cho các cột tương ứng (giống hệt quá trình train)
    model_dir = _get_model_dir()
    encoder_mgr = CategoricalEncoderManager(model_dir=model_dir)
    encoders = encoder_mgr.load_encoders()
    if encoders:
        df_features = encoder_mgr.transform_df(df_features, encoders)

    # Lấy Global Scaler
    gsm = _load_global_scaler()

    # Lọc đúng danh sách cột lúc Train (tránh dư cột ID hoặc lệch thứ tự)
    if getattr(gsm, "feature_names", None):
        missing_cols = [c for c in gsm.feature_names if c not in df_features.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"File test.csv thiếu các cột đặc trưng: {missing_cols}")
        df_features = df_features[gsm.feature_names]
    else:
        # Tương thích ngược nếu GlobalScaler form cũ
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise HTTPException(status_code=400, detail="File CSV không có cột số nào sau khi mã hóa.")
        df_features = df_features[numeric_cols]
    
    # Xử lý NaN
    df_features = df_features.dropna()
    try:
        X_raw = df_features.values.astype(float)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi convert dữ liệu: {e}")

    # Transform qua GlobalScaler
    try:
        X_scaled = gsm.transform(X_raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Dự đoán
    mc = get_mc_manager()
    if not mc.models:
        raise HTTPException(status_code=400, detail="Chưa có model nào.")

    try:
        results_raw = mc.predict_with_confidence(X_scaled, min_margin=min_margin)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi dự đoán: {e}")

    # Tính accuracy nếu có nhãn thật
    y_pred = [r["predicted_class"] for r in results_raw]
    accuracy = None
    if true_labels and len(true_labels) == len(y_pred):
        correct  = sum(1 for t, p in zip(true_labels, y_pred) if t == p)
        accuracy = round(correct / len(y_pred), 4)

    # Đếm từng nhãn
    from collections import Counter
    label_dist = dict(Counter(y_pred))
    n_unknown  = label_dist.get("unknown", 0)

    # Xây dựng response
    result_rows = []
    for i, r in enumerate(results_raw):
        row = {
            "predicted_class"   : r["predicted_class"],
            "confidence"        : round(r["confidence"], 4),
            "is_low_confidence" : r["is_low_confidence"],
        }
        if true_labels and i < len(true_labels):
            row["true_class"] = true_labels[i]
            row["correct"]    = (true_labels[i] == r["predicted_class"])
        result_rows.append(row)

    if return_csv:
        # Trả về CSV kết quả
        result_df = df_features.copy()
        result_df["predicted_class"] = y_pred
        result_df["confidence"] = [round(r["confidence"], 4) for r in results_raw]
        if true_labels and len(true_labels) == len(result_df):
            result_df["true_class"] = true_labels
            result_df["correct"]    = result_df["predicted_class"] == result_df["true_class"]
        
        csv_buffer = io.StringIO()
        result_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        return StreamingResponse(
            io.BytesIO(csv_buffer.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=predictions_{file.filename}"},
        )

    return {
        "success"         : True,
        "n_samples"       : len(X_raw),
        "n_classes_active": len(mc.models),
        "active_classes"  : mc.list_classes(),
        "accuracy"        : accuracy,
        "label_distribution": label_dist,
        "n_unknown"       : n_unknown,
        "unknown_ratio"   : round(n_unknown / len(X_raw), 4),
        "results"         : result_rows,
        "scaler_info"     : gsm.get_info(),
    }


# ---------------------------------------------------------------------------
# GET /predict/info
# ---------------------------------------------------------------------------

@router.get("/info")
def get_predict_info():
    """Thông tin các model đang active và trạng thái GlobalScaler."""
    mc = get_mc_manager()

    # GlobalScaler status
    scaler_info = {}
    try:
        gsm = _load_global_scaler()
        scaler_info = gsm.get_info()
    except HTTPException:
        scaler_info = {"is_fitted": False}

    if not mc.models:
        return {
            "active_classes": [],
            "n_features"    : scaler_info.get("n_features", 0),
            "n_classes"     : 0,
            "global_scaler" : scaler_info,
        }

    # n_features từ GlobalScaler hoặc memory_data đầu tiên
    n_features = scaler_info.get("n_features", 0)
    if n_features == 0:
        first_model = next(iter(mc.models.values()))
        md = getattr(first_model, "memory_data", None)
        if md is not None and hasattr(md, "shape"):
            n_features = md.shape[1]

    return {
        "active_classes": mc.list_classes(),
        "n_features"    : n_features,
        "n_classes"     : len(mc.models),
        "global_scaler" : scaler_info,
    }


# ---------------------------------------------------------------------------
# POST /predict/reload
# ---------------------------------------------------------------------------

@router.post("/reload")
def reload_models():
    """Reload tất cả model từ manifest vào memory."""
    global _mc_manager
    try:
        _mc_manager = MultiClassOCSVM(model_dir=MODEL_DIR, manifest_path=MANIFEST_PATH)
        _mc_manager.load_all_from_manifest()
        classes = _mc_manager.list_classes()
        return {
            "success": True,
            "message": f"Đã reload {len(classes)} model(s).",
            "classes": classes,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi reload: {e}")


# ---------------------------------------------------------------------------
# GET /predict/sample
# ---------------------------------------------------------------------------

@router.get("/sample")
def get_random_sample(
    session_id: str           = Query(..., description="session_id từ upload CSV thô"),
    class_name: Optional[str] = Query(None, description="Tên class (bỏ trống = random)"),
    n_samples:  int           = Query(1,    description="Số mẫu", ge=1, le=50),
):
    """Lấy n_samples mẫu ngẫu nhiên từ session đã xử lý."""
    base_dir      = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    session_dir   = os.path.join(base_dir, PROCESSED_DIR, session_id)
    features_path = os.path.join(session_dir, "features.csv")
    classes_path  = os.path.join(session_dir, "classes.csv")

    if not os.path.exists(features_path) or not os.path.exists(classes_path):
        raise HTTPException(
            status_code=404,
            detail=f"Không tìm thấy session '{session_id}'. Hãy upload CSV thô trước.",
        )

    try:
        df_feat    = pd.read_csv(features_path)
        df_classes = pd.read_csv(classes_path)
        class_col  = "class" if "class" in df_classes.columns else df_classes.columns[0]
        labels     = df_classes[class_col].astype(str)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi đọc file: {e}")

    if len(df_feat) != len(labels):
        raise HTTPException(status_code=500, detail="features.csv và classes.csv không đồng bộ.")

    if class_name:
        mask = labels.str.strip() == class_name.strip()
        if not mask.any():
            raise HTTPException(
                status_code=404,
                detail=f"Class '{class_name}' không có. Có sẵn: {sorted(labels.unique().tolist())}",
            )
        df_feat_filtered = df_feat[mask].reset_index(drop=True)
        labels_filtered  = labels[mask].reset_index(drop=True)
    else:
        df_feat_filtered = df_feat
        labels_filtered  = labels

    total   = len(df_feat_filtered)
    indices = random.sample(range(total), min(n_samples, total))

    samples_out = []
    for idx in indices:
        row = df_feat_filtered.iloc[idx]
        numeric_values = []
        for val in row.values:
            try:
                numeric_values.append(float(val))
            except (ValueError, TypeError):
                numeric_values.append(0.0)

        samples_out.append({
            "index"        : int(idx),
            "true_class"   : str(labels_filtered.iloc[idx]),
            "feature_names": list(df_feat_filtered.columns),
            "values"       : numeric_values,
        })

    return {
        "session_id"     : session_id,
        "class_filter"   : class_name,
        "n_returned"     : len(samples_out),
        "total_in_class" : total,
        "samples"        : samples_out,
    }
