"""
Router: Predict
================
Endpoints:
  POST /predict         – Dự đoán nhãn cho mảng dữ liệu đầu vào
  POST /predict/reload  – Reload tất cả model từ manifest vào memory
  GET  /predict/sample  – Lấy 1 mẫu ngẫu nhiên từ session đã upload (để test nhanh)
"""

import os
import random
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from sklearn.decomposition import PCA

from mocsvm.core.multiclass import MultiClassOCSVM

router = APIRouter(prefix="/predict", tags=["Predict"])

# Đường dẫn model và manifest toàn cục
MODEL_DIR     = os.getenv("MODEL_DIR",     "models")
MANIFEST_PATH = os.getenv("MANIFEST_PATH", "models/global_manifest.xml")
PROCESSED_DIR = os.getenv("PROCESSED_DIR", "data/processed")

# Singleton manager (dùng chung với train.py thông qua import)
_mc_manager: MultiClassOCSVM | None = None


def get_mc_manager() -> MultiClassOCSVM:
    """Trả về singleton MultiClassOCSVM, khởi tạo nếu chưa có."""
    global _mc_manager
    if _mc_manager is None:
        _mc_manager = MultiClassOCSVM(model_dir=MODEL_DIR, manifest_path=MANIFEST_PATH)
        _mc_manager.load_all_from_manifest()
    return _mc_manager


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    data: List[List[float]] = Field(..., description="Ma trận dữ liệu, shape (n_samples, n_features)")
    return_scores: bool = Field(True,  description="Có trả kèm điểm từng class không")
    min_margin: float    = Field(0.0,  description="Biên độ tối thiểu giữa lớp nhất và nhì (0 = tắt)")
    return_plot_data: bool = Field(False, description="Yêu cầu trả về data PCA 2D cho Scatter Plot")
    true_labels: Optional[List[str]] = Field(None, description="Nhãn thật (nếu có, dùng để visualize)")


class ClassScore(BaseModel):
    class_name: str
    score: float


class PredictSingleResult(BaseModel):
    predicted_class:    str
    confidence:         float
    margin:             float
    is_low_confidence:  bool
    all_scores:         dict


class PredictResponse(BaseModel):
    n_samples:      int
    n_classes:      int
    active_classes: List[str]
    results:        List[PredictSingleResult]
    plot_data:      Optional[dict] = None


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------

@router.post("", response_model=PredictResponse)
def predict(payload: PredictRequest):
    """
    Dự đoán nhãn cho từng mẫu trong `data`.

    Trả kèm điểm từng class và trạng thái tin cậy.
    """
    mc = get_mc_manager()

    if not mc.models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Chưa có model nào. Hãy train ít nhất một lớp trước.",
        )

    try:
        X = np.array(payload.data, dtype=float)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Dữ liệu đầu vào không hợp lệ: {e}")

    if X.ndim != 2:
        raise HTTPException(status_code=400, detail="data phải là ma trận 2D (list of lists).")

    try:
        results_raw = mc.predict_with_confidence(X, min_margin=payload.min_margin)
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
            # Thu thập dữ liệu để chạy PCA
            all_sv_data = []
            sv_labels = []
            for class_name, model in mc.models.items():
                if hasattr(model, "memory_data") and model.memory_data is not None:
                    sv = model.memory_data
                    if len(sv) > 0:
                        all_sv_data.append(sv)
                        sv_labels.extend([class_name] * len(sv))
            
            # Nếu có dữ liệu SV, gộp với Test Data
            if all_sv_data:
                stacked_sv = np.vstack(all_sv_data)
                combined = np.vstack([stacked_sv, X])
                
                pca = PCA(n_components=2)
                combined_2d = pca.fit_transform(combined)
                
                # Cắt kết quả 2D ra lại SV và Test
                n_sv_total = len(stacked_sv)
                sv_2d = combined_2d[:n_sv_total]
                test_2d = combined_2d[n_sv_total:]
                
                support_vectors_plot = [
                    {"class": lbl, "x": float(sv_2d[i][0]), "y": float(sv_2d[i][1])}
                    for i, lbl in enumerate(sv_labels)
                ]
                
                test_points_plot = [
                    {
                        "idx": i,
                        "true_class": payload.true_labels[i] if payload.true_labels else "Unknown",
                        "predicted_class": results[i].predicted_class,
                        "is_unknown": results[i].predicted_class == "unknown",
                        "x": float(test_2d[i][0]),
                        "y": float(test_2d[i][1])
                    }
                    for i in range(len(test_2d))
                ]
                
                plot_data = {
                    "support_vectors": support_vectors_plot,
                    "test_points": test_points_plot
                }
        except Exception as e:
            print(f"[PCA Plot] Error generating plot data: {e}")
            plot_data = None

    return PredictResponse(
        n_samples      = len(X),
        n_classes      = len(mc.models),
        active_classes = mc.list_classes(),
        results        = results,
        plot_data      = plot_data
    )


# ---------------------------------------------------------------------------
# POST /predict/reload
# ---------------------------------------------------------------------------

@router.get("/info")
def get_predict_info():
    """
    Trả về thông tin các model đang hoạt động trong memory:
    danh sách class và n_features. Frontend dùng để sinh vector random đúng chiều.
    """
    mc = get_mc_manager()
    if not mc.models:
        return {"active_classes": [], "n_features": 0, "n_classes": 0}

    # Lấy n_features từ model đầu tiên
    first_model = next(iter(mc.models.values()))
    n_features = 0
    try:
        if hasattr(first_model, "_model") and first_model._model is not None:
            sv = getattr(first_model._model, "support_vectors_", None)
            if sv is not None:
                n_features = sv.shape[1]
        if n_features == 0 and hasattr(first_model, "training_data"):
            td = first_model.training_data
            if td is not None and hasattr(td, "shape"):
                n_features = td.shape[1]
    except Exception:
        pass

    return {
        "active_classes": mc.list_classes(),
        "n_features"    : n_features,
        "n_classes"     : len(mc.models),
    }


@router.post("/reload")
def reload_models():
    """Reload tất cả model từ manifest vào memory (hữu ích sau khi train xong)."""
    global _mc_manager
    try:
        _mc_manager = MultiClassOCSVM(model_dir=MODEL_DIR, manifest_path=MANIFEST_PATH)
        _mc_manager.load_all_from_manifest()
        classes = _mc_manager.list_classes()
        return {
            "success" : True,
            "message" : f"Đã reload {len(classes)} model(s).",
            "classes" : classes,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi reload: {e}")


# ---------------------------------------------------------------------------
# GET /predict/sample
# ---------------------------------------------------------------------------

@router.get("/sample")
def get_random_sample(
    session_id: str            = Query(..., description="session_id trả về sau khi upload CSV thô"),
    class_name: Optional[str]  = Query(None, description="Tên class muốn lấy mẫu (bỏ trống = random)"),
    n_samples:  int            = Query(1,    description="Số mẫu muốn lấy", ge=1, le=50),
):
    """
    Lấy n_samples mẫu ngẫu nhiên từ session đã xử lý.

    Trả về vector số (từ features.csv) kèm class thật (từ classes.csv) để
    frontend dùng làm input test nhanh.
    """
    base_dir      = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    session_dir   = os.path.join(base_dir, PROCESSED_DIR, session_id)
    features_path = os.path.join(session_dir, "features.csv")
    classes_path  = os.path.join(session_dir, "classes.csv")

    if not os.path.exists(features_path) or not os.path.exists(classes_path):
        raise HTTPException(
            status_code=404,
            detail=f"Không tìm thấy dữ liệu session '{session_id}'. Hãy upload CSV thô trước.",
        )

    try:
        df_feat    = pd.read_csv(features_path)
        df_classes = pd.read_csv(classes_path)
        class_col  = "class" if "class" in df_classes.columns else df_classes.columns[0]
        labels     = df_classes[class_col].astype(str)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi đọc file: {e}")

    if len(df_feat) != len(labels):
        raise HTTPException(status_code=500, detail="features.csv và classes.csv không đồng bộ số dòng.")

    # Lọc theo class_name nếu có
    if class_name:
        mask = labels.str.strip() == class_name.strip()
        if not mask.any():
            available = sorted(labels.unique().tolist())
            raise HTTPException(
                status_code=404,
                detail=f"Class '{class_name}' không có trong session. Có sẵn: {available}",
            )
        df_feat_filtered = df_feat[mask].reset_index(drop=True)
        labels_filtered  = labels[mask].reset_index(drop=True)
    else:
        df_feat_filtered = df_feat
        labels_filtered  = labels

    # Random lấy n_samples dòng
    total = len(df_feat_filtered)
    indices = random.sample(range(total), min(n_samples, total))

    samples_out = []
    for idx in indices:
        row = df_feat_filtered.iloc[idx]
        # Chỉ lấy các cột số
        numeric_values = []
        for val in row.values:
            try:
                numeric_values.append(float(val))
            except (ValueError, TypeError):
                numeric_values.append(0.0)

        samples_out.append({
            "index"       : int(idx),
            "true_class"  : str(labels_filtered.iloc[idx]),
            "feature_names": list(df_feat_filtered.columns),
            "values"      : numeric_values,
        })

    return {
        "session_id"   : session_id,
        "class_filter" : class_name,
        "n_returned"   : len(samples_out),
        "total_in_class": total,
        "samples"      : samples_out,
    }
