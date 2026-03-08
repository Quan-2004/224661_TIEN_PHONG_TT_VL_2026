"""
Data Loader & Validator
========================
Module tiện ích để tải và kiểm tra tính nhất quán của 3 file CSV đầu vào:
  - samples.csv  : Ma trận mẫu dữ liệu (n_samples × n_features)
  - features.csv : Tên các features (1 hàng header)
  - classes.csv  : Nhãn lớp cho mỗi mẫu (n_samples × 1)

Quy tắc validate:
  1. Cả 3 file phải tồn tại.
  2. Xử lý dòng trống/NaN: dropna(all) + fillna(mean) trước khi đếm dòng.
  3. Ba file phải có cúng số dòng sau khi làm sạch (triple-lock assertion).
  4. features.csv phải chứa toàn giá trị số.
  5. Kiểm tra shape (X, y) tại điểm huấn luyện bằng validate_X_y_shape().
"""

import logging
import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def load_and_validate_csv(
    samples_path: str,
    features_path: str,
    classes_path: str,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Tải và validate 3 file CSV (tất cả đều có HEADER – khớp với DataProcessor).

    DataProcessor lưu 3 file CÓ header (dòng đầu là tên cột):
      - features.csv  : Ma trận số đã scale, header = tên features  → X lấy từ đây
      - classes.csv   : Cột 'class', header = 'class'
      - samples.csv   : Các cột định danh (ID, tên, ...)

    Args:
        samples_path:  Đường dẫn file samples.csv.
        features_path: Đường dẫn file features.csv (ma trận số với header là tên features).
        classes_path:  Đường dẫn file classes.csv  (cột nhãn, header 'class').

    Returns:
        Tuple (X, feature_names, class_labels)
            X              : np.ndarray shape (n_samples, n_features) – từ features.csv
            feature_names  : List[str] tên features (header của features.csv)
            class_labels   : List[str] nhãn lớp cho từng mẫu

    Raises:
        FileNotFoundError : Nếu một trong 3 file không tồn tại.
        ValueError        : Nếu dữ liệu không nhất quán.
    """
    # --- Kiểm tra file tồn tại ---
    for path in (samples_path, features_path, classes_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File không tồn tại: '{path}'")

    # --- Đọc file VỚI header (header=0 là mặc định pandas) ---
    df_samples  = pd.read_csv(samples_path,  header=0)
    df_features = pd.read_csv(features_path, header=0)
    df_classes  = pd.read_csv(classes_path,  header=0)

    # --- Bước 1: Xử lý nhiễu – xoá dòng hoàn toàn trống ---
    n_raw = len(df_samples)
    df_samples  = df_samples.dropna(how="all").reset_index(drop=True)
    df_features = df_features.dropna(how="all").reset_index(drop=True)
    df_classes  = df_classes.dropna(how="all").reset_index(drop=True)

    dropped = n_raw - len(df_samples)
    if dropped > 0:
        logger.warning(
            "[DataLoader] Đã bỏ %d dòng trống hoàn toàn khỏi samples.csv.", dropped
        )

    # --- Bước 2: Điền giá trị thiếu trong features (NaN cục bộ) ---
    numeric_cols = df_features.select_dtypes(include="number").columns
    col_means    = df_features[numeric_cols].mean()
    df_features[numeric_cols] = df_features[numeric_cols].fillna(col_means)

    # --- Bước 3: Triple-lock – cả 3 file phải có cùng số dòng ---
    n_samples   = len(df_samples)
    n_feat_rows = len(df_features)
    n_cls_rows  = len(df_classes)

    if not (n_samples == n_feat_rows == n_cls_rows):
        raise ValueError(
            f"Số dòng không khớp – samples={n_samples}, "
            f"features={n_feat_rows}, classes={n_cls_rows}. "
            "Ba file phải có cùng số hàng sau khi loại dòng trống."
        )

    # --- Bước 4: Lấy tên features từ header của features.csv ---
    feature_names: List[str] = df_features.columns.tolist()

    # --- Bước 5: Chuyển features → numpy array (X dùng để train) ---
    # Phát hiện cột không phải số và báo cụ thể
    non_numeric_cols = [
        col for col in df_features.columns
        if not pd.api.types.is_numeric_dtype(df_features[col])
    ]
    if non_numeric_cols:
        raise ValueError(
            f"features.csv chứa {len(non_numeric_cols)} cột không phải số: "
            f"{non_numeric_cols[:5]}{'...' if len(non_numeric_cols) > 5 else ''}. "
            "\n\nNếu bạn đang upload file CSV thô (raw dataset), hãy dùng endpoint "
            "POST /upload-raw thay vì POST /upload. "
            "Endpoint /upload-raw sẽ tự động encode categorical columns và chuẩn hoá dữ liệu. "
            "\nNếu bạn đang dùng POST /upload (3 file thủ công), file features.csv "
            "phải chứa toàn bộ giá trị số (đã được xử lý trước)."
        )

    try:
        X = df_features.values.astype(np.float64)
    except ValueError as e:
        raise ValueError(f"features.csv không thể chuyển sang float64: {e}") from e

    # --- Bước 6: Đọc nhãn lớp từ cột 'class' (hoặc cột đầu tiên nếu tên khác) ---
    class_col = "class" if "class" in df_classes.columns else df_classes.columns[0]
    class_labels: List[str] = df_classes[class_col].astype(str).tolist()

    # --- Bước 7: Post-parse shape assertion ---
    assert len(X) == len(class_labels), (
        f"[DataLoader] BUG: X.shape[0]={len(X)} != len(class_labels)={len(class_labels)} "
        "sau khi parse. Báo cáo lỗi này cho developer."
    )

    logger.info(
        "[DataLoader] ✓ Đã load: %d mẫu × %d features | Classes: %s",
        n_samples, len(feature_names), sorted(set(class_labels)),
    )
    print(f"  [DataLoader] ✓ Đã load: {n_samples} mẫu × {len(feature_names)} features")
    print(f"  [DataLoader] Classes: {sorted(set(class_labels))}")

    return X, feature_names, class_labels


def load_numpy_from_csv(filepath: str, dtype: type = np.float64, has_header: bool = True) -> np.ndarray:
    """
    Tải file CSV đơn giản thành numpy array.

    Args:
        filepath:   Đường dẫn file CSV.
        dtype:      Kiểu dữ liệu numpy.
        has_header: True nếu file có dòng header (mặc định True – khớp với DataProcessor).

    Returns:
        np.ndarray
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File không tồn tại: '{filepath}'")
    return pd.read_csv(filepath, header=0 if has_header else None).values.astype(dtype)


def split_by_class(
    X: np.ndarray,
    class_labels: List[str],
) -> Dict[str, np.ndarray]:
    """
    Chia dữ liệu theo nhãn lớp.

    Args:
        X:             Ma trận dữ liệu (n_samples, n_features).
        class_labels:  Danh sách nhãn (length = n_samples).

    Returns:
        Dict mapping class_name → X_class (np.ndarray)
    """
    labels_array = np.array(class_labels)
    unique_classes = np.unique(labels_array)

    result: Dict[str, np.ndarray] = {}
    for cls in unique_classes:
        mask = labels_array == cls
        result[cls] = X[mask]
        print(f"  [DataLoader] Class '{cls}': {int(np.sum(mask))} mẫu")

    return result


def validate_csv_consistency(
    samples_path: str,
    features_path: str,
    classes_path: str,
) -> Dict[str, Any]:
    """
    Kiểm tra tính nhất quán mà không load toàn bộ dữ liệu.
    Trả về report dict thay vì raise exception.

    Returns:
        {
          "valid": bool,
          "n_samples": int,
          "n_features": int,
          "n_classes": int,
          "unique_classes": list,
          "errors": list[str]
        }
    """
    errors: List[str] = []
    report: Dict[str, Any] = {
        "valid": False,
        "n_samples": 0,
        "n_features": 0,
        "n_classes": 0,
        "unique_classes": [],
        "errors": errors,
    }

    for name, path in [("samples", samples_path), ("features", features_path), ("classes", classes_path)]:
        if not os.path.exists(path):
            errors.append(f"File '{name}' không tồn tại: {path}")

    if errors:
        return report

    try:
        X, feature_names, class_labels = load_and_validate_csv(
            samples_path, features_path, classes_path
        )
        unique_classes = sorted(set(class_labels))
        report.update({
            "valid"         : True,
            "n_samples"     : len(X),
            "n_features"    : X.shape[1],
            "n_classes"     : len(unique_classes),
            "unique_classes": unique_classes,
        })
    except (ValueError, FileNotFoundError) as e:
        errors.append(str(e))

    return report


def validate_X_y_shape(X: np.ndarray, y, context: str = "") -> None:
    """
    Kiểm tra tính nhất quán giữa ma trận đặc trưng X và mảng nhãn y
    trước khi gọi model.fit().

    Args:
        X:       Ma trận đặc trưng, shape (n_samples, n_features).
        y:       Mảng nhãn lớp, length = n_samples.
        context: Chuỗi mô tả ngữ cảnh (ví dụ tên class) để lỗi rõ ràng hơn.

    Raises:
        ValueError: Nếu số dòng của X và độ dài y không khớp.
        ValueError: Nếu X rỗng hoặc có 0 features.
    """
    n_X = len(X)
    n_y = len(y)
    ctx = f"[{context}] " if context else ""

    if n_X == 0:
        raise ValueError(f"{ctx}X rỗng – không có dữ liệu để huấn luyện.")
    if X.ndim < 2 or X.shape[1] == 0:
        raise ValueError(
            f"{ctx}X phải là ma trận 2D với ít nhất 1 feature. "
            f"Shape hiện tại: {X.shape}"
        )
    if n_X != n_y:
        raise ValueError(
            f"{ctx}Shape không khớp – X có {n_X} dòng nhưng y có {n_y} phần tử. "
            "Kiểm tra lại bước lọc dữ liệu theo lớp."
        )
    logger.debug("%svalidate_X_y_shape OK – %d×%d", ctx, n_X, X.shape[1])
