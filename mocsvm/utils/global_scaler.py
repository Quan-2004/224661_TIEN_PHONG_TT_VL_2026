"""
mocsvm/utils/global_scaler.py
==============================
GlobalScalerManager – Quản lý StandardScaler toàn cục duy nhất.

Tầm quan trọng kỹ thuật:
    Đảm bảo mọi model OC-SVM hoạt động trên cùng một hệ quy chiếu tọa độ.
    Tránh sai lệch khoảng cách do từng lớp tự chuẩn hóa riêng.
    
Quy tắc bất biến:
    - fit_and_save()   → CHỈ gọi 1 lần khi nhận dữ liệu train MỚI HOÀN TOÀN.
    - transform()      → Luôn dùng để scale dữ liệu train / retrain / test.
    - TUYỆT ĐỐI CẤM   → gọi fit_transform() sau lần fit đầu tiên.
"""

import os
import joblib
import numpy as np
from datetime import datetime
from typing import Optional

from sklearn.preprocessing import StandardScaler


_GLOBAL_SCALER_FILENAME = "global_scaler.pkl"


class GlobalScalerManager:
    """
    Quản lý một StandardScaler toàn cục duy nhất cho toàn bộ hệ thống.

    Scaler được fit một lần trên toàn bộ X_train (mọi lớp gộp lại),
    sau đó đóng băng và chỉ dùng transform() cho mọi dữ liệu sau này.

    Attributes:
        model_dir  (str):             Thư mục chứa file global_scaler.pkl.
        scaler     (StandardScaler):  Scaler đã fit hoặc None.
        fitted_at  (datetime|None):   Thời điểm fit.
        n_samples  (int):             Số mẫu dùng để fit.
        n_features (int):             Số features.
    """

    def __init__(self, model_dir: str = "models"):
        self.model_dir  = model_dir
        self.scaler: Optional[StandardScaler] = None
        self.fitted_at: Optional[datetime] = None
        self.n_samples  = 0
        self.n_features = 0
        self.feature_names: Optional[list] = None

    # ------------------------------------------------------------------
    # Fit & Save
    # ------------------------------------------------------------------

    def fit_and_save(self, X: np.ndarray, feature_names: Optional[list] = None) -> "GlobalScalerManager":
        """
        Fit scaler trên toàn bộ X (gộp mọi lớp) và lưu vào disk.

        CẢNH BÁO: Gọi hàm này sẽ tạo ra một hệ quy chiếu mới hoàn toàn.
        Mọi model đang tồn tại sẽ làm việc trên không gian cũ → không tương thích.
        Nên xóa và train lại tất cả model sau khi gọi hàm này.

        Args:
            X: Ma trận dữ liệu thô (chưa scale), shape (n_samples, n_features).
            feature_names: (Tùy chọn) Danh sách tên cột dùng để Predict về sau.

        Returns:
            self (để chaining).
        """
        if len(X) == 0:
            raise ValueError("[GlobalScaler] X rỗng – không thể fit.")

        print(f"  [GlobalScaler] Fitting trên {len(X):,} mẫu, {X.shape[1]} features...")
        self.scaler     = StandardScaler()
        self.scaler.fit(X)
        self.fitted_at  = datetime.now()
        self.n_samples  = len(X)
        self.n_features = X.shape[1]
        self.feature_names = feature_names

        os.makedirs(self.model_dir, exist_ok=True)
        self._save()
        print(f"  [GlobalScaler] ✓ Fit xong. μ={self.scaler.mean_[:3]}... σ={self.scaler.scale_[:3]}...")
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Chuẩn hóa X bằng scaler đã đóng băng.

        TUYỆT ĐỐI KHÔNG gọi fit() hay fit_transform() ở đây.

        Args:
            X: Ma trận thô, shape (n_samples, n_features).

        Returns:
            X đã scale, cùng shape.

        Raises:
            RuntimeError: Nếu scaler chưa được fit.
        """
        self._check_fitted()
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"[GlobalScaler] Số features không khớp! "
                f"Scaler được fit với {self.n_features} features, "
                f"nhưng X có {X.shape[1]} features."
            )
        return self.scaler.transform(X)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def _save(self) -> str:
        """Lưu scaler vào disk."""
        path = self._pkl_path()
        payload = {
            "scaler"    : self.scaler,
            "fitted_at" : self.fitted_at,
            "n_samples" : self.n_samples,
            "n_features": self.n_features,
            "feature_names": self.feature_names,
        }
        joblib.dump(payload, path)
        print(f"  [GlobalScaler] ✓ Đã lưu tại: {path}")
        return path

    def load(self) -> bool:
        """
        Nạp scaler từ disk nếu tồn tại.

        Returns:
            True nếu load thành công, False nếu file chưa có.
        """
        path = self._pkl_path()
        if not os.path.exists(path):
            return False
        payload = joblib.load(path)
        self.scaler     = payload["scaler"]
        self.fitted_at  = payload.get("fitted_at")
        self.n_samples  = payload.get("n_samples", 0)
        self.n_features = payload.get("n_features", 0)
        self.feature_names = payload.get("feature_names", None)
        print(f"  [GlobalScaler] ✓ Đã load từ {path} (fit lúc {self.fitted_at})")
        return True

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def is_fitted(self) -> bool:
        return self.scaler is not None

    def get_info(self) -> dict:
        return {
            "is_fitted" : self.is_fitted(),
            "fitted_at" : self.fitted_at.isoformat() if self.fitted_at else None,
            "n_samples" : self.n_samples,
            "n_features": self.n_features,
            "pkl_path"  : self._pkl_path(),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pkl_path(self) -> str:
        return os.path.join(self.model_dir, _GLOBAL_SCALER_FILENAME)

    def _check_fitted(self):
        if self.scaler is None:
            raise RuntimeError(
                "[GlobalScaler] Scaler chưa được fit. "
                "Hãy gọi fit_and_save() trước khi transform()."
            )

    def __repr__(self):
        status = f"fitted={self.n_samples}samples" if self.is_fitted() else "NOT FITTED"
        return f"GlobalScalerManager({status}, model_dir='{self.model_dir}')"
