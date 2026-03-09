"""
Incremental One-Class SVM
=========================
Module triển khai IncrementalOCSVM – One-Class SVM có khả năng học tăng cường.

Mỗi instance đại diện cho một lớp duy nhất.
File .pkl được đặt tên theo phiên bản: ví dụ "successful-01.pkl".
"""

import os
import joblib
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any

from sklearn.svm import OneClassSVM


class IncrementalOCSVM:
    """
    Modified One-Class Incremental SVM cho một lớp duy nhất.

    Attributes:
        class_name     (str):          Tên lớp, ví dụ 'successful'.
        version_name   (str):          Tên phiên bản, ví dụ 'successful-01'.
        nu             (float):        Tham số nu (tỷ lệ outlier dự kiến).
        gamma          (str|float):    Tham số gamma cho kernel RBF.
        kernel         (str):          Loại kernel ('rbf', 'linear', …).
        model          (OneClassSVM):  Model sklearn.
        memory_data    (np.ndarray):   Chỉ chứa Support Vectors sau mỗi lần train.
        is_trained     (bool):         Trạng thái đã train chưa.
    """

    def __init__(
        self,
        class_name: str,
        nu: float = 0.1,
        gamma: str = "scale",
        kernel: str = "rbf",
        model_dir: str = "models",
    ):
        self.class_name   = class_name
        self.version_name = f"{class_name}-01"   # Phiên bản mặc định
        self.nu           = nu
        self.gamma        = gamma
        self.kernel       = kernel
        self.model_dir    = model_dir

        # Khởi tạo model
        self.model   = OneClassSVM(nu=nu, gamma=gamma, kernel=kernel)

        # Trạng thái
        # NOTE (thiết kế): Chỉ lưu trữ các Support Vectors (SVs) vào memory_data
        # để tiết kiệm bộ nhớ, tránh data overlap và giữ được ranh giới quyết định.
        self.memory_data: Optional[np.ndarray] = None
        self.is_trained    = False
        self.created_at    = datetime.now()
        self.last_trained: Optional[datetime] = None
        self.training_history: list = []
        self.score_stats = {"min": -1.0, "max": 1.0}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        version_name: Optional[str] = None,
        save_model: bool = True,
    ) -> Dict[str, Any]:
        """
        Huấn luyện model lần đầu (hoặc huấn luyện lại toàn bộ).

        Args:
            X:            Dữ liệu huấn luyện, shape (n_samples, n_features).
            version_name: Tên phiên bản (ví dụ 'successful-01'). Nếu None dùng giá trị hiện tại.
            save_model:   Có lưu file .pkl hay không.

        Returns:
            Dict chứa metadata quá trình huấn luyện.
        """
        if len(X) < 2:
            raise ValueError("Cần ít nhất 2 mẫu để huấn luyện.")

        # Gán tên phiên bản
        if version_name:
            self.version_name = version_name

        print(f"  [Train] Class='{self.class_name}' | Version='{self.version_name}' | Samples={len(X)}")

        # Không chuẩn hoá lại – dữ liệu đã được Scale toàn cục ở Phase 0
        X_scaled = X

        # Tái khởi tạo model với tham số hiện tại (quan trọng khi retrain)
        self.model = OneClassSVM(nu=self.nu, gamma=self.gamma, kernel=self.kernel)
        self.model.fit(X_scaled)

        # Lấy chỉ số của các Support Vectors từ model sau khi fit
        # Lưu lại chính xác các điểm dữ liệu thô (hoặc đã scaled ở vòng ngoài) tương ứng
        if hasattr(self.model, "support_"):
            sv_indices = self.model.support_
            self.memory_data = X_scaled[sv_indices].copy()
        else:
            # Fallback (hiếm gặp nếu sklearn fail internals)
            self.memory_data = X_scaled.copy()

        self.is_trained   = True
        self.last_trained = datetime.now()

        # Lưu thống kê decision_function để sau này Normalize
        train_scores = self.model.decision_function(X_scaled)
        self.score_stats = {
            "min": float(np.min(train_scores)),
            "max": float(np.max(train_scores))
        }

        # Thống kê
        predictions = self.model.predict(X_scaled)
        n_inliers   = int(np.sum(predictions == 1))
        n_outliers  = int(np.sum(predictions == -1))

        info = {
            "class_name"        : self.class_name,
            "version_name"      : self.version_name,
            "n_input_samples"   : len(X),                 # Tổng điểm đưa vào fit (chỉ để tham khảo)
            "n_samples"         : len(self.memory_data),  # SVs được lưu lại thực tế
            "n_features"        : X.shape[1],
            "n_support_vectors" : len(self.memory_data),
            "n_inliers"         : n_inliers,
            "n_outliers"        : n_outliers,
            "inlier_ratio"      : round(n_inliers / len(X), 4),
            "trained_at"        : self.last_trained.isoformat(),
            "parameters"        : {"nu": self.nu, "gamma": self.gamma, "kernel": self.kernel},
        }
        self.training_history.append(info)

        if save_model:
            self.save()

        print(
            f"  [Train] ✓ Hoàn tất – Input: {len(X)} mẫu → "
            f"SVs lưu lại: {len(self.memory_data)} | "
            f"Inliers: {n_inliers}/{len(X)} ({info['inlier_ratio']*100:.1f}%)"
        )
        return info

    def retrain(
        self,
        X_new: np.ndarray,
        new_version: Optional[str] = None,
        save_model: bool = True,
    ) -> Dict[str, Any]:
        """
        Học tăng cường: gộp dữ liệu cũ + mới, huấn luyện lại, tăng phiên bản.

        Args:
            X_new:       Dữ liệu mới, shape (n_new_samples, n_features).
            new_version: Tên phiên bản mới. Nếu None tự động tăng (A-01 -> A-02).
            save_model:  Có lưu file .pkl hay không.

        Returns:
            Dict metadata quá trình huấn luyện.
        """
        if not self.is_trained:
            print(f"  [Retrain] Model chưa train. Chuyển sang train() lần đầu.")
            return self.train(X_new, version_name=new_version, save_model=save_model)

        # 1. Tự động tăng phiên bản nếu không truyền vào
        if new_version:
            self.version_name = new_version
        else:
            self.version_name = self._bump_version(self.version_name)

        print(f"  [Retrain] Class='{self.class_name}' | NewVersion='{self.version_name}'")

        # 2. Lấy dữ liệu cốt lõi (Support Vectors) từ memory
        if self.memory_data is not None and len(self.memory_data) >= 2:
            X_base = self.memory_data
            source = f"Support Vectors nòng cốt ({len(X_base)} điểm)"
        else:
            raise RuntimeError(
                f"Model '{self.class_name}' không có dữ liệu Support Vectors để retrain. "
                "Hãy train lại từ đầu."
            )

        print(f"  [Retrain] Dữ liệu: {source} + {len(X_new)} mẫu mới")

        # ── Kiểm tra số chiều features phải khớp nhau ─────────────────────────
        n_features_sv  = X_base.shape[1]
        n_features_new = X_new.shape[1]
        if n_features_sv != n_features_new:
            raise ValueError(
                f"[Retrain] Số features KHÔNG KHỚP!\n"
                f"  Model '{self.class_name}' (v{self.version_name}) được train với {n_features_sv} features.\n"
                f"  Dữ liệu mới có {n_features_new} features.\n"
                f"  → Hãy dùng đúng tập features như lúc train ban đầu, "
                f"hoặc train lại từ đầu với dữ liệu mới (retrain=False)."
            )

        # 3. TỰ ĐỘNG BẢO VỆ RANH GIỚI:
        # Gộp (Support Vectors cũ) + (Dữ liệu mới)
        X_combined = np.vstack([X_base, X_new])
        print(f"  [Retrain] Tối ưu hóa: Tổng dữ liệu huấn luyện = {len(X_combined)}")

        # 4. Huấn luyện lại trên tập dữ liệu đã tinh lọc
        return self.train(X_combined, version_name=self.version_name, save_model=save_model)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Dự đoán: 1 = inlier (thuộc lớp), -1 = outlier (không thuộc lớp).
        """
        self._check_trained()
        return self.model.predict(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Trả về giá trị decision function (điểm tin cậy).
        Giá trị càng cao -> càng thuộc lớp.
        """
        self._check_trained()
        return self.model.decision_function(X)

    def normalized_decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Trả về decision function đã chuẩn hoá MinMaxScaler [0, 1].
        """
        scores = self.decision_function(X)
        s_min = self.score_stats["min"]
        s_max = self.score_stats["max"]
        
        # Ngăn chia cho 0
        if s_max - s_min == 0:
            return np.zeros_like(scores)

        # Scale min-max
        norms = (scores - s_min) / (s_max - s_min)
        
        # Tuy inlier score dương (>0), những điểm outlier score < 0 sẽ làm norms < min.
        # Để đảm bảo tính công bằng (nếu score < 0 => out), ta có thể giữ scale tuyến tính, hoặc clamp.
        # Ở đây ta clamp dưới -1, trên 1 tránh nổ giá trị.
        return np.clip(norms, -1.0, 1.0)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, custom_path: Optional[str] = None) -> str:
        """
        Lưu model, scaler và metadata vào file .pkl.

        File được đặt tên theo version_name: models/successful-01.pkl

        Returns:
            Đường dẫn file đã lưu.
        """
        self._check_trained()
        os.makedirs(self.model_dir, exist_ok=True)

        filepath = custom_path or os.path.join(self.model_dir, f"{self.version_name}.pkl")

        payload = {
            "class_name"           : self.class_name,
            "version_name"         : self.version_name,
            "nu"                   : self.nu,
            "gamma"                : self.gamma,
            "kernel"               : self.kernel,
            "model"                : self.model,
            "memory_data"          : self.memory_data,
            "is_trained"           : self.is_trained,
            "created_at"           : self.created_at,
            "last_trained"         : self.last_trained,
            "training_history"     : self.training_history,
            "score_stats"          : self.score_stats,
        }
        joblib.dump(payload, filepath)
        print(f"  [Save] ✓ Lưu model tại: {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: str) -> "IncrementalOCSVM":
        """
        Load model từ file .pkl.

        Returns:
            Instance IncrementalOCSVM đã được khôi phục đầy đủ.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File model không tồn tại: '{filepath}'")

        payload = joblib.load(filepath)

        instance = cls(
            class_name = payload["class_name"],
            nu         = payload["nu"],
            gamma      = payload["gamma"],
            kernel     = payload["kernel"],
        )
        instance.version_name          = payload["version_name"]
        instance.model                 = payload["model"]
        
        # Backward compatibility cho các file .pkl cũ
        if "memory_data" in payload:
            instance.memory_data = payload["memory_data"]
        elif "support_vectors_cache" in payload and payload["support_vectors_cache"] is not None:
            instance.memory_data = payload["support_vectors_cache"]
        elif "training_data" in payload and payload["training_data"] is not None:
            instance.memory_data = payload["training_data"]
        else:
            instance.memory_data = None
            
        instance.is_trained            = payload["is_trained"]
        instance.created_at            = payload["created_at"]
        instance.last_trained          = payload["last_trained"]
        instance.training_history      = payload.get("training_history", [])
        instance.score_stats           = payload.get("score_stats", {"min": -1.0, "max": 1.0})
        print(f"  [Load] ✓ Đã load '{instance.version_name}' từ {filepath}")
        return instance

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def get_info(self) -> Dict[str, Any]:
        """Trả về thông tin tổng quan model."""
        n_sv = len(self.memory_data) if self.memory_data is not None else 0
        return {
            "class_name"        : self.class_name,
            "version_name"      : self.version_name,
            "is_trained"        : self.is_trained,
            "n_samples"         : n_sv,  # Từ nay n_samples chính thức là số lượng Support Vectors
            "n_support_vectors" : n_sv,
            "n_features"        : self.memory_data.shape[1] if self.memory_data is not None else 0,
            "last_trained"      : self.last_trained.isoformat() if self.last_trained else None,
            "parameters"        : {"nu": self.nu, "gamma": self.gamma, "kernel": self.kernel},
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_trained(self):
        if not self.is_trained:
            raise RuntimeError(
                f"Model '{self.class_name}' chưa được huấn luyện. Gọi train() trước."
            )

    @staticmethod
    def _bump_version(version_name: str) -> str:
        """
        Tự động tăng số phiên bản: 'A-01' -> 'A-02', 'X-09' -> 'X-10'.
        """
        try:
            base, num = version_name.rsplit("-", 1)
            return f"{base}-{int(num) + 1:02d}"
        except ValueError:
            return f"{version_name}-02"

    def __repr__(self) -> str:
        return (
            f"IncrementalOCSVM("
            f"class='{self.class_name}', "
            f"version='{self.version_name}', "
            f"trained={self.is_trained})"
        )
