"""
Incremental One-Class SVM
=========================
Module triển khai IncrementalOCSVM – One-Class SVM có khả năng học tăng cường.

Kiến trúc mới (Global Scaler):
    - Model KHÔNG tự scale dữ liệu. Dữ liệu đầu vào phải đã được
      chuẩn hóa bởi GlobalScalerManager trước khi truyền vào.
    - memory_data lưu trữ Support Vectors trong không gian đã scale toàn cục.
    
Cơ chế Retrain (SV Pruning):
    - Age Pruning:   Loại bỏ SVs có tuổi thọ ≥ age_threshold chu kỳ retrain.
    - Error Pruning: Nếu accuracy của model cũ trên X_new < error_threshold,
                     toàn bộ SVs lịch sử bị hủy bỏ hoàn toàn.
    - Memory Fusion: X_merged = X_new ∪ SV_pruned (sau pruning).
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

    Nhận dữ liệu ĐÃ ĐƯỢC SCALE bởi GlobalScalerManager từ bên ngoài.
    KHÔNG tự scale nội bộ – đây là nguyên tắc bất biến của kiến trúc mới.

    Attributes:
        class_name     (str):         Tên lớp.
        version_name   (str):         Tên phiên bản (ví dụ 'A-01').
        nu             (float):       Tham số nu.
        gamma          (str|float):   Tham số gamma cho kernel RBF.
        kernel         (str):         Loại kernel.
        model          (OneClassSVM): Model sklearn.
        memory_data    (np.ndarray):  Support Vectors (không gian đã scale toàn cục).
        sv_ages        (np.ndarray):  Tuổi thọ của từng SV (đơn vị: chu kỳ retrain).
        is_trained     (bool):        Trạng thái đã train chưa.
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
        self.version_name = f"{class_name}-01"
        self.nu           = nu
        self.gamma        = gamma
        self.kernel       = kernel
        self.model_dir    = model_dir

        self.model = OneClassSVM(nu=nu, gamma=gamma, kernel=kernel)

        self.memory_data: Optional[np.ndarray] = None   # SVs (in global-scaled space)
        self.sv_ages: Optional[np.ndarray] = None       # Tuổi thọ từng SV
        self.is_trained    = False
        self.created_at    = datetime.now()
        self.last_trained: Optional[datetime] = None
        self.training_history: list = []
        self.score_stats = {"min": -1.0, "max": 1.0}

    # ------------------------------------------------------------------
    # Training (Phase 1)
    # ------------------------------------------------------------------

    def train(
        self,
        X_scaled: np.ndarray,
        X_neg_scaled: Optional[np.ndarray] = None,
        version_name: Optional[str] = None,
        save_model: bool = True,
        nu_candidates: Optional[list] = None,
        gamma_candidates: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Huấn luyện model lần đầu (hoặc huấn luyện lại toàn bộ).

        Args:
            X_scaled:         Dữ liệu huấn luyện ĐÃ SCALE qua GlobalScaler,
                              shape (n_samples, n_features).
            X_neg_scaled:     (Tuỳ chọn) Dữ liệu các lớp khác ĐÃ SCALE,
                              dùng để GridSearch tối ưu hơn.
            version_name:     Tên phiên bản. Nếu None dùng giá trị hiện tại.
            save_model:       Có lưu file .pkl hay không.
            nu_candidates:    Danh sách nu để GridSearch.
            gamma_candidates: Danh sách gamma để GridSearch.

        Returns:
            Dict chứa metadata quá trình huấn luyện.
        """
        if len(X_scaled) < 4:
            raise ValueError(
                f"Cần ít nhất 4 mẫu để huấn luyện lớp '{self.class_name}'."
            )

        if version_name:
            self.version_name = version_name

        print(
            f"  [Train] Class='{self.class_name}' | "
            f"Version='{self.version_name}' | Samples={len(X_scaled)}"
        )

        # Grid Search nếu có dữ liệu âm (Tăng cường Nu và Gamma để ép ranh giới Khít lại)
        _nu_list    = nu_candidates    or [0.05, 0.1, 0.15, 0.2, 0.25]
        _gamma_list = gamma_candidates or ["scale", 0.1, 1.0, 5.0, 10.0]

        if X_neg_scaled is not None and len(X_neg_scaled) > 0:
            print(
                f"  [Train] GridSearch cho '{self.class_name}' "
                f"với {len(X_neg_scaled)} Negative samples..."
            )
            best_score = -np.inf
            best_nu    = self.nu
            best_gamma = self.gamma
            best_model = None

            for n in _nu_list:
                for g in _gamma_list:
                    tmp = OneClassSVM(nu=n, gamma=g, kernel=self.kernel)
                    tmp.fit(X_scaled)

                    tp = np.mean(tmp.predict(X_scaled)    ==  1)   # inlier rate
                    tn = np.mean(tmp.predict(X_neg_scaled) == -1)   # rejection rate
                    score = (tp + tn) / 2.0

                    if score > best_score:
                        best_score = score
                        best_nu    = n
                        best_gamma = g
                        best_model = tmp

            print(
                f"  [Train] GridSearch xong. "
                f"Best (nu={best_nu}, gamma={best_gamma}), score={best_score:.4f}"
            )
            self.nu    = best_nu
            self.gamma = best_gamma
            self.model = best_model
        else:
            self.model = OneClassSVM(nu=self.nu, gamma=self.gamma, kernel=self.kernel)
            self.model.fit(X_scaled)

        # Trích xuất Support Vectors
        if hasattr(self.model, "support_"):
            sv_indices       = self.model.support_
            self.memory_data = X_scaled[sv_indices].copy()
        else:
            self.memory_data = X_scaled.copy()

        # SVs mới → tuổi thọ = 0
        self.sv_ages = np.zeros(len(self.memory_data), dtype=int)

        self.is_trained   = True
        self.last_trained = datetime.now()

        # Lưu thống kê score để normalize sau này (nếu cần)
        train_scores = self.model.decision_function(X_scaled)
        self.score_stats = {
            "min": float(np.min(train_scores)),
            "max": float(np.max(train_scores)),
        }

        predictions = self.model.predict(X_scaled)
        n_inliers   = int(np.sum(predictions == 1))

        info = {
            "class_name"        : self.class_name,
            "version_name"      : self.version_name,
            "n_input_samples"   : len(X_scaled),
            "n_samples"         : len(self.memory_data),
            "n_features"        : X_scaled.shape[1],
            "n_support_vectors" : len(self.memory_data),
            "n_inliers"         : n_inliers,
            "n_outliers"        : int(np.sum(predictions == -1)),
            "inlier_ratio"      : round(n_inliers / len(X_scaled), 4),
            "trained_at"        : self.last_trained.isoformat(),
            "parameters"        : {"nu": self.nu, "gamma": self.gamma, "kernel": self.kernel},
        }
        self.training_history.append(info)

        if save_model:
            self.save()

        print(
            f"  [Train] ✓ Hoàn tất – Input: {len(X_scaled)} mẫu → "
            f"SVs: {len(self.memory_data)} | "
            f"Inliers: {n_inliers}/{len(X_scaled)} ({info['inlier_ratio']*100:.1f}%)"
        )
        return info

    # ------------------------------------------------------------------
    # Retrain with SV Pruning (Phase 2)
    # ------------------------------------------------------------------

    def retrain(
        self,
        X_new_scaled: np.ndarray,
        X_neg_scaled: Optional[np.ndarray] = None,
        new_version: Optional[str] = None,
        save_model: bool = True,
        age_threshold: int = 5,
        error_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Học tăng cường với cơ chế SV Pruning.

        Pha 2 – Quy trình:
          1. Age Pruning:   Loại SVs cũ có tuổi ≥ age_threshold.
          2. Error Pruning: Nếu model cũ predict sai trên X_new < error_threshold,
                            hủy toàn bộ SVs lịch sử.
          3. Memory Fusion: X_merged = X_new_scaled ∪ SV_pruned.
          4. Refit:         Train lại trên X_merged, sinh SVs thế hệ mới.

        Args:
            X_new_scaled:    Dữ liệu mới ĐÃ SCALE qua GlobalScaler.
            X_neg_scaled:    (Tuỳ chọn) Dữ liệu âm ĐÃ SCALE, cho GridSearch.
            new_version:     Tên phiên bản mới. None → tự tăng.
            save_model:      Có lưu .pkl hay không.
            age_threshold:   Số chu kỳ tối đa một SV được giữ lại (default=5).
            error_threshold: Ngưỡng accuracy tối thiểu (default=0.5).
                             Nếu accuracy < threshold, hủy toàn bộ SVs cũ.

        Returns:
            Dict metadata.
        """
        if not self.is_trained:
            print(f"  [Retrain] Model chưa train. Chuyển sang train() lần đầu.")
            return self.train(
                X_new_scaled, X_neg_scaled=X_neg_scaled,
                version_name=new_version, save_model=save_model
            )

        # Tăng phiên bản
        self.version_name = new_version or self._bump_version(self.version_name)
        print(f"  [Retrain] Class='{self.class_name}' | NewVersion='{self.version_name}'")

        if X_new_scaled.shape[1] != self.memory_data.shape[1]:
            raise ValueError(
                f"[Retrain] Số features không khớp! "
                f"Model='{self.class_name}' đã train với {self.memory_data.shape[1]} features, "
                f"dữ liệu mới có {X_new_scaled.shape[1]} features."
            )

        # ── Bước 1: Error Pruning ─────────────────────────────────────────────
        if self.memory_data is not None and len(self.memory_data) > 0:
            preds_on_new = self.model.predict(X_new_scaled)
            accuracy     = float(np.mean(preds_on_new == 1))
            print(f"  [Retrain] Error Pruning – Model accuracy trên X_new: {accuracy:.2%}")

            if accuracy < error_threshold:
                print(
                    f"  [Retrain] ⚠ Accuracy {accuracy:.2%} < ngưỡng {error_threshold:.2%}. "
                    f"Hủy toàn bộ {len(self.memory_data)} SVs lịch sử (Concept Drift detected)."
                )
                SV_pruned = np.empty((0, X_new_scaled.shape[1]), dtype=float)
                sv_ages_pruned = np.empty(0, dtype=int)
            else:
                # ── Bước 2: Age Pruning ───────────────────────────────────────────
                if self.sv_ages is None:
                    self.sv_ages = np.zeros(len(self.memory_data), dtype=int)
                aged_sv_ages = self.sv_ages + 1  # Tăng tuổi
                keep_mask   = aged_sv_ages < age_threshold
                n_total     = len(self.memory_data)
                n_pruned_age = int(np.sum(~keep_mask))
                SV_pruned   = self.memory_data[keep_mask]
                sv_ages_pruned = aged_sv_ages[keep_mask]
                print(
                    f"  [Retrain] Age Pruning – Giữ lại {len(SV_pruned)}/{n_total} SVs "
                    f"(loại {n_pruned_age} SVs già ≥ {age_threshold} chu kỳ)."
                )
        else:
            SV_pruned      = np.empty((0, X_new_scaled.shape[1]), dtype=float)
            sv_ages_pruned = np.empty(0, dtype=int)

        # ── Bước 3: Memory Fusion ─────────────────────────────────────────────
        if len(SV_pruned) > 0:
            X_merged = np.vstack([SV_pruned, X_new_scaled])
            # SVs cũ giữ nguyên tuổi, mẫu mới tuổi = 0
            merged_ages = np.concatenate([
                sv_ages_pruned,
                np.zeros(len(X_new_scaled), dtype=int)
            ])
            print(f"  [Retrain] Memory Fusion: {len(SV_pruned)} SVs cũ + {len(X_new_scaled)} mới = {len(X_merged)} tổng")
        else:
            X_merged    = X_new_scaled.copy()
            merged_ages = np.zeros(len(X_new_scaled), dtype=int)
            print(f"  [Retrain] Chỉ dùng dữ liệu mới ({len(X_merged)} mẫu) – SVs cũ đã bị hủy.")

        if len(X_merged) < 4:
            raise ValueError(
                f"Dữ liệu gộp sau pruning chỉ có {len(X_merged)} mẫu "
                f"(cần ≥ 4). Hãy cung cấp thêm dữ liệu mới."
            )

        # ── Bước 4: Refit ─────────────────────────────────────────────────────
        # Lưu merged_ages tạm để hàm train() không ghi đè sv_ages
        _tmp_ages = merged_ages

        info = self.train(
            X_merged,
            X_neg_scaled  = X_neg_scaled,
            version_name  = self.version_name,
            save_model    = False,  # Tự save bên dưới sau khi cập nhật ages
        )

        # Cập nhật sv_ages theo SVs mới được chọn sau refit
        if hasattr(self.model, "support_") and self.memory_data is not None:
            sv_idx  = self.model.support_
            # Ánh xạ index SV → tuổi tương ứng trong X_merged
            new_sv_ages = _tmp_ages[sv_idx] if len(sv_idx) > 0 else np.zeros(0, dtype=int)
            self.sv_ages = new_sv_ages
        else:
            self.sv_ages = np.zeros(len(self.memory_data) if self.memory_data is not None else 0, dtype=int)

        if save_model:
            self.save()

        return info

    # ------------------------------------------------------------------
    # Prediction (Phase 3)
    # ------------------------------------------------------------------

    def predict(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Dự đoán: 1 = inlier (thuộc lớp), -1 = outlier.
        
        X_scaled phải đã được transform bởi GlobalScaler.
        """
        self._check_trained()
        return self.model.predict(X_scaled)

    def decision_function(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Trả về điểm quyết định (Dương = trong vỏ bọc; Âm = ngoài).
        
        X_scaled phải đã được transform bởi GlobalScaler.
        """
        self._check_trained()
        return self.model.decision_function(X_scaled)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, custom_path: Optional[str] = None) -> str:
        """Lưu model và metadata vào file .pkl."""
        self._check_trained()
        os.makedirs(self.model_dir, exist_ok=True)

        filepath = custom_path or os.path.join(self.model_dir, f"{self.version_name}.pkl")

        payload = {
            "class_name"      : self.class_name,
            "version_name"    : self.version_name,
            "nu"              : self.nu,
            "gamma"           : self.gamma,
            "kernel"          : self.kernel,
            "model"           : self.model,
            "memory_data"     : self.memory_data,
            "sv_ages"         : self.sv_ages,
            "is_trained"      : self.is_trained,
            "created_at"      : self.created_at,
            "last_trained"    : self.last_trained,
            "training_history": self.training_history,
            "score_stats"     : self.score_stats,
        }
        joblib.dump(payload, filepath)
        print(f"  [Save] ✓ Lưu model tại: {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: str) -> "IncrementalOCSVM":
        """Load model từ file .pkl."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File model không tồn tại: '{filepath}'")

        payload  = joblib.load(filepath)
        instance = cls(
            class_name = payload["class_name"],
            nu         = payload.get("nu", 0.1),
            gamma      = payload.get("gamma", "scale"),
            kernel     = payload.get("kernel", "rbf"),
        )
        instance.version_name     = payload["version_name"]
        instance.model            = payload["model"]
        instance.is_trained       = payload["is_trained"]
        instance.created_at       = payload["created_at"]
        instance.last_trained     = payload["last_trained"]
        instance.training_history = payload.get("training_history", [])
        instance.score_stats      = payload.get("score_stats", {"min": -1.0, "max": 1.0})

        # memory_data: backward compat
        if "memory_data" in payload and payload["memory_data"] is not None:
            instance.memory_data = payload["memory_data"]
        elif "support_vectors_cache" in payload:
            instance.memory_data = payload["support_vectors_cache"]
        elif "training_data" in payload:
            instance.memory_data = payload["training_data"]
        else:
            instance.memory_data = None

        # sv_ages: mới thêm, backward compat
        if "sv_ages" in payload and payload["sv_ages"] is not None:
            instance.sv_ages = payload["sv_ages"]
        elif instance.memory_data is not None:
            instance.sv_ages = np.zeros(len(instance.memory_data), dtype=int)
        else:
            instance.sv_ages = None

        print(f"  [Load] ✓ Đã load '{instance.version_name}' từ {filepath}")
        return instance

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def get_info(self) -> Dict[str, Any]:
        n_sv = len(self.memory_data) if self.memory_data is not None else 0
        return {
            "class_name"        : self.class_name,
            "version_name"      : self.version_name,
            "is_trained"        : self.is_trained,
            "n_samples"         : n_sv,
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
        """'A-01' → 'A-02', 'X-09' → 'X-10'."""
        try:
            base, num = version_name.rsplit("-", 1)
            return f"{base}-{int(num) + 1:02d}"
        except ValueError:
            return f"{version_name}-02"

    def __repr__(self):
        return (
            f"IncrementalOCSVM("
            f"class='{self.class_name}', "
            f"version='{self.version_name}', "
            f"trained={self.is_trained})"
        )
