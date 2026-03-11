"""
Multi-Class One-Class SVM Manager
===================================
Module điều phối nhiều model IncrementalOCSVM (phương pháp One-vs-Rest).

Chiến lược dự đoán (Phase 3):
    1. Tính decision_function() của mọi model cho từng mẫu.
    2. Lọc các model có raw_score > 0 (nằm trong vỏ bọc).
    3. Nếu KHÔNG có model nào > 0 → "unknown" (Unknown Concept).
    4. Nếu đúng 1 model > 0 → gán nhãn đó.
    5. Nếu ≥ 2 model > 0 (chồng chéo) → Euclidean Tie-break:
       Tính khoảng cách Euclid từ x đến SV gần nhất của từng lớp.
       Lớp nào có d_min nhỏ hơn → thắng.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from mocsvm.core.incremental import IncrementalOCSVM
from mocsvm.core.manifest_manager import ManifestManager


class MultiClassOCSVM:
    """
    Quản lý tập hợp các IncrementalOCSVM, mỗi model ứng một lớp.

    Attributes:
        model_dir (str):            Thư mục lưu file .pkl.
        manifest  (ManifestManager): Bộ quản lý file XML.
        models    (Dict[str, IncrementalOCSVM]): Models đang hoạt động.
    """

    def __init__(
        self,
        model_dir: str = "models",
        manifest_path: str = "models/global_manifest.xml",
    ):
        self.model_dir = model_dir
        self.manifest  = ManifestManager(manifest_path)
        self.models: Dict[str, IncrementalOCSVM] = {}
        os.makedirs(model_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_class(
        self,
        class_name: str,
        X_scaled: np.ndarray,
        X_neg_scaled: Optional[np.ndarray] = None,
        version_name: Optional[str] = None,
        nu: float = 0.05,
        gamma: str = "scale",
        kernel: str = "rbf",
    ) -> Dict[str, Any]:
        """
        Huấn luyện model lần đầu cho một lớp.

        Args:
            class_name:    Tên lớp.
            X_scaled:      Dữ liệu ĐÃ SCALE qua GlobalScaler.
            X_neg_scaled:  (Tuỳ chọn) Dữ liệu lớp khác ĐÃ SCALE, cho GridSearch.
            version_name:  Tên phiên bản.
            nu, gamma, kernel: Tham số OC-SVM.

        Returns:
            Dict metadata.
        """
        vname = version_name or f"{class_name}-01"

        model = IncrementalOCSVM(
            class_name = class_name,
            nu         = nu,
            gamma      = gamma,
            kernel     = kernel,
            model_dir  = self.model_dir,
        )
        info = model.train(X_scaled, X_neg_scaled=X_neg_scaled, version_name=vname, save_model=True)
        self.models[class_name] = model

        pkl_path = os.path.join(self.model_dir, f"{vname}.pkl")
        self.manifest.update_class(
            class_name       = class_name,
            version_name     = vname,
            pkl_path         = pkl_path,
            metadata         = {
                "kernel"    : kernel,
                "gamma"     : str(gamma),
                "nu"        : str(nu),
                "n_samples" : info["n_samples"],
                "n_features": info["n_features"],
                "trained_at": info["trained_at"],
            },
            performance      = {
                "inlier_ratio"     : info.get("inlier_ratio", ""),
                "n_support_vectors": info.get("n_support_vectors", ""),
            },
            pkl_save_success = os.path.exists(pkl_path),
        )
        return info

    def retrain_class(
        self,
        class_name: str,
        X_new_scaled: np.ndarray,
        X_neg_scaled: Optional[np.ndarray] = None,
        new_version: Optional[str] = None,
        age_threshold: int = 5,
        error_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Học tăng cường (retrain) với SV Pruning cho một lớp đã có model.

        Args:
            class_name:       Tên lớp cần retrain.
            X_new_scaled:     Dữ liệu mới ĐÃ SCALE qua GlobalScaler.
            X_neg_scaled:     (Tuỳ chọn) Dữ liệu âm ĐÃ SCALE, cho GridSearch.
            new_version:      Phiên bản mới. None → tự tăng.
            age_threshold:    Ngưỡng tuổi SV (chu kỳ).
            error_threshold:  Ngưỡng accuracy để error pruning.
        """
        if class_name not in self.models:
            try:
                self.load_class(class_name)
            except Exception:
                raise ValueError(
                    f"Lớp '{class_name}' chưa được train. Hãy gọi train_class() trước."
                )

        model = self.models[class_name]
        info  = model.retrain(
            X_new_scaled    = X_new_scaled,
            X_neg_scaled    = X_neg_scaled,
            new_version     = new_version,
            save_model      = True,
            age_threshold   = age_threshold,
            error_threshold = error_threshold,
        )

        pkl_path = os.path.join(self.model_dir, f"{model.version_name}.pkl")
        self.manifest.update_class(
            class_name       = class_name,
            version_name     = model.version_name,
            pkl_path         = pkl_path,
            metadata         = {
                "kernel"    : model.kernel,
                "gamma"     : str(model.gamma),
                "nu"        : str(model.nu),
                "n_samples" : info["n_samples"],
                "n_features": info["n_features"],
                "trained_at": info["trained_at"],
            },
            performance      = {
                "inlier_ratio"     : info.get("inlier_ratio", ""),
                "n_support_vectors": info.get("n_support_vectors", ""),
            },
            pkl_save_success = os.path.exists(pkl_path),
        )
        return info

    # ------------------------------------------------------------------
    # Prediction (Phase 3) – Euclidean SV Tie-break
    # ------------------------------------------------------------------

    def predict_multi(
        self,
        X_scaled: np.ndarray,
        min_margin: float = 0.0,
        return_scores: bool = False,
    ) -> Tuple[List[str], Optional[np.ndarray]]:
        """
        Dự đoán nhãn với chiến lược One-vs-Rest + Euclidean SV Tie-break.

        Thuật toán (theo đặc tả Phase 3):
          1. Tính raw_score = decision_function(x) cho từng model.
          2. positive_mask = models với score > 0.
          3. Không có model nào > 0 → "unknown".
          4. Đúng 1 model > 0 → gán nhãn.
          5. ≥ 2 models > 0 (overlap) → Euclidean Tie-break:
             d_min_i = min(||x - sv||₂, for sv in SV_i)
             winner = argmin(d_min).

        Args:
            X_scaled:     Dữ liệu ĐÃ SCALE qua GlobalScaler.
            min_margin:   (Tuỳ chọn) Ngưỡng chênh lệch score tối thiểu giữa
                          lớp nhất và lớp nhì trong vùng overlap.
            return_scores: Có trả về ma trận raw_scores không.

        Returns:
            (predictions, raw_scores_matrix | None)
        """
        if not self.models:
            raise RuntimeError("Không có model nào. Hãy train ít nhất một lớp.")

        class_names = list(self.models.keys())
        n_samples   = len(X_scaled)
        n_classes   = len(class_names)

        # Ma trận raw scores: shape (n_samples, n_classes)
        raw_scores = np.zeros((n_samples, n_classes))
        for i, cname in enumerate(class_names):
            raw_scores[:, i] = self.models[cname].decision_function(X_scaled)

        predictions = []
        for j in range(n_samples):
            x_j = X_scaled[j]

            # ── Bước 1: Lọc models có score > 0 ──────────────────────────────
            positive_mask = raw_scores[j] > 0
            n_positive    = int(np.sum(positive_mask))

            # ── Bước 2: Unknown ───────────────────────────────────────────────
            if n_positive == 0:
                predictions.append("unknown")
                continue

            pos_indices = np.where(positive_mask)[0]

            # ── Bước 3: Argmax trivial (1 model dương) ────────────────────────
            if n_positive == 1:
                predictions.append(class_names[pos_indices[0]])
                continue

            # ── Bước 4: Overlap → Euclidean Tie-break ────────────────────────
            # Tính d_min từ x đến SV gần nhất của từng lớp dương
            d_mins  = {}
            for ci in pos_indices:
                cname = class_names[ci]
                sv    = self.models[cname].memory_data   # SVs ở không gian đã scale toàn cục
                if sv is not None and len(sv) > 0:
                    distances   = np.linalg.norm(sv - x_j, axis=1)
                    d_mins[ci]  = float(np.min(distances))
                else:
                    # Không có SV → fallback về raw_score tốt nhất
                    d_mins[ci] = 1e9

            winner_ci = min(d_mins, key=d_mins.get)
            winner    = class_names[winner_ci]

            # ── Bước 5 (Tuỳ chọn): Margin gate ──────────────────────────────
            if min_margin > 0.0 and n_positive >= 2:
                pos_raw    = raw_scores[j][pos_indices]
                sorted_raw = np.sort(pos_raw)[::-1]
                margin     = float(sorted_raw[0] - sorted_raw[1])
                if margin < min_margin:
                    predictions.append(f"low_confidence/{winner}")
                    continue

            predictions.append(winner)

        if return_scores:
            return predictions, raw_scores
        return predictions, None

    def predict_with_confidence(
        self,
        X_scaled: np.ndarray,
        min_margin: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Dự đoán và trả về chi tiết scores cho từng lớp.

        Returns:
            List of dict:
                predicted_class, confidence, margin, is_low_confidence, all_scores
        """
        class_names  = list(self.models.keys())
        predictions, scores_matrix = self.predict_multi(
            X_scaled, return_scores=True, min_margin=min_margin
        )
        n_classes = len(class_names)
        results   = []

        for i, pred in enumerate(predictions):
            all_scores = {
                cn: float(scores_matrix[i, j])
                for j, cn in enumerate(class_names)
            }
            sorted_vals = sorted(all_scores.values(), reverse=True)
            best_score  = sorted_vals[0] if sorted_vals else 0.0
            margin      = float(sorted_vals[0] - sorted_vals[1]) if n_classes >= 2 else 0.0

            is_lc = pred.startswith("low_confidence/")
            results.append({
                "predicted_class"   : pred,
                "confidence"        : best_score if pred not in ("unknown",) and not is_lc else 0.0,
                "margin"            : margin,
                "is_low_confidence" : is_lc,
                "all_scores"        : all_scores,
            })
        return results

    # ------------------------------------------------------------------
    # Model Management
    # ------------------------------------------------------------------

    def load_class(self, class_name: str) -> None:
        """Load một model từ manifest vào session."""
        pkl_path = self.manifest.get_model_path(class_name)
        self.models[class_name] = IncrementalOCSVM.load(pkl_path)

    def load_all_from_manifest(self) -> None:
        """Load toàn bộ model trong manifest vào session."""
        for class_name in self.manifest.list_classes():
            try:
                self.load_class(class_name)
            except Exception as e:
                print(f"  [Load] ⚠ Không thể load '{class_name}': {e}")

    def delete_class(self, class_name: str, delete_file: bool = False) -> None:
        """Xoá một lớp khỏi session (và tuỳ chọn xoá file .pkl)."""
        if class_name not in self.models:
            raise ValueError(f"Lớp '{class_name}' không tồn tại trong session.")
        if delete_file:
            pkl_path = os.path.join(
                self.model_dir, f"{self.models[class_name].version_name}.pkl"
            )
            if os.path.exists(pkl_path):
                os.remove(pkl_path)
                print(f"  [Delete] Đã xoá: {pkl_path}")
        del self.models[class_name]
        self.manifest.remove_class(class_name)

    def list_classes(self) -> List[str]:
        return list(self.models.keys())

    def get_all_info(self) -> Dict[str, Any]:
        return {
            "n_classes": len(self.models),
            "classes"  : [m.get_info() for m in self.models.values()],
        }

    def __repr__(self):
        return (
            f"MultiClassOCSVM("
            f"n_classes={len(self.models)}, "
            f"classes={self.list_classes()})"
        )
