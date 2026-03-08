"""
Multi-Class One-Class SVM Manager
===================================
Module điều phối nhiều model IncrementalOCSVM (phương pháp One-vs-Rest).

Chiến lược dự đoán:
    1. Đưa dữ liệu qua tất cả các model.
    2. Chọn nhãn của model có decision_function cao nhất.
    3. Nếu tất cả decision scores < 0 → trả về "unknown".
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from mocsvm.core.incremental import IncrementalOCSVM
from mocsvm.core.manifest_manager import ManifestManager


class MultiClassOCSVM:
    """
    Quản lý tập hợp các IncrementalOCSVM, mỗi model ứng với một lớp.

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
        X: np.ndarray,
        version_name: Optional[str] = None,
        nu: float = 0.1,
        gamma: str = "scale",
        kernel: str = "rbf",
    ) -> Dict[str, Any]:
        """
        Huấn luyện model lần đầu cho một lớp.

        Nếu lớp đã tồn tại trong session, ghi đè.

        Args:
            class_name:   Tên lớp.
            X:            Dữ liệu huấn luyện.
            version_name: Phiên bản (ví dụ 'successful-01'). Nếu None tự tạo.
            nu, gamma, kernel: Tham số OC-SVM.

        Returns:
            Dict metadata quá trình huấn luyện.
        """
        vname = version_name or f"{class_name}-01"

        model = IncrementalOCSVM(
            class_name = class_name,
            nu         = nu,
            gamma      = gamma,
            kernel     = kernel,
            model_dir  = self.model_dir,
        )
        info = model.train(X, version_name=vname, save_model=True)
        self.models[class_name] = model

        # --- Đảm bảo Atomic: chỉ cập nhật XML sau khi .pkl đã được lưu thành công ---
        pkl_path = os.path.join(self.model_dir, f"{vname}.pkl")
        pkl_ok   = os.path.exists(pkl_path)
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
            pkl_save_success = pkl_ok,
        )
        return info

    def retrain_class(
        self,
        class_name: str,
        X_new: np.ndarray,
        new_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Học tăng cường (retrain) cho một lớp đã có model.

        Args:
            class_name:   Tên lớp cần retrain.
            X_new:        Dữ liệu mới.
            new_version:  Tên phiên bản mới. Nếu None, tự động tăng.

        Raises:
            ValueError: Nếu lớp chưa được train hoặc không có trong manager.
        """
        if class_name not in self.models:
            # Thử load từ manifest
            try:
                self.load_class(class_name)
            except Exception:
                raise ValueError(
                    f"Lớp '{class_name}' chưa được train. Hãy gọi train_class() trước."
                )

        model = self.models[class_name]
        info  = model.retrain(X_new, new_version=new_version, save_model=True)

        # --- Đảm bảo Atomic: chỉ cập nhật XML sau khi .pkl đã được lưu thành công ---
        pkl_path = os.path.join(self.model_dir, f"{model.version_name}.pkl")
        pkl_ok   = os.path.exists(pkl_path)
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
            pkl_save_success = pkl_ok,
        )
        return info

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_multi(
        self,
        X: np.ndarray,
        return_scores: bool = False,
        min_margin: float = 0.0,
    ) -> Tuple[List[str], Optional[np.ndarray]]:
        """
        Dự đoán nhãn cho mỗi mẫu bằng chiến lược One-vs-Rest.

        Thuật toán “Modified”:
            1. Tính normalized_decision_function của mọi model cho từng mẫu.
            2. Chọn nhãn có norm_score cao nhất.
            3. Nếu raw_score cao nhất < 0 → nhãn = "unknown"  (Unknown mechanism cốt lõi).
            4. Nếu biên độ giữa lớp nhất và lớp nhì (norm) < min_margin →
               nhãn = "low_confidence/<lớp_thắng>" (để tíđ tin cậy cần xét thêm).

        Args:
            X:             Dữ liệu đầu vào, shape (n_samples, n_features).
            return_scores: Nếu True, trả về thêm ma trận norm_scores.
            min_margin:    Biên độ tối thiểu giữa lớp nhất và lớp nhì (0.0 = tắt).

        Returns:
            (predictions, norm_scores_matrix | None)
        """
        if not self.models:
            raise RuntimeError("Không có model nào. Hãy train ít nhất một lớp.")

        class_names  = list(self.models.keys())
        n_samples    = len(X)
        n_classes    = len(class_names)

        # Ma trận scores: shape (n_samples, n_classes)
        raw_scores  = np.zeros((n_samples, n_classes))
        norm_scores = np.zeros((n_samples, n_classes))

        for i, cname in enumerate(class_names):
            model = self.models[cname]
            raw_scores[:, i]  = model.decision_function(X)
            # Hàm normalized có thể fallback về np.zeros nếu pkl cũ chưa có stats
            if hasattr(model, "normalized_decision_function"):
                norm_scores[:, i] = model.normalized_decision_function(X)
            else:
                norm_scores[:, i] = raw_scores[:, i]

        # Sắp xếp để lấy lớp nhất và lớp nhì
        sorted_indices = np.argsort(norm_scores, axis=1)[:, ::-1]  # giảm dần
        best_indices   = sorted_indices[:, 0]
        second_indices = sorted_indices[:, 1] if n_classes >= 2 else sorted_indices[:, 0]

        best_raws    = raw_scores[np.arange(n_samples),  best_indices]
        best_norms   = norm_scores[np.arange(n_samples), best_indices]
        second_norms = norm_scores[np.arange(n_samples), second_indices]

        # Hệ số kết hợp: raw_score chiếm phần lớn, nearest-neighbor tiebreak để phá tie
        ALPHA = 0.05  # trọng số khoảng cách NN

        predictions = []
        for j in range(n_samples):
            # ── Bước 1: Lọc các model có raw_score > 0 ─────────────────────
            positive_mask = raw_scores[j] > 0
            if not np.any(positive_mask):
                predictions.append("unknown")
                continue

            pos_indices = np.where(positive_mask)[0]
            pos_raws    = raw_scores[j][pos_indices]

            # ── Bước 2: Composite score = raw + alpha * (1 / nn_dist) ───────
            # Phá tie khi nhiều class cùng chấp nhận mẫu.
            # Model nào có dữ liệu train gần nhất với mẫu sẽ được ưu tiên.
            x_j          = X[j]
            best_score   = -np.inf
            winner       = class_names[pos_indices[np.argmax(pos_raws)]]

            for k, ci in enumerate(pos_indices):
                cname = class_names[ci]
                model = self.models[cname]
                raw   = float(pos_raws[k])

                # Lấy tập dữ liệu đại diện: memory_data (Support Vectors)
                ref = getattr(model, "memory_data", None)
                
                if ref is not None and len(ref) > 0:
                    nn_dist  = float(np.linalg.norm(ref - x_j, axis=1).min())
                    # Tránh chia cho 0; score_nn cao hơn khi NN gần hơn
                    score_nn = 1.0 / (nn_dist + 1e-6)
                else:
                    score_nn = 0.0

                composite = raw + ALPHA * score_nn
                if composite > best_score:
                    best_score = composite
                    winner     = cname

            # ── Bước 3: Margin gate (tuỳ chọn) ─────────────────────────────
            if min_margin > 0.0 and len(pos_indices) >= 2:
                sorted_pos = np.sort(pos_raws)[::-1]
                margin = float(sorted_pos[0] - sorted_pos[1])
                if margin < min_margin:
                    predictions.append(f"low_confidence/{winner}")
                    continue

            predictions.append(winner)

        if return_scores:
            return predictions, norm_scores
        return predictions, None

    def predict_with_confidence(self, X: np.ndarray, min_margin: float = 0.0) -> List[Dict[str, Any]]:
        """
        Dự đoán và trả về chi tiết scores cho từng lớp.

        Returns:
            List of dict: [
                {
                    "predicted_class": ...,
                    "confidence": ...,
                    "margin": ...,          # khoảng cách giữa lớp nhất và lớp nhì
                    "is_low_confidence": bool,
                    "all_scores": {...}
                }
            ]
        """
        class_names  = list(self.models.keys())
        predictions, scores_matrix = self.predict_multi(X, return_scores=True, min_margin=min_margin)

        n_classes = len(class_names)
        results = []
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
        """Load một model từ đường dẫn trong manifest và đưa vào session."""
        pkl_path = self.manifest.get_model_path(class_name)
        model    = IncrementalOCSVM.load(pkl_path)
        self.models[class_name] = model

    def load_all_from_manifest(self) -> None:
        """Load toàn bộ model được ghi trong manifest vào session."""
        for class_name in self.manifest.list_classes():
            try:
                self.load_class(class_name)
            except Exception as e:
                print(f"  [Load] Cảnh báo: Không thể load lớp '{class_name}': {e}")

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
                print(f"  [Delete] Đã xoá file: {pkl_path}")
        del self.models[class_name]
        self.manifest.remove_class(class_name)

    def list_classes(self) -> List[str]:
        """Danh sách tên các lớp đang active trong session."""
        return list(self.models.keys())

    def get_all_info(self) -> Dict[str, Any]:
        """Thông tin tổng quan toàn bộ hệ thống."""
        return {
            "n_classes": len(self.models),
            "classes"  : [m.get_info() for m in self.models.values()],
        }

    def __repr__(self) -> str:
        return (
            f"MultiClassOCSVM("
            f"n_classes={len(self.models)}, "
            f"classes={self.list_classes()})"
        )
