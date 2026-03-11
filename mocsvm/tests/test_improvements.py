"""
Tests for mOC-iSVM Algorithm – New Architecture (Global Scaler + SV Pruning)
==============================================================================
Kiến trúc mới:
  - GlobalScalerManager: fit 1 lần trên toàn bộ X_train, dùng transform() sau đó.
  - IncrementalOCSVM: không có per-class scaler; nhận data đã scale từ GlobalScaler.
  - SV Pruning: Age Pruning + Error Pruning trong retrain().
  - MultiClassOCSVM.predict_multi(): Euclidean Nearest-SV Tie-break khi overlap.
"""

import io
import os
import sys
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from mocsvm.core.incremental import IncrementalOCSVM
from mocsvm.core.multiclass import MultiClassOCSVM
from mocsvm.core.manifest_manager import ManifestManager
from mocsvm.utils.global_scaler import GlobalScalerManager
from mocsvm.utils.data_loader import load_and_validate_csv, validate_X_y_shape


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_data():
    """Hai cụm đơn giản trong không gian 2D."""
    rng = np.random.default_rng(42)
    cls_a = rng.normal(loc=[0, 0], scale=0.3, size=(30, 2))
    cls_b = rng.normal(loc=[5, 5], scale=0.3, size=(30, 2))
    return cls_a, cls_b


@pytest.fixture
def global_scaler_and_data(simple_data, tmp_path):
    """GlobalScaler đã fit + dữ liệu đã scale."""
    cls_a, cls_b = simple_data
    X_all = np.vstack([cls_a, cls_b])
    gsm   = GlobalScalerManager(model_dir=str(tmp_path))
    gsm.fit_and_save(X_all)
    cls_a_scaled = gsm.transform(cls_a)
    cls_b_scaled = gsm.transform(cls_b)
    return gsm, cls_a_scaled, cls_b_scaled


@pytest.fixture
def trained_mc(global_scaler_and_data, tmp_path):
    """MultiClassOCSVM với hai lớp đã train trên global scale space."""
    gsm, cls_a_scaled, cls_b_scaled = global_scaler_and_data
    manifest_path = str(tmp_path / "manifest.xml")
    mc = MultiClassOCSVM(model_dir=str(tmp_path), manifest_path=manifest_path)
    mc.train_class("class_a", cls_a_scaled, nu=0.05)
    mc.train_class("class_b", cls_b_scaled, nu=0.05)
    return mc, gsm


# ---------------------------------------------------------------------------
# 1. GlobalScaler – Kiểm tra tính nhất quán
# ---------------------------------------------------------------------------

def test_global_scaler_fit_and_transform(simple_data, tmp_path):
    """
    GlobalScaler phải fit trên toàn bộ X và transform nhất quán.
    Sau transform, mean ≈ 0 và std ≈ 1.
    """
    cls_a, cls_b = simple_data
    X_all = np.vstack([cls_a, cls_b])
    gsm   = GlobalScalerManager(model_dir=str(tmp_path))
    gsm.fit_and_save(X_all)
    X_scaled = gsm.transform(X_all)
    # Kiểm tra mean ≈ 0 và std ≈ 1 trên từng feature
    assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-6), "Mean sau scale phải ≈ 0"
    assert np.allclose(X_scaled.std(axis=0),  1, atol=1e-2), "Std sau scale phải ≈ 1"


def test_global_scaler_save_and_load(simple_data, tmp_path):
    """GlobalScaler lưu và load phải cho kết quả transform giống nhau."""
    cls_a, cls_b = simple_data
    X_all = np.vstack([cls_a, cls_b])
    gsm1  = GlobalScalerManager(model_dir=str(tmp_path))
    gsm1.fit_and_save(X_all)
    X1    = gsm1.transform(cls_a)

    gsm2  = GlobalScalerManager(model_dir=str(tmp_path))
    gsm2.load()
    X2    = gsm2.transform(cls_a)

    np.testing.assert_array_almost_equal(X1, X2, decimal=8,
        err_msg="Scale sau load không khớp scale ban đầu – hệ quy chiếu bị thay đổi!")


def test_global_scaler_feature_mismatch_raises(simple_data, tmp_path):
    """GlobalScaler phải raise ValueError nếu số features không khớp."""
    cls_a, _ = simple_data
    X_3d = np.hstack([cls_a, cls_a[:, :1]])   # 3 features
    gsm  = GlobalScalerManager(model_dir=str(tmp_path))
    gsm.fit_and_save(cls_a)   # fit với 2 features
    with pytest.raises(ValueError, match="features"):
        gsm.transform(X_3d)


# ---------------------------------------------------------------------------
# 2. Unknown Gate – Điểm nằm ngoài mọi vỏ bọc → "unknown"
# ---------------------------------------------------------------------------

def test_unknown_when_all_scores_negative(trained_mc):
    """Nếu tất cả decision scores < 0 → predict_multi phải trả 'unknown'."""
    mc, gsm = trained_mc
    # Điểm rất xa trong không gian thô → transform qua gsm
    outlier_raw = np.array([[1000.0, -1000.0]])
    outlier_scaled = gsm.transform(outlier_raw)
    preds, _ = mc.predict_multi(outlier_scaled)
    assert preds[0] == "unknown", f"Dự kiến 'unknown' nhưng nhận '{preds[0]}'"


# ---------------------------------------------------------------------------
# 3. Euclidean Tie-break – Vùng Chồng chéo
# ---------------------------------------------------------------------------

def test_euclidean_tiebreak_assigns_nearest_sv_class(tmp_path):
    """
    Khi ≥ 2 model dương, Euclidean Nearest-SV phải chọn đúng lớp (Overlap).
    Sử dụng mock model để test riêng biệt logic tie-break.
    """
    mc = MultiClassOCSVM(model_dir=str(tmp_path), manifest_path=str(tmp_path / "m.xml"))

    # Fake model A với phần lớn SV ở gần [0,0]
    mA = IncrementalOCSVM("class_a", model_dir=str(tmp_path))
    mA.is_trained = True
    mA.memory_data = np.array([[0.0, 0.0]])
    mA.decision_function = lambda x: np.array([1.0] * len(x))  # Luôn dương

    # Fake model B với phần lớn SV ở gần [10,10]
    mB = IncrementalOCSVM("class_b", model_dir=str(tmp_path))
    mB.is_trained = True
    mB.memory_data = np.array([[10.0, 10.0]])
    mB.decision_function = lambda x: np.array([1.0] * len(x))  # Luôn dương

    mc.models = {"class_a": mA, "class_b": mB}

    # Điểm test [1, 1] -> khoảng cách tới A (~1.41) < tới B (~12.7) -> Chọn A
    test_pt_a = np.array([[1.0, 1.0]])
    preds_a, _ = mc.predict_multi(test_pt_a)
    assert preds_a[0] == "class_a"

    # Điểm test [9, 9] -> khoảng cách tới A (~12.7) > tới B (~1.41) -> Chọn B
    test_pt_b = np.array([[9.0, 9.0]])
    preds_b, _ = mc.predict_multi(test_pt_b)
    assert preds_b[0] == "class_b"


# ---------------------------------------------------------------------------
# 4. SV Pruning – Age Pruning
# ---------------------------------------------------------------------------

def test_age_pruning_removes_old_svs(simple_data, tmp_path):
    """
    Sau nhiều lần retrain, Age Pruning phải loại bỏ SVs già.
    Memory_data sau retrain phải nhỏ hơn tổng tích lũy thuần túy.
    """
    cls_a, _ = simple_data
    gsm = GlobalScalerManager(model_dir=str(tmp_path))
    gsm.fit_and_save(cls_a)
    cls_a_scaled = gsm.transform(cls_a)

    model = IncrementalOCSVM("test_cls", nu=0.1, model_dir=str(tmp_path))
    model.train(cls_a_scaled[:20], version_name="test_cls-01", save_model=False)

    sv_after_train = len(model.memory_data)

    # Retrain với 20 mẫu mới, age_threshold thấp để pruning diễn ra
    extra_scaled   = cls_a_scaled[10:30]
    model.retrain(extra_scaled, new_version="test_cls-02", save_model=False, age_threshold=2)

    sv_after_retrain = len(model.memory_data)
    total_if_naive   = 20 + 20

    assert sv_after_retrain < total_if_naive, (
        f"SV sau retrain ({sv_after_retrain}) nên < tổng tích lũy ({total_if_naive}). "
        "SV Pruning chưa hoạt động đúng."
    )
    print(f"  SVs: train={sv_after_train} → retrain={sv_after_retrain} (naive={total_if_naive})")


# ---------------------------------------------------------------------------
# 5. SV Pruning – Error Pruning
# ---------------------------------------------------------------------------

def test_error_pruning_discards_all_svs_on_concept_drift(tmp_path):
    """
    Nếu accuracy < error_threshold, error pruning phải hủy toàn bộ SVs cũ.
    X_new là dữ liệu hoàn toàn khác với X_train → accuracy thấp.
    """
    rng = np.random.default_rng(99)
    X_train = rng.normal([0, 0], 0.3, (30, 2))
    X_drift = rng.normal([10, 10], 0.2, (20, 2))   # Concept Drift
    X_all = np.vstack([X_train, X_drift])

    gsm = GlobalScalerManager(model_dir=str(tmp_path))
    gsm.fit_and_save(X_all)
    X_train_scaled = gsm.transform(X_train)
    X_drift_scaled = gsm.transform(X_drift)

    model = IncrementalOCSVM("drift_cls", nu=0.05, model_dir=str(tmp_path))
    model.train(X_train_scaled, version_name="drift_cls-01", save_model=False)
    
    sv_before = len(model.memory_data)

    # Retrain với dữ liệu drift – accuracy cũ trên X_drift sẽ rất thấp
    model.retrain(
        X_drift_scaled, new_version="drift_cls-02",
        save_model=False, error_threshold=0.99  # Ngưỡng cao → kích hoạt error pruning
    )

    # Sau error pruning + refit, memory_data phải chỉ dựa trên X_drift
    # (không còn SVs cũ từ X_train)
    assert len(model.memory_data) <= len(X_drift_scaled), (
        f"Memory sau error pruning ({len(model.memory_data)}) "
        f"không nên lớn hơn X_drift ({len(X_drift_scaled)})"
    )
    print(f"  SVs trước error pruning: {sv_before} → sau: {len(model.memory_data)}")


# ---------------------------------------------------------------------------
# 6. get_info – n_support_vectors
# ---------------------------------------------------------------------------

def test_get_info_reports_n_support_vectors(simple_data, tmp_path):
    """get_info() phải chứa 'n_support_vectors' > 0 sau khi train."""
    cls_a, _ = simple_data
    gsm = GlobalScalerManager(model_dir=str(tmp_path))
    gsm.fit_and_save(cls_a)
    cls_a_scaled = gsm.transform(cls_a)
    
    model = IncrementalOCSVM("test", nu=0.1, model_dir=str(tmp_path))
    model.train(cls_a_scaled, version_name="test-01", save_model=False)
    info = model.get_info()
    assert "n_support_vectors" in info
    assert info["n_support_vectors"] > 0


# ---------------------------------------------------------------------------
# 7. Data Integrity – dropna & shape mismatch
# ---------------------------------------------------------------------------

def test_dropna_handles_blank_rows(tmp_path):
    """load_and_validate_csv phải xử lý dòng trống cuối file."""
    s_csv = tmp_path / "samples.csv"
    f_csv = tmp_path / "features.csv"
    c_csv = tmp_path / "classes.csv"
    s_csv.write_text("id\n1\n2\n3\n\n", encoding="utf-8")
    f_csv.write_text("f1,f2\n1.0,2.0\n3.0,4.0\n5.0,6.0\n\n", encoding="utf-8")
    c_csv.write_text("class\nA\nB\nA\n\n", encoding="utf-8")
    X, feat_names, labels = load_and_validate_csv(str(s_csv), str(f_csv), str(c_csv))
    assert len(X) == 3, f"Mong đợi 3 mẫu, nhận {len(X)}"


def test_shape_mismatch_raises_value_error(tmp_path):
    """Samples 3 dòng nhưng classes 2 dòng → ValueError."""
    s_csv = tmp_path / "samples.csv"
    f_csv = tmp_path / "features.csv"
    c_csv = tmp_path / "classes.csv"
    s_csv.write_text("id\n1\n2\n3", encoding="utf-8")
    f_csv.write_text("f1,f2\n1.0,2.0\n3.0,4.0\n5.0,6.0", encoding="utf-8")
    c_csv.write_text("class\nA\nB", encoding="utf-8")
    with pytest.raises(ValueError, match="Số dòng không khớp"):
        load_and_validate_csv(str(s_csv), str(f_csv), str(c_csv))


def test_validate_X_y_shape_raises_on_mismatch():
    """validate_X_y_shape phải raise ValueError nếu n_X != n_y."""
    X = np.zeros((10, 4))
    y = ["A"] * 9
    with pytest.raises(ValueError, match="Shape không khớp"):
        validate_X_y_shape(X, y, context="test_class")


# ---------------------------------------------------------------------------
# 8. Atomic Manifest – pkl_save_success guard
# ---------------------------------------------------------------------------

def test_atomic_manifest_not_updated_when_pkl_missing(tmp_path):
    """Khi pkl_save_success=False, manifest không được ghi XML."""
    mgr = ManifestManager(str(tmp_path / "manifest.xml"))
    mgr.update_class(
        class_name       = "test_cls",
        version_name     = "test_cls-01",
        pkl_path         = "/non/existent/path.pkl",
        pkl_save_success = False,
    )
    assert "test_cls" not in mgr.list_classes(), "Manifest không được cập nhật khi pkl_save_success=False"
