"""
Tests for mOC-iSVM Algorithm Improvements
==========================================
Bao gồm 7 test cases cho 4 khu vực cải tiến:
  1. Modified Logic (min_margin, unknown gate)
  2. Incremental Memory (SV-only retrain)
  3. Data Integrity (dropna, shape mismatch)
  4. Atomic Manifest (pkl_save_success guard, performance metrics in XML)
"""

import io
import os
import sys
import tempfile
import textwrap
import numpy as np
import pytest

# Đảm bảo import từ thư mục gốc của project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from mocsvm.core.incremental import IncrementalOCSVM
from mocsvm.core.multiclass import MultiClassOCSVM
from mocsvm.core.manifest_manager import ManifestManager
from mocsvm.utils.data_loader import load_and_validate_csv, validate_X_y_shape


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_data():
    """Dữ liệu hai cụm đơn giản cho test."""
    rng = np.random.default_rng(42)
    cls_a = rng.normal(loc=[0, 0], scale=0.3, size=(30, 2))
    cls_b = rng.normal(loc=[5, 5], scale=0.3, size=(30, 2))
    return cls_a, cls_b


@pytest.fixture
def trained_mc(simple_data, tmp_path):
    """MultiClassOCSVM với hai lớp đã được train."""
    cls_a, cls_b = simple_data
    manifest_path = str(tmp_path / "manifest.xml")
    mc = MultiClassOCSVM(model_dir=str(tmp_path), manifest_path=manifest_path)
    mc.train_class("class_a", cls_a, nu=0.05)
    mc.train_class("class_b", cls_b, nu=0.05)
    return mc


# ---------------------------------------------------------------------------
# 1. Modified Logic – Unknown Gate
# ---------------------------------------------------------------------------

def test_unknown_when_all_scores_negative(trained_mc):
    """
    Nếu tất cả decision scores < 0 (điểm nằm ngoài biên của mọi lớp),
    predict_multi phải trả về 'unknown'.
    """
    # Điểm nằm ở vị trí hoàn toàn xa lạ
    outlier = np.array([[50.0, -50.0]])
    preds, _ = trained_mc.predict_multi(outlier)
    assert preds[0] == "unknown", (
        f"Dự kiến 'unknown' nhưng nhận được '{preds[0]}'"
    )


# ---------------------------------------------------------------------------
# 2. Modified Logic – Margin Gate (low_confidence)
# ---------------------------------------------------------------------------

def test_low_confidence_label_when_margin_too_small(simple_data, tmp_path):
    """
    Với min_margin cực cao (lớn hơn bất kỳ margin thực nào có thể đạt được),
    mọi điểm trong-phân phối phải cho nhãn 'low_confidence/...'
    (vì norm_score nằm trong [−1, 1] nên margin max = 2.0).
    """
    cls_a, cls_b = simple_data
    manifest_path = str(tmp_path / "manifest.xml")
    mc = MultiClassOCSVM(model_dir=str(tmp_path), manifest_path=manifest_path)
    mc.train_class("class_a", cls_a, nu=0.05)
    mc.train_class("class_b", cls_b, nu=0.05)

    # Điểm rõ ràng thuộc class_a
    point_a = np.array([[0.0, 0.0]])

    # min_margin=10.0 không thể đạt được (norm scores trong [-1, 1], max margin = 2)
    preds, _ = mc.predict_multi(point_a, min_margin=10.0)
    label = preds[0]

    assert label == "unknown" or label.startswith("low_confidence/"), (
        f"Với min_margin=10.0 (không thể đạt được), dự kiến 'unknown' hoặc "
        f"'low_confidence/...' nhưng nhận được '{label}'"
    )


def test_no_low_confidence_when_margin_is_zero(trained_mc, simple_data):
    """
    min_margin=0.0 (mặc định) không được tạo ra nhãn low_confidence.
    """
    cls_a, _ = simple_data
    # Điểm rõ ràng thuộc lớp A
    clear_a = np.array([[0.0, 0.0]])
    preds, _ = trained_mc.predict_multi(clear_a, min_margin=0.0)
    assert not preds[0].startswith("low_confidence/"), (
        f"min_margin=0.0 không nên tạo low_confidence, nhận được '{preds[0]}'"
    )


# ---------------------------------------------------------------------------
# 3. Incremental Memory – SV-only retrain
# ---------------------------------------------------------------------------

def test_sv_retrain_does_not_grow_unboundedly(simple_data, tmp_path):
    """
    Sau mỗi lần retrain, memory_data gộp SV+new chứ không phải full_data+new.
    Kích thước memory_data phải nhỏ hơn số mẫu tích lũy thực tế.
    """
    cls_a, _ = simple_data
    model = IncrementalOCSVM("test_cls", nu=0.1, model_dir=str(tmp_path))
    model.train(cls_a[:20], version_name="test_cls-01", save_model=False)

    sv_after_train = len(model.memory_data)

    # Retrain với 20 mẫu mới
    extra = cls_a[10:30]  # 20 mẫu
    model.retrain(extra, new_version="test_cls-02", save_model=False)

    sv_after_retrain = len(model.memory_data)
    total_if_naive   = 20 + 20  # naive stack: data cũ + data mới

    # Support Vectors phải ít hơn tổng tích lũy thuần túy
    assert sv_after_retrain < total_if_naive, (
        f"SV sau retrain ({sv_after_retrain}) nên < tổng tích lũy ({total_if_naive}). "
        "Cơ chế SV-only chưa hoạt động đúng."
    )
    print(f"  SVs sau train: {sv_after_train}, SVs sau retrain: {sv_after_retrain} / {total_if_naive} mẫu cộng gộp thủ công")


def test_get_info_reports_n_support_vectors(simple_data, tmp_path):
    """get_info() phải chứa 'n_support_vectors'."""
    cls_a, _ = simple_data
    model = IncrementalOCSVM("test_cls", nu=0.1, model_dir=str(tmp_path))
    model.train(cls_a, version_name="test_cls-01", save_model=False)
    info = model.get_info()
    assert "n_support_vectors" in info, "get_info() thiếu trường 'n_support_vectors'"
    assert info["n_support_vectors"] > 0, "n_support_vectors phải > 0 sau khi train"


# ---------------------------------------------------------------------------
# 4. Data Integrity – dropna & shape mismatch
# ---------------------------------------------------------------------------

def test_dropna_handles_blank_rows(tmp_path):
    """
    load_and_validate_csv phải xử lý được dòng trống cuối file.
    """
    samples_csv = tmp_path / "samples.csv"
    features_csv = tmp_path / "features.csv"
    classes_csv = tmp_path / "classes.csv"

    # 3 dòng dữ liệu + 1 dòng trống cuối
    samples_csv.write_text("id\n1\n2\n3\n\n", encoding="utf-8")
    features_csv.write_text("f1,f2\n1.0,2.0\n3.0,4.0\n5.0,6.0\n\n", encoding="utf-8")
    classes_csv.write_text("class\nA\nB\nA\n\n", encoding="utf-8")

    X, feat_names, labels = load_and_validate_csv(
        str(samples_csv), str(features_csv), str(classes_csv)
    )
    assert len(X) == 3, f"Mong đợi 3 mẫu sau dropna, nhận được {len(X)}"
    assert len(labels) == 3


def test_shape_mismatch_raises_value_error(tmp_path):
    """
    samples.csv có 3 dòng nhưng classes.csv có 2 dòng → ValueError.
    """
    samples_csv = tmp_path / "samples.csv"
    features_csv = tmp_path / "features.csv"
    classes_csv = tmp_path / "classes.csv"

    samples_csv.write_text("id\n1\n2\n3", encoding="utf-8")
    features_csv.write_text("f1,f2\n1.0,2.0\n3.0,4.0\n5.0,6.0", encoding="utf-8")
    classes_csv.write_text("class\nA\nB", encoding="utf-8")  # chỉ 2 dòng

    with pytest.raises(ValueError, match="Số dòng không khớp"):
        load_and_validate_csv(
            str(samples_csv), str(features_csv), str(classes_csv)
        )


def test_validate_X_y_shape_raises_on_mismatch():
    """validate_X_y_shape phải raise ValueError nếu n_X != n_y."""
    X = np.zeros((10, 4))
    y = ["A"] * 9  # 9 ≠ 10
    with pytest.raises(ValueError, match="Shape không khớp"):
        validate_X_y_shape(X, y, context="test_class")


# ---------------------------------------------------------------------------
# 5. Atomic Manifest – pkl_save_success guard
# ---------------------------------------------------------------------------

def test_atomic_manifest_not_updated_when_pkl_missing(tmp_path):
    """
    Khi pkl_save_success=False, manifest.update_class() không được ghi XML.
    """
    manifest_path = str(tmp_path / "manifest.xml")
    mgr = ManifestManager(manifest_path)

    # Gọi update_class với pkl_save_success=False
    mgr.update_class(
        class_name       = "test_cls",
        version_name     = "test_cls-01",
        pkl_path         = "/non/existent/path.pkl",
        pkl_save_success = False,
    )

    # Lớp không được đăng ký
    classes = mgr.list_classes()
    assert "test_cls" not in classes, (
        "Manifest không được cập nhật khi pkl_save_success=False (atomic guard failed)"
    )


def test_performance_metrics_stored_in_xml(simple_data, tmp_path):
    """
    Sau khi train, XML phải chứa <performance> với inlier_ratio và n_support_vectors.
    """
    cls_a, _ = simple_data
    manifest_path = str(tmp_path / "manifest.xml")
    mc = MultiClassOCSVM(model_dir=str(tmp_path), manifest_path=manifest_path)
    mc.train_class("class_a", cls_a, nu=0.05)

    info = mc.manifest.get_class_info("class_a")
    perf = info.get("performance", {})

    assert "inlier_ratio" in perf, (
        f"XML thiếu <inlier_ratio> trong <performance>. Nhận được: {perf}"
    )
    assert "n_support_vectors" in perf, (
        f"XML thiếu <n_support_vectors> trong <performance>. Nhận được: {perf}"
    )
    # Giá trị phải hợp lệ
    assert 0.0 <= float(perf["inlier_ratio"]) <= 1.0, (
        f"inlier_ratio ngoài khoảng [0, 1]: {perf['inlier_ratio']}"
    )
    assert int(perf["n_support_vectors"]) > 0
