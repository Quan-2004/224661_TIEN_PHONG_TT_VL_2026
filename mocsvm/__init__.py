"""
mOC-iSVM – Modified One-Class Incremental SVM
==============================================
Thư viện học máy hỗ trợ One-Class SVM đa lớp với khả năng học tăng cường
(Incremental Learning) và quản lý phiên bản tự động.

Cách dùng nhanh:
    from mocsvm.core.multiclass import MultiClassOCSVM
    import numpy as np

    mc = MultiClassOCSVM()
    mc.train_class("cat", np.random.randn(100, 4))
    mc.train_class("dog", np.random.randn(80, 4))
    predictions, _ = mc.predict_multi(np.random.randn(10, 4))
"""

from mocsvm.core.incremental import IncrementalOCSVM
from mocsvm.core.multiclass import MultiClassOCSVM
from mocsvm.core.manifest_manager import ManifestManager

__version__ = "1.0.0"
__all__ = ["IncrementalOCSVM", "MultiClassOCSVM", "ManifestManager"]
