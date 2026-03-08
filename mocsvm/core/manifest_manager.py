"""
Manifest Manager – Quản lý Registry XML
========================================
Module quản lý file global_manifest.xml, đóng vai trò là Registry trung tâm
cho toàn bộ hệ thống mOC-iSVM.

Cấu trúc XML:
    <manifest updated="...">
        <model class_name="successful" version="successful-02">
            <pkl_path>models/successful-02.pkl</pkl_path>
            <metadata>
                <kernel>rbf</kernel>
                <gamma>scale</gamma>
                <nu>0.1</nu>
                <n_samples>150</n_samples>
                <trained_at>2025-01-01T12:00:00</trained_at>
            </metadata>
        </model>
        ...
    </manifest>
"""

import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import xml.etree.ElementTree as ET


class ManifestManager:
    """
    Quản lý file global_manifest.xml – Registry lưu trữ thông tin
    tất cả các model trong hệ thống.

    Attributes:
        manifest_path (str): Đường dẫn đến file XML.
        tree (ET.ElementTree): Cây XML đang hoạt động.
        root (ET.Element): Phần tử gốc của XML.
    """

    def __init__(self, manifest_path: str = "models/global_manifest.xml"):
        self.manifest_path = manifest_path
        self._ensure_file_exists()
        self._load()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_file_exists(self) -> None:
        """Tạo file XML mới nếu chưa tồn tại."""
        os.makedirs(os.path.dirname(self.manifest_path), exist_ok=True)
        if not os.path.exists(self.manifest_path):
            root = ET.Element("manifest")
            root.set("updated", datetime.now().isoformat())
            tree = ET.ElementTree(root)
            self._indent(root)
            tree.write(self.manifest_path, encoding="utf-8", xml_declaration=True)
            print(f"  [Manifest] ✓ Tạo file mới: {self.manifest_path}")

    def _load(self) -> None:
        """Load cây XML vào bộ nhớ."""
        self.tree = ET.parse(self.manifest_path)
        self.root = self.tree.getroot()

    def _save(self) -> None:
        """Lưu cây XML ra file."""
        self.root.set("updated", datetime.now().isoformat())
        self._indent(self.root)
        self.tree.write(self.manifest_path, encoding="utf-8", xml_declaration=True)

    @staticmethod
    def _indent(elem: ET.Element, level: int = 0) -> None:
        """Thêm indent để XML dễ đọc (pretty-print)."""
        indent = "\n" + "    " * level
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = indent + "    "
            if not elem.tail or not elem.tail.strip():
                elem.tail = indent
            for child in elem:
                ManifestManager._indent(child, level + 1)
            # Sửa lại tail của phần tử cuối
            if not child.tail or not child.tail.strip():  # type: ignore[reportPossiblyUnbound]
                child.tail = indent  # type: ignore[reportPossiblyUnbound]
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = indent

    def _find_model(self, class_name: str) -> Optional[ET.Element]:
        """Tìm phần tử <model> theo class_name."""
        for model_el in self.root.findall("model"):
            if model_el.get("class_name") == class_name:
                return model_el
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_class(
        self,
        class_name: str,
        version_name: str,
        pkl_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        performance: Optional[Dict[str, Any]] = None,
        pkl_save_success: bool = True,
    ) -> None:
        """
        Thêm mới hoặc cập nhật thông tin một lớp trong manifest.

        Đảm bảo tính nguyên tử (Atomic): Manifest chỉ được ghi khi
        pkl_save_success=True (tức là file .pkl đã được lưu thành công).

        Args:
            class_name:       Tên lớp (ví dụ 'successful').
            version_name:     Phiên bản mới nhất (ví dụ 'successful-02').
            pkl_path:         Đường dẫn file .pkl.
            metadata:         Dict các tham số mô hình (kernel, gamma, nu, ...).
            performance:      Dict các chỉ số hiệu suất (inlier_ratio, n_support_vectors, ...).
            pkl_save_success: Nếu False, bỏ qua việc ghi XML (atomic guard).
        """
        if not pkl_save_success:
            print(
                f"  [Manifest] Cảnh báo: Bỏ qua cập nhật XML cho '{class_name}' "
                "vì .pkl chưa được lưu thành công (atomic guard)."
            )
            return

        model_el = self._find_model(class_name)

        if model_el is None:
            # Tạo phần tử mới
            model_el = ET.SubElement(self.root, "model")
            model_el.set("class_name", class_name)

        # Cập nhật attributes
        model_el.set("version", version_name)

        # Cập nhật <pkl_path>
        pkl_el = model_el.find("pkl_path")
        if pkl_el is None:
            pkl_el = ET.SubElement(model_el, "pkl_path")
        pkl_el.text = pkl_path

        # Cập nhật <metadata> (tham số mô hình)
        if metadata:
            meta_el = model_el.find("metadata")
            if meta_el is None:
                meta_el = ET.SubElement(model_el, "metadata")
            for key, value in metadata.items():
                child = meta_el.find(key)
                if child is None:
                    child = ET.SubElement(meta_el, key)
                child.text = str(value)

        # Cập nhật <performance> (chỉ số hiệu suất – hỗ trợ frontend vẽ biểu đồ)
        if performance:
            perf_el = model_el.find("performance")
            if perf_el is None:
                perf_el = ET.SubElement(model_el, "performance")
            for key, value in performance.items():
                child = perf_el.find(key)
                if child is None:
                    child = ET.SubElement(perf_el, key)
                child.text = str(value)

        self._save()
        print(f"  [Manifest] ✓ Cập nhật lớp '{class_name}' → version '{version_name}'")

    def get_model_path(self, class_name: str) -> str:
        """
        Lấy đường dẫn file .pkl của lớp.

        Raises:
            KeyError: Nếu lớp không tồn tại trong manifest.
        """
        model_el = self._find_model(class_name)
        if model_el is None:
            raise KeyError(f"Lớp '{class_name}' không tìm thấy trong manifest.")
        pkl_el = model_el.find("pkl_path")
        if pkl_el is None or not pkl_el.text:
            raise KeyError(f"Lớp '{class_name}' không có pkl_path hợp lệ.")
        return pkl_el.text

    def list_classes(self) -> List[str]:
        """Trả về danh sách tên tất cả các lớp trong manifest."""
        return [el.get("class_name", "") for el in self.root.findall("model")]

    def get_class_info(self, class_name: str) -> Dict[str, Any]:
        """
        Lấy thông tin đầy đủ của một lớp từ manifest.

        Returns:
            Dict chứa class_name, version, pkl_path, metadata, performance.

        Raises:
            KeyError: Nếu lớp không tồn tại.
        """
        model_el = self._find_model(class_name)
        if model_el is None:
            raise KeyError(f"Lớp '{class_name}' không tìm thấy trong manifest.")

        info: Dict[str, Any] = {
            "class_name"  : model_el.get("class_name"),
            "version"     : model_el.get("version"),
            "pkl_path"    : getattr(model_el.find("pkl_path"), "text", None),
            "metadata"    : {},
            "performance" : {},
        }

        meta_el = model_el.find("metadata")
        if meta_el is not None:
            for child in meta_el:
                info["metadata"][child.tag] = child.text

        perf_el = model_el.find("performance")
        if perf_el is not None:
            for child in perf_el:
                # Chuyển sang float nếu có thể (để frontend vẽ biểu đồ thuận tiện hơn)
                try:
                    info["performance"][child.tag] = float(child.text)  # type: ignore[arg-type]
                except (ValueError, TypeError):
                    info["performance"][child.tag] = child.text

        return info

    def list_all(self) -> List[Dict[str, Any]]:
        """Trả về danh sách thông tin tất cả các lớp."""
        return [self.get_class_info(cn) for cn in self.list_classes()]

    def remove_class(self, class_name: str) -> None:
        """
        Xoá một lớp khỏi manifest.

        Raises:
            KeyError: Nếu lớp không tồn tại.
        """
        model_el = self._find_model(class_name)
        if model_el is None:
            raise KeyError(f"Lớp '{class_name}' không tìm thấy trong manifest.")
        self.root.remove(model_el)
        self._save()
        print(f"  [Manifest] ✓ Đã xoá lớp '{class_name}' khỏi manifest.")

    def get_last_updated(self) -> str:
        """Trả về timestamp lần cập nhật cuối."""
        return self.root.get("updated", "unknown")

    def __repr__(self) -> str:
        classes = self.list_classes()
        return f"ManifestManager(path='{self.manifest_path}', classes={classes})"
