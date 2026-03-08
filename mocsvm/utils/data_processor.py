"""
mocsvm/utils/data_processor.py
==============================
PHASE 0 – Module Tiền Xử Lý CSV Thô
--------------------------------------
Nhận 1 file CSV thô (ví dụ: ks-projects-201612.csv) và xuất ra đúng 3 file
đồng bộ để đưa vào pipeline huấn luyện:

    samples.csv   – Các cột nhận dạng (ID, tên, ...)
    features.csv  – Vector số đã scale
    classes.csv   – Cột nhãn mục tiêu

YÊU CẦU NGHIÊM NGẶT (Strict Alignment):
    if not (len(samples) == len(features) == len(classes)):
        raise ValueError(...)
Chỉ khi 3 tập có số dòng KHỚP NHAU 100% mới được lưu file.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kết quả trả về
# ---------------------------------------------------------------------------
class ProcessingReport:
    """Tóm tắt kết quả sau khi xử lý."""

    def __init__(
        self,
        n_rows: int,
        n_features: int,
        unique_classes: List[str],
        class_counts: Dict[str, int],
        columns_used: List[str],
        columns_dropped: List[str],
        encoded_columns: List[str],
        scaled_columns: List[str],
        saved_paths: Dict[str, str],
    ):
        self.n_rows = n_rows
        self.n_features = n_features
        self.unique_classes = unique_classes
        self.class_counts = class_counts
        self.columns_used = columns_used
        self.columns_dropped = columns_dropped
        self.encoded_columns = encoded_columns
        self.scaled_columns = scaled_columns
        self.saved_paths = saved_paths
        self.aligned = True  # Luôn True nếu không raise

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rows"          : self.n_rows,
            "n_features"      : self.n_features,
            "unique_classes"  : self.unique_classes,
            "class_counts"    : self.class_counts,
            "columns_used"    : self.columns_used,
            "columns_dropped" : self.columns_dropped,
            "encoded_columns" : self.encoded_columns,
            "scaled_columns"  : self.scaled_columns,
            "saved_paths"     : self.saved_paths,
            "aligned"         : self.aligned,
        }


# ---------------------------------------------------------------------------
# DataProcessor
# ---------------------------------------------------------------------------
class DataProcessor:
    """
    Pipeline tiền xử lý dữ liệu CSV thô → 3 file đồng bộ.

    Parameters
    ----------
    class_column : str
        Tên cột chứa nhãn mục tiêu (bắt buộc).
    id_columns : list[str]
        Các cột dùng làm ID/tên mẫu (sẽ vào samples.csv, không vào features).
    drop_columns : list[str]
        Các cột bỏ qua hoàn toàn (không vào samples lẫn features).
    scale : bool
        Nếu True, áp dụng StandardScaler cho tất cả cột số trong features.
    fill_strategy : str
        Chiến lược điền NaN cho cột số: 'mean' | 'median' | 'zero'.
    encoding : str
        Mã hoá ký tự file đầu vào. Mặc định 'utf-8', thử 'latin-1' nếu lỗi.
    """

    def __init__(
        self,
        class_column: str,
        id_columns: Optional[List[str]] = None,
        drop_columns: Optional[List[str]] = None,
        scale: bool = True,
        fill_strategy: str = "mean",
        encoding: str = "utf-8",
    ):
        self.class_column  = class_column.strip()
        self.id_columns    = [c.strip() for c in (id_columns or [])]
        self.drop_columns  = [c.strip() for c in (drop_columns or [])]
        self.scale         = scale
        self.fill_strategy = fill_strategy
        self.encoding      = encoding

        self._label_encoders: Dict[str, LabelEncoder] = {}
        self._scaler: Optional[StandardScaler] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(
        self,
        csv_path: str,
        output_dir: str,
    ) -> ProcessingReport:
        """
        Đọc CSV thô, tiền xử lý, kiểm tra alignment nghiêm ngặt, lưu 3 file.

        Parameters
        ----------
        csv_path  : str – Đường dẫn tới file CSV thô
        output_dir: str – Thư mục lưu samples/features/classes.csv

        Returns
        -------
        ProcessingReport
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 1. Load -----------------------------------------------------------
        df = self._load(csv_path)
        logger.info(f"[DataProcessor] Đã load {len(df)} dòng, {len(df.columns)} cột.")

        # 2. Validate class column ------------------------------------------
        if self.class_column not in df.columns:
            raise ValueError(
                f"Cột class '{self.class_column}' không tồn tại trong file. "
                f"Các cột hiện có: {list(df.columns)}"
            )

        # 3. Drop unwanted columns ------------------------------------------
        to_drop = [c for c in self.drop_columns if c in df.columns]
        if to_drop:
            df = df.drop(columns=to_drop)
            logger.info(f"[DataProcessor] Bỏ cột: {to_drop}")

        # 3.5 Tự động drop cột Unnamed và cột toàn NaN ----------------------
        # Cột Unnamed thường xuất hiện khi xuất Excel (Unnamed: 13, ...)
        unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)
            logger.warning(f"[DataProcessor] Tự động bỏ {len(unnamed_cols)} cột Unnamed: {unnamed_cols}")

        # Drop cột mà toàn bộ giá trị là NaN (không có thông tin gì)
        all_nan_cols = [c for c in df.columns if df[c].isna().all()]
        if all_nan_cols:
            df = df.drop(columns=all_nan_cols)
            logger.warning(f"[DataProcessor] Tự động bỏ {len(all_nan_cols)} cột toàn NaN: {all_nan_cols}")

        # 4. Drop NaN theo hàng (chỉ xóa hàng thực sự thiếu dữ liệu) ------
        before = len(df)
        df = df.dropna().reset_index(drop=True)
        after = len(df)
        if before != after:
            logger.warning(f"[DataProcessor] Đã xóa {before - after} dòng NaN. Còn {after} dòng.")

        if len(df) == 0:
            raise ValueError("Không còn dòng nào sau khi xóa NaN!")

        # 5. Tách 3 phần (trước encode) ------------------------------------
        id_cols_available = [c for c in self.id_columns if c in df.columns]
        reserved          = set(id_cols_available + [self.class_column])
        feature_cols      = [c for c in df.columns if c not in reserved]

        # 6. Label Encode ---------------------------------------------------
        encoded_cols = []
        df_feat      = df[feature_cols].copy()

        for col in df_feat.columns:
            if df_feat[col].dtype == object or str(df_feat[col].dtype) == "category":
                le = LabelEncoder()
                df_feat[col]              = le.fit_transform(df_feat[col].astype(str))
                self._label_encoders[col] = le
                encoded_cols.append(col)

        # Encode class column giữ nguyên string (để manifest đọc được)
        class_series = df[self.class_column].astype(str)

        # 7. Scale ----------------------------------------------------------
        scaled_cols = []
        if self.scale and len(df_feat.columns) > 0:
            num_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                scaler = StandardScaler()
                df_feat[num_cols]  = scaler.fit_transform(df_feat[num_cols])
                self._scaler       = scaler
                scaled_cols        = num_cols
                logger.info(f"[DataProcessor] StandardScaler áp dụng cho {len(num_cols)} cột số.")

        # 8. Xây dựng 3 DataFrame ------------------------------------------
        # samples.csv – chứa id_columns (nếu có), hoặc chỉ index
        if id_cols_available:
            samples_df = df[id_cols_available].reset_index(drop=True)
        else:
            samples_df = pd.DataFrame({"sample_id": range(len(df))})

        features_df = df_feat.reset_index(drop=True)
        classes_df  = class_series.reset_index(drop=True).rename("class")

        # 9. STRICT ALIGNMENT CHECK ----------------------------------------
        n_s, n_f, n_c = len(samples_df), len(features_df), len(classes_df)
        self._log_alignment(n_s, n_f, n_c)

        if not (n_s == n_f == n_c):
            raise ValueError(
                f"ALIGNMENT FAIL! samples={n_s}, features={n_f}, classes={n_c}. "
                "Ba tập dữ liệu PHẢI có CÙNG SỐ DÒNG trước khi lưu."
            )

        logger.info(f"[DataProcessor] ✓ Alignment OK – {n_s} dòng đồng bộ.")

        # 10. Lưu 3 file ---------------------------------------------------
        out = Path(output_dir)
        samples_path  = str(out / "samples.csv")
        features_path = str(out / "features.csv")
        classes_path  = str(out / "classes.csv")

        samples_df.to_csv(samples_path,  index=False)
        features_df.to_csv(features_path, index=False)
        classes_df.to_csv(classes_path,  index=False, header=True)

        logger.info(f"[DataProcessor] Đã lưu: {samples_path}")
        logger.info(f"[DataProcessor] Đã lưu: {features_path}")
        logger.info(f"[DataProcessor] Đã lưu: {classes_path}")

        # 11. Thống kê class -----------------------------------------------
        class_counts = class_series.value_counts().to_dict()
        unique_cls   = sorted(class_counts.keys())

        return ProcessingReport(
            n_rows          = n_s,
            n_features      = len(features_df.columns),
            unique_classes  = unique_cls,
            class_counts    = {k: int(v) for k, v in class_counts.items()},
            columns_used    = list(feature_cols),
            columns_dropped = to_drop,
            encoded_columns = encoded_cols,
            scaled_columns  = scaled_cols,
            saved_paths     = {
                "samples" : samples_path,
                "features": features_path,
                "classes" : classes_path,
            },
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _load(self, csv_path: str) -> pd.DataFrame:
        """Load CSV – thử UTF-8 rồi Latin-1 nếu lỗi. Strip whitespace tên cột."""
        try:
            df = pd.read_csv(csv_path, encoding=self.encoding, low_memory=False)
        except UnicodeDecodeError:
            logger.warning("[DataProcessor] UTF-8 thất bại, thử latin-1...")
            df = pd.read_csv(csv_path, encoding="latin-1", low_memory=False)
        # Strip whitespace khỏi tên cột (CSV hay có trailing space)
        df.columns = df.columns.str.strip()
        return df

    def _log_alignment(self, n_s: int, n_f: int, n_c: int) -> None:
        """In bảng kiểm tra alignment để người dùng đối chiếu."""
        print("\n" + "=" * 52)
        print("  📊 ALIGNMENT CHECK – Kiểm tra đồng bộ số dòng")
        print("=" * 52)
        print(f"  samples.csv  : {n_s:>8,} dòng")
        print(f"  features.csv : {n_f:>8,} dòng")
        print(f"  classes.csv  : {n_c:>8,} dòng")
        print("-" * 52)
        if n_s == n_f == n_c:
            print(f"  ✅ KẾT QUẢ   : ĐỒNG BỘ – {n_s:,} dòng hợp lệ")
        else:
            print("  ❌ KẾT QUẢ   : LỖI – Số dòng KHÔNG KHỚP!")
        print("=" * 52 + "\n")


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------
def process_raw_csv(
    csv_path        : str,
    output_dir      : str,
    class_column    : str,
    id_columns      : Optional[List[str]] = None,
    drop_columns    : Optional[List[str]] = None,
    scale           : bool = True,
    fill_strategy   : str  = "mean",
) -> ProcessingReport:
    """
    Hàm tiện ích gọi nhanh DataProcessor.

    Example
    -------
    >>> report = process_raw_csv(
    ...     csv_path     = "ks-projects-201612.csv",
    ...     output_dir   = "data/processed/session_001",
    ...     class_column = "state",
    ...     id_columns   = ["ID", "name"],
    ...     drop_columns = ["currency", "deadline"],
    ... )
    >>> print(f"Đã tách {report.n_rows} dòng, {report.n_features} features")
    """
    processor = DataProcessor(
        class_column  = class_column,
        id_columns    = id_columns,
        drop_columns  = drop_columns,
        scale         = scale,
        fill_strategy = fill_strategy,
    )
    return processor.process(csv_path, output_dir)
