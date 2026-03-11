"""
mocsvm/utils/encoder_manager.py
===============================
Lưu và tải LabelEncoders để đảm bảo string/category được chuyển đổi 
ra số đồng nhất giữa tập Train (Phase 0) và Test (Phase 3).
"""

import os
import joblib
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)

class CategoricalEncoderManager:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.encoders_path = self.model_dir / "label_encoders.pkl"

    def save_encoders(self, encoders: Dict[str, LabelEncoder]):
        """Lưu toàn bộ dictionary các encoder (cột -> LabelEncoder)."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(encoders, self.encoders_path)
        logger.info(f"Đã lưu {len(encoders)} LabelEncoders vào {self.encoders_path}")

    def load_encoders(self) -> Optional[Dict[str, LabelEncoder]]:
        """Tải encoders. Trả về None nếu file chưa tồn tại."""
        if self.encoders_path.exists():
            return joblib.load(self.encoders_path)
        return None

    def transform_df(self, df: pd.DataFrame, encoders: Dict[str, LabelEncoder]) -> pd.DataFrame:
        """
        Áp dụng encoders lên DataFrame mới. 
        Nếu dính giá trị chưa từng học, gán thành một nhãn đặc biệt (hoặc map về <unknown> class nội bộ).
        Để đơn giản, fallback về class cuối cùng hoặc nhãn -1 nếu cần (LabelEncoder mặc định raise error nếu unknown,
        vì vậy ta handle thủ công).
        """
        df_out = df.copy()
        for col, le in encoders.items():
            if col in df_out.columns:
                # Xử lý các nhãn lạ (Unseen labels) bằng việc đưa về giá trị đặc biệt hoặc fill NaN
                # Cách lành tính: thêm <unknown> lúc train, hoặc fallback. Ở đây dùng mapping pandas:
                known_classes = set(le.classes_)
                # Biến unseen thành pd.NA hoặc map tay
                mapped = df_out[col].astype(str).map(
                    lambda x: x if x in known_classes else "<unknown>"
                )
                
                # Nếu '<unknown>' không có trong training classes, LabelEncoder sẽ lỗi khi fit.
                # Cách an toàn nhất cho inference: tạm encode bằng classes có sẵn, những thằng lạ gán giá trị = -1
                
                # Biến list classes thành từ điển {name: index}
                le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
                df_out[col] = df_out[col].astype(str).map(le_dict).fillna(-1).astype(int)
                
        return df_out
