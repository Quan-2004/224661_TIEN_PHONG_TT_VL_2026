"""
Database – SQLite + SQLAlchemy
=================================
Định nghĩa engine, session và bảng chính:
  - training_logs: Lịch sử huấn luyện model
"""

import os
from datetime import datetime

from sqlalchemy import (
    create_engine, Column, Integer, String,
    Float, DateTime, Text
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from dotenv import load_dotenv

load_dotenv()

# URL database từ .env (mặc định: SQLite local)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./mocsvm.db")

# Tạo engine SQLAlchemy
# connect_args chỉ cần cho SQLite (để hỗ trợ đa luồng FastAPI)
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class cho tất cả ORM models
Base = declarative_base()


# --------------------------------------------------------------------------
# ORM Models
# --------------------------------------------------------------------------

class TrainingLog(Base):
    """Bảng lịch sử huấn luyện model."""
    __tablename__ = "training_logs"

    id           = Column(Integer, primary_key=True, index=True)
    class_name   = Column(String(100), nullable=False)
    version_name = Column(String(100), nullable=False)
    n_samples    = Column(Integer, default=0)
    n_features   = Column(Integer, default=0)
    kernel       = Column(String(20), default="rbf")
    gamma        = Column(String(20), default="scale")
    nu           = Column(Float,  default=0.1)
    status       = Column(String(20), default="success")  # "success" | "failed"
    error_msg    = Column(Text, nullable=True)
    created_at   = Column(DateTime, default=datetime.utcnow)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def create_tables() -> None:
    """Tạo tất cả bảng trong database nếu chưa tồn tại."""
    Base.metadata.create_all(bind=engine)
    print("  [DB] ✓ Bảng database đã được khởi tạo.")


def get_db():
    """
    Dependency FastAPI: cấp phát session và đảm bảo đóng sau mỗi request.

    Dùng với: db: Session = Depends(get_db)
    """
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
