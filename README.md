# mOC-iSVM – Modified One-Class Incremental SVM

> Hệ thống MLOps hoàn chỉnh cho thuật toán One-Class SVM đa lớp với học tăng cường.

## Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────┐
│                   FRONTEND (React + Vite)                │
│  Dashboard │ Training Page │ Scatter Visualization       │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP/REST (JWT)
┌──────────────────────▼──────────────────────────────────┐
│                 BACKEND (FastAPI + Python 3.10)          │
│  /auth  │ /upload │ /train │ /models │ /predict         │
│                   SQLite Database                        │
└─────────┬───────────────────────┬───────────────────────┘
          │                       │
   ┌──────▼──────┐        ┌───────▼──────┐
   │  mocsvm/    │        │   models/    │
   │  (Core ML   │        │  *.pkl files │
   │   Library)  │        │  manifest.xml│
   └─────────────┘        └──────────────┘
```

## Cấu trúc thư mục

```
mOC-isvm2/
├── mocsvm/                      # Core ML Library
│   ├── core/
│   │   ├── incremental.py       # IncrementalOCSVM
│   │   ├── multiclass.py        # MultiClassOCSVM (One-vs-Rest)
│   │   └── manifest_manager.py  # XML Registry
│   └── utils/
│       └── data_loader.py       # CSV validation
├── backend/                     # FastAPI Backend
│   ├── main.py                  # Entry point
│   ├── database.py              # SQLite + SQLAlchemy
│   ├── auth.py                  # JWT + bcrypt
│   ├── schemas.py               # Pydantic schemas
│   ├── routers/                 # API endpoints
│   ├── .env                     # Environment config
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/                    # React Frontend
│   ├── src/
│   │   ├── components/          # Dashboard, TrainingPage, ScatterPlot
│   │   ├── services/api.js      # API layer
│   │   └── index.css            # Dark mode styles
│   ├── nginx.conf
│   └── Dockerfile
├── models/                      # Model files (.pkl) + XML
│   └── global_manifest.xml      # Model registry
├── data/uploads/                # Uploaded CSV files
└── docker-compose.yml
```

## Cách XML điều phối hệ thống

File `models/global_manifest.xml` là **Registry trung tâm**. Mỗi khi một model được train/retrain:

```xml
<?xml version="1.0" encoding="utf-8"?>
<manifest updated="2025-01-01T12:00:00">
    <model class_name="successful" version="successful-02">
        <pkl_path>models/successful-02.pkl</pkl_path>
        <metadata>
            <kernel>rbf</kernel>
            <gamma>scale</gamma>
            <nu>0.1</nu>
            <n_samples>150</n_samples>
            <trained_at>2025-01-01T10:00:00</trained_at>
        </metadata>
    </model>
</manifest>
```

- **Frontend** đọc `/models` API → Backend đọc XML → trả JSON cho Dashboard.
- **Retrain** tự động tăng phiên bản: `successful-01` → `successful-02`.

## Cách SQLite quản lý dữ liệu người dùng

Hai bảng chính:

- **`users`**: username, hashed_password (bcrypt), role (admin/user), is_active
- **`training_logs`**: lịch sử mỗi lần train/retrain, liên kết với user_id

## Cài đặt nhanh (Development)

```bash
# 1. Tạo môi trường ảo Python 3.10
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/Mac

# 2. Cài đặt backend
pip install -r backend/requirements.txt

# 3. Khởi động backend
uvicorn backend.main:app --reload --port 8000
# → Swagger UI: http://localhost:8000/docs

# 4. Cài đặt & khởi động frontend
cd frontend
npm install
npm run dev
# → App: http://localhost:5173
```

## API Endpoints

| Method | Endpoint          | Auth | Mô tả            |
| ------ | ----------------- | ---- | ---------------- |
| POST   | `/auth/register`  | ❌   | Đăng ký          |
| POST   | `/auth/login`     | ❌   | Đăng nhập → JWT  |
| GET    | `/auth/me`        | ✅   | Thông tin user   |
| POST   | `/upload`         | ✅   | Upload 3 CSV     |
| POST   | `/train`          | ✅   | Train/Retrain    |
| GET    | `/train/history`  | ✅   | Lịch sử train    |
| GET    | `/models`         | ❌   | Danh sách models |
| POST   | `/predict`        | ❌   | Dự đoán          |
| POST   | `/predict/reload` | ❌   | Reload models    |

## Docker

```bash
docker-compose up --build
# Backend: http://localhost:8000/docs
# Frontend: http://localhost:3000
```
