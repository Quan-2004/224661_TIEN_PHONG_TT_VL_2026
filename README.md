<div align="center">

# mOC-iSVM

**Modified One-Class Incremental Support Vector Machine**

A complete MLOps system for multi-class anomaly detection using incremental One-Class SVM with a modern web interface.

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the App](#running-the-app)
- [API Reference](#api-reference)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**mOC-iSVM** is a research-oriented MLOps platform that implements a **multi-class incremental One-Class SVM** (One-vs-Rest strategy). Instead of retraining from scratch, the system learns incrementally by retaining only Support Vectors between training sessions, making it efficient for continuous learning scenarios.

The system provides a full-stack web interface for:

- Uploading raw CSV datasets and auto-training class models
- Visualizing model details and support vectors
- Running inference on new data with detailed confidence scores
- Managing model versions via an XML-based model registry

---

## Features

- 🧠 **Incremental Learning** — Only Support Vectors are stored between sessions, enabling efficient retraining
- 🔁 **Auto Train Pipeline** — Upload raw CSV → preprocess → train/retrain all classes automatically
- 📊 **PCA Visualization** — 2D scatter plot of Support Vectors per class
- 🗂️ **Model Registry** — XML manifest tracks all model versions with metadata
- 🔐 **JWT Authentication** — Secure login/register with role-based access (admin/user)
- 📁 **CSV Testing** — Upload new CSV files and batch-classify all rows
- 📈 **Training History** — Full audit log of all training sessions

---

## Architecture

```
┌──────────────────────────────────────────────┐
│            FRONTEND (React + Vite)            │
│  Dashboard │ Train │ Upload │ Test │ Inference │
└─────────────────────┬────────────────────────┘
                      │ HTTP REST API (JWT)
┌─────────────────────▼────────────────────────┐
│           BACKEND (FastAPI + Python)          │
│  /auth  /upload  /train  /models  /predict   │
│            SQLite (users + logs)              │
└──────────┬───────────────────┬───────────────┘
           │                   │
    ┌──────▼──────┐    ┌───────▼──────┐
    │   mocsvm/   │    │   models/    │
    │  Core ML    │    │  *.pkl files │
    │  Library    │    │  manifest.xml│
    └─────────────┘    └──────────────┘
```

---

## Project Structure

```
mOC-iSVM/
├── mocsvm/                      # Core ML Library (Python package)
│   ├── core/
│   │   ├── incremental.py       # IncrementalOCSVM – single-class model
│   │   ├── multiclass.py        # MultiClassOCSVM – One-vs-Rest wrapper
│   │   └── manifest_manager.py  # XML model registry manager
│   └── utils/
│       └── data_loader.py       # CSV validation & preprocessing
│
├── backend/                     # FastAPI Application
│   ├── main.py                  # App entry point & middleware
│   ├── schemas.py               # Pydantic request/response models
│   ├── requirements.txt         # Python dependencies
│   ├── .env                     # Environment variables (not committed)
│   └── routers/
│       ├── upload.py            # /upload – processed CSV upload
│       ├── upload_raw.py        # /upload-raw – raw CSV upload
│       ├── auto_train.py        # /auto-train – full pipeline
│       ├── train.py             # /train – manual training
│       ├── predict.py           # /predict – inference
│       ├── models_router.py     # /models – model listing & details
│       └── test_csv.py          # /test-csv – batch CSV classification
│
├── frontend/                    # React Web Application
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dashboard.jsx       # Model overview & stats
│   │   │   ├── TrainingPage.jsx    # Manual train/retrain UI
│   │   │   ├── UploadRawPage.jsx   # Auto-train pipeline UI
│   │   │   ├── TestCsvPage.jsx     # CSV batch testing UI
│   │   │   └── ModelDetailModal.jsx # Model details & scatter plot
│   │   ├── services/api.js         # Centralized API client
│   │   └── index.css               # Global dark-mode styles
│   ├── package.json
│   └── vite.config.js
│
├── models/                      # Trained model files
│   ├── *.pkl                    # Serialized model per class
│   └── manifest.xml             # Model version registry
│
├── data/
│   ├── uploads/                 # Uploaded CSV files
│   └── processed/               # Preprocessed CSV files
│
├── setup.py                     # Package setup for mocsvm
├── DEPLOYMENT.md                # Production deployment guide
└── README.md
```

---

## Getting Started

### Prerequisites

| Tool    | Version |
| ------- | ------- |
| Python  | 3.10+   |
| Node.js | 18+     |
| npm     | 9+      |

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/Quan-2004/mOC-iSVM.git
cd mOC-iSVM
```

**2. Set up the Python environment**

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r backend/requirements.txt
pip install -e .   # install the mocsvm package in editable mode
```

**3. Configure environment variables**

```bash
# backend/.env (already provided, edit as needed)
SECRET_KEY=your-secret-key
DATABASE_URL=sqlite:///./mocsvm.db
```

**4. Set up the frontend**

```bash
cd frontend
npm install
```

### Running the App

Run both services in separate terminals:

```bash
# Terminal 1 – Backend
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

```bash
# Terminal 2 – Frontend
cd frontend
npm run dev
```

| Service            | URL                         |
| ------------------ | --------------------------- |
| Web App            | http://localhost:5173       |
| API Docs (Swagger) | http://localhost:8000/docs  |
| API Docs (ReDoc)   | http://localhost:8000/redoc |

---

## API Reference

### Authentication

| Method | Endpoint         | Auth | Description               |
| ------ | ---------------- | ---- | ------------------------- |
| `POST` | `/auth/register` | ❌   | Register a new user       |
| `POST` | `/auth/login`    | ❌   | Login → returns JWT token |
| `GET`  | `/auth/me`       | ✅   | Get current user info     |

### Data & Training

| Method | Endpoint         | Auth | Description                                |
| ------ | ---------------- | ---- | ------------------------------------------ |
| `POST` | `/upload`        | ✅   | Upload preprocessed CSV per class          |
| `POST` | `/upload-raw`    | ✅   | Upload raw CSV, auto-split by class        |
| `POST` | `/auto-train`    | ✅   | Full pipeline: upload + preprocess + train |
| `POST` | `/train`         | ✅   | Manually train or retrain a class model    |
| `GET`  | `/train/history` | ✅   | List all past training sessions            |

### Models & Inference

| Method | Endpoint               | Auth | Description                         |
| ------ | ---------------------- | ---- | ----------------------------------- |
| `GET`  | `/models`              | ❌   | List all registered models          |
| `GET`  | `/models/{class_name}` | ❌   | Get model details + support vectors |
| `POST` | `/predict`             | ❌   | Classify a single data vector       |
| `POST` | `/predict/reload`      | ❌   | Reload models from disk             |
| `POST` | `/test-csv`            | ❌   | Batch classify all rows in a CSV    |

> Full interactive documentation available at `http://localhost:8000/docs`

---

## How It Works

### Incremental Learning

Each class has its own `IncrementalOCSVM` model. When retraining:

1. The existing model's **Support Vectors** are loaded from the `.pkl` file
2. New training data is **merged** with the stored Support Vectors
3. A new OC-SVM is fitted on the combined data
4. Only the resulting **Support Vectors** are saved back — keeping memory footprint minimal

### Model Registry (XML)

`models/manifest.xml` is the central registry. Every train/retrain operation:

- Creates a new versioned `.pkl` file (e.g., `classname-02.pkl`)
- Updates the manifest with metadata (kernel, nu, gamma, n_samples, trained_at)
- Allows the backend to always load the **latest version** of each model

```xml
<manifest updated="2026-03-10T12:00:00">
    <model class_name="Successful" version="Successful-02">
        <pkl_path>models/Successful-02.pkl</pkl_path>
        <metadata>
            <kernel>rbf</kernel>
            <gamma>scale</gamma>
            <nu>0.1</nu>
            <n_samples>150</n_samples>
            <trained_at>2026-03-10T10:00:00</trained_at>
        </metadata>
    </model>
</manifest>
```

### One-vs-Rest Inference

When predicting a new sample, **every class model** is evaluated. The system:

1. Computes the decision score from each OC-SVM
2. Returns the class with the **highest positive score** as the prediction
3. Reports "unknown" if all scores are negative (sample belongs to no known class)

---

## Tech Stack

| Layer         | Technology                          |
| ------------- | ----------------------------------- |
| ML Core       | scikit-learn, numpy, pandas         |
| Backend       | FastAPI, SQLAlchemy, SQLite         |
| Auth          | JWT (python-jose), bcrypt (passlib) |
| Frontend      | React 18, Vite, Vanilla CSS         |
| Serialization | joblib (`.pkl`), XML (manifest)     |

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m 'feat: add some feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

Please follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
Made with ❤️ for research on incremental One-Class SVM
</div>
