<div align="center">

# mOC-iSVM

**Modified One-Class Incremental Support Vector Machine**

_A pure algorithmic research project implementing multi-class anomaly detection via an incremental One-Class SVM approach._

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19.2-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)

[Overview](#overview) • [Core Algorithm](#core-algorithm) • [Installation](#installation) • [Data Structures](#data-structures)

</div>

---

## 📖 Overview

**mOC-iSVM** constitutes a dedicated, pure algorithmic implementation designed to solve multi-class anomaly detection utilizing an **incremental One-Class SVM (One-vs-Rest strategy)**.

Traditional Support Vector Machine (SVM) algorithms demand a full dataset retraining from scratch every time new data points emerge. To resolve this inefficiency, mOC-iSVM incorporates an **incremental learning mechanism** that retains explicit Support Vectors from previous training iterations, making it highly robust and drastically reducing computational overhead during continuous data influx.

The architecture comprises a foundational ML package (`mocsvm`), exposed interactively via a lightweight FastAPI backend and visualized seamlessly using React.

## 🧠 Core Algorithm: How It Works

### Incremental Target Learning

At its core, `mOC-iSVM` instantiates a distinct `IncrementalOCSVM` object assigned to each detected classification class. Upon initiating a model retraining sequence:

1. **Extraction**: Preceding Support Vectors (the bounding threshold data points defining the hypersphere boundary) are efficiently loaded from serialized `.pkl` arrays.
2. **Aggregation**: Newly ingested data bounds are structurally appended strictly to these legacy Support Vectors.
3. **Execution**: A succeeding scikit-learn `OneClassSVM` kernel executes optimization fitting exclusively over this minimized footprint dataset.
4. **Distillation**: The output isolates new Support Vectors which overwrite the prior records.

By only caching bounding Support Vectors instead of full datasets, memory utilization and optimization compute times undergo exponential reduction over extensive lifecycle loops.

### One-vs-Rest (OvR) Inference Logic

Prediction evaluates unstructured data across all distinct incremental class models concurrently.

1. Computing individual Decision Scores per OC-SVM.
2. The maximal positive score dictates the predicted label.
3. System outputs an "Unknown" classification if all margins yield negatively, accurately flagging anomaly occurrences.

## ✨ Technical Features

- **Algorithmic Primacy**: Focused completely on the ML matrix math and modeling; stripped of bloated databases or extraneous authentication logic.
- **🔄 Auto Train Pipeline**: Upload raw CSV matrices → Auto-preprocess → System trains/retrains all detected bounds dynamically.
- **📊 Dimensionality Reduction**: Real-time Principal Component Analysis (PCA) maps multidimensional Support Vectors into 2D scatter visualizations.
- **🗂️ Stateless Model Registry**: A declarative XML-based paradigm records tuning hyperparameters (`gamma`, `nu`, `kernel`) linking to distinct sequential `.pkl` versions.
- **🧪 Batch Heuristics**: Extensive pipeline logic designed to ingest full unclassified CSV datasets to simulate large-scale testing (OvR).

## 🚀 Getting Started

### Prerequisites

- **Python:** 3.10+
- **Node.js:** 18.x+

### Installation & Execution

1. **Clone the repository**

   ```bash
   git clone https://github.com/Quan-2004/mOC-iSVM.git
   cd mOC-iSVM
   ```

2. **Initialize Python Environment**

   ```bash
   python -m venv .venv

   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate

   pip install -r backend/requirements.txt

   # Emplace core algorithmic library globally
   pip install -e .
   ```

3. **Start the API Server**

   ```bash
   # From project root
   uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Start the Visualizer Frontend**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

### Application URLs

- **Web Analytics UI:** [http://localhost:5173](http://localhost:5173)
- **Algorithm API Docs (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📂 Algorithmic Data Structures

Ensure inputted CSV pipelines match structural heuristics for matrix parsing.

### 1. `samples.csv` (Dense Feature Matrix)

A strict numerical matrix lacking header delineations.

```csv
1.2,0.5,3.1,2.0
1.8,0.3,2.9,1.7
2.1,0.2,3.3,1.9
```

### 2. `features.csv` (Feature Dimensions)

Explicit 1D vector mapping to sequence columns.

```csv
feature_1,feature_2,feature_3,feature_4
```

### 3. `classes.csv` (Target Vector Object)

1D string array denoting sequential targets corresponding directly to the `samples.csv` bounds. No headers.

```csv
successful
successful
failed
```

_Note: The platform provides an `Upload CSV Thô` (Raw CSV) functionality that inherently splits and parses unified datasets based on the final column._

---

## 🤝 Contributing

This algorithmic repository encourages ML engineering research expansions.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/Optimization`)
3. Commit Changes (`git commit -m 'feat: Enhance matrix parsing'`)
4. Push to the Branch (`git push origin feature/Optimization`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

<div align="center">
<br/>
<em>Pioneered for advanced research on incremental One-Class SVM strategies.</em>
</div>
