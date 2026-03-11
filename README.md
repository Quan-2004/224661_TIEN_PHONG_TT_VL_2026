<div align="center">

# mOC-iSVM

**Modified One-Class Incremental Support Vector Machine**

_A pure algorithmic research project implementing multi-class anomaly detection via an incremental One-Class SVM approach._

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19.2-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)

[Overview](#overview) • [Core Architecture](#core-architecture) • [Getting Started](#getting-started) • [Contributing](#contributing)

</div>

---

## 📖 Overview

**mOC-iSVM** constitutes a dedicated, pure algorithmic implementation designed to solve multi-class anomaly detection utilizing an **incremental One-Class SVM (One-vs-Rest strategy)**.

Traditional Support Vector Machine (SVM) algorithms demand a full dataset retraining from scratch every time new data points emerge. To resolve this inefficiency, mOC-iSVM incorporates an **incremental learning mechanism** that retains explicit Support Vectors from previous training iterations, making it highly robust and drastically reducing computational overhead during continuous data influx.

The architecture comprises a foundational ML package (`mocsvm`), exposed interactively via a lightweight FastAPI backend and visualized seamlessly using React.

## 🧠 Core Architecture: 4 Phases of mOC-iSVM

Based on our `ARCHITECTURE.md`, the system is built around 4 highly optimized algorithmic phases:

### Phase 0: Preprocessing & Global Scaler
Before any SVM model is trained, the system establishes a **Global Scaler** utilizing a unified standardized coordinate system. 
- **The Global Scaler** prevents the isolated coordinate displacement that causes extreme boundary overlap when individual classes are normalized separately.
- **Categorical Encoder Manager** persists mapping dictionaries to guarantee exact string-to-integer translation between Train and Test CSV phases.

### Phase 1: Train Workflow & SV Extraction
1. The global dataset is split into isolated class segments ($X_A, X_B, X_C$).
2. A unique **One-Class SVM** model is trained independently for each class.
3. The system executes a rigorous GridSearch across $(\gamma, \nu)$ parameters (specifically tuned for tight boundaries using large gamma and nu bounds) using other classes as negative reinforcement.
4. **Support Vectors (SVs)** are extracted. The raw data is deleted to free RAM, keeping only mathematically significant SV boundary descriptors as "Compressed Memory".

### Phase 2: Retrain Workflow & SV Pruning
When incrementally retraining with new data, the system fuses incoming data with historical SV memory. To prevent "Concept Drift" and bloated memory boundaries:
- **Age Pruning**: Erases historical SVs that exceed an alpha-cycle lifespan threshold.
- **Error Pruning**: Conducts an accuracy benchmark against incoming data, completely flushing legacy SVs if performance drops below the precision threshold.

### Phase 3: Test Workflow & Euclidean Tie-Break
During prediction, a completely unknown sample ($x_{test}$) is broadcast via OVR to all established models. 
- If multiple models claim the sample (returning positive margin > 0), the system enters into an overlap resolution.
- **Euclidean Nearest-SV Tie-break**: The system suspends complex Kernel evaluations and measures the absolute geometric Euclidean Distance to the closest internal Support Vector of the contending classes. The class possessing the nearest "outpost SV" decisively wins the Tie-break.

## ✨ Technical Features

- **Algorithmic Primacy**: Focused completely on the ML matrix math and modeling; stripped of bloated databases or extraneous authentication logic.
- **🔄 Auto Train Pipeline**: Upload raw CSV matrices → Auto-preprocess → System trains/retrains all detected bounds dynamically.
- **📊 Dimensionality Reduction**: Real-time Principal Component Analysis (PCA) maps multidimensional Support Vectors into 2D scatter visualizations.
- **🗂️ Stateless Model Registry**: A declarative XML-based paradigm (`global_manifest.xml`) records tuning hyperparameters linking to distinct sequential `.pkl` versions.
- **🧪 Complete CSV Inference**: Entirely anonymous CSV inferencing utilizing automated saved `LabelEncoders` and exact-match automated feature filtering.

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
   
   # Setup global python module
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
