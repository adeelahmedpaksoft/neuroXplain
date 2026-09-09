# 🧠 NeuroXplain — Explainable EEG Intelligence

Streamlit dashboard for the NeuroXplain CNN-LSTM + Attention framework.

## Folder Structure

```
neuroxplain_app/
├── app.py                        ← Main Streamlit application
├── requirements.txt              ← Python dependencies
├── README.md                     ← This file
└── neuroxplain_artifacts/        ← All model & data files
    ├── neuroxplain_model.keras   ← Trained CNN-LSTM model
    ├── best_model.keras          ← Best checkpoint
    ├── metadata.pkl              ← Dataset & training metadata
    ├── metrics.pkl               ← Evaluation metrics
    ├── scaler.pkl                ← StandardScaler
    ├── label_encoder.pkl         ← LabelEncoder
    ├── class_meaning.pkl         ← Class label descriptions
    ├── training_history.pkl      ← Training loss/accuracy curves
    ├── saliency_data.pkl         ← Precomputed saliency maps
    └── shap_data.pkl             ← Precomputed SHAP values
```

## Setup & Run (VS Code)

### Step 1 — Open the folder in VS Code
```
File → Open Folder → select neuroxplain_app/
```

### Step 2 — Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run the app
```bash
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**

---

## App Pages

| Page | Description |
|------|-------------|
| 🏠 Overview | Hero metrics, objectives, research gap |
| 📊 Dataset & EDA | Class distribution, EEG signal plots, statistics |
| 🏗️ Model Architecture | Layer-by-layer CNN-LSTM diagram, training config |
| 📈 Training & Evaluation | Loss/accuracy curves, all 7 metrics, confusion matrix, ROC |
| 🔍 SHAP Explanations | Global importance, beeswarm, waterfall, per-class SHAP |
| 🗺️ Saliency Maps | Vanilla gradient, SmoothGrad, attention weights, dashboard |
| 🩺 Live Prediction | Classify EEG sample + real-time saliency + attention + export |
| ⚖️ Model Comparison | NeuroXplain vs Random Forest, SVM, Dummy — bar + radar |

---

## Notes
- Tested with Python 3.10 / 3.11
- GPU not required (CPU inference is fast)
- Upload a CSV with 178 numeric columns on the Live Prediction page to classify your own EEG
