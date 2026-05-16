# 🛡️ FraudShield AI — Credit Card Fraud Detection Dashboard

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.20%2B-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-10B981?style=for-the-badge)

**A production-grade fintech fraud monitoring system built with Python and Streamlit.**  
Real-time AI predictions · Interactive visualisations · Enterprise dark UI

</div>

---

## 📸 Screenshots

| Dashboard | Prediction | Analytics |
|-----------|-----------|-----------|
| *Main KPI overview with live feed* | *Gauge meter + AI explanation panel* | *Fraud heatmap & distributions* |

> **Tip:** Run the app and capture screenshots for the `assets/` folder to fill this section.

---

## ✨ Features

### 🔍 Prediction Engine
- **Real-time fraud scoring** (0–100) with colour-coded risk levels
- **Fraud probability gauge** using Plotly indicators
- **AI Explanation Panel** — explains *why* a transaction was flagged
- **Prediction history table** with session memory

### 📊 Analytics
- Transaction amount distribution (Normal vs Fraud overlay)
- Class imbalance pie chart
- Fraud rate heatmap (Hour-of-Day × Amount Band)

### 🤖 Model Insights
- Side-by-side model comparison bar chart (Logistic Regression vs Random Forest)
- Confusion matrices for both models
- Top-15 feature importance visualisation
- Full metrics table (Accuracy · Precision · Recall · F1 · ROC-AUC)

### 📡 Live Monitor
- Real-time transaction simulation feed
- Colour-coded status dots (🟢 Normal · 🟡 Suspicious · 🔴 Fraud)
- Auto-refresh mode

### 📥 Report
- Download prediction history as CSV
- Download model metrics as CSV

### 🎨 UI / UX
- Cyber / glassmorphism dark theme
- Animated fraud alert pulses
- Custom CSS — grid background, glow effects, monospace numbers
- Fully responsive sidebar layout

---

## 🏗️ Project Structure

```
fraud_dashboard/
├── app.py                  # Streamlit dashboard (main entry point)
├── model_training.py       # ML training script (run this first)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── creditcard.csv          # Kaggle dataset (download separately)
├── saved_model/            # Auto-created by model_training.py
│   ├── best_model.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── scaler.pkl
│   ├── feature_names.pkl
│   ├── feature_importance.csv
│   ├── metrics.pkl
│   └── test_sample.csv
└── assets/                 # Screenshots / images
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/fraudshield-ai.git
cd fraudshield-ai
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Go to [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
and download `creditcard.csv` into the project root.

> **No Kaggle account?** The training script auto-generates synthetic data so you can still demo the app.

### 5. Train the model
```bash
python model_training.py
```
This creates all artefacts in `saved_model/`. Takes ~2–5 minutes on a laptop.

### 6. Launch the dashboard
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Google Colab Setup

```python
# Cell 1 — Install
!pip install streamlit imbalanced-learn plotly scikit-learn joblib pyngrok -q

# Cell 2 — Upload files
from google.colab import files
uploaded = files.upload()  # upload creditcard.csv (optional)

# Cell 3 — Train
!python model_training.py

# Cell 4 — Run with tunnel
from pyngrok import ngrok
!streamlit run app.py &
public_url = ngrok.connect(8501)
print("Dashboard URL:", public_url)
```

---

## 🧠 ML Details

| Property | Detail |
|---|---|
| Dataset | Kaggle Credit Card Fraud (284,807 transactions, 0.17% fraud) |
| Features | V1–V28 (PCA-anonymised) + Amount + Time |
| Imbalance handling | SMOTE (Synthetic Minority Over-sampling) |
| Models | Logistic Regression · Random Forest (200 trees) |
| Scaling | StandardScaler on Amount & Time |
| Selection criterion | F1-Score (fraud class) |
| Artefact format | joblib `.pkl` |

### Typical Results (real dataset)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~99.2% | ~85% | ~78% | ~81% | ~97% |
| Random Forest | ~99.9% | ~96% | ~84% | ~90% | ~98% |

---

## 🔮 Future Improvements

- [ ] XGBoost / LightGBM model options
- [ ] SHAP value explanations per prediction
- [ ] PostgreSQL transaction log backend
- [ ] WebSocket-based live feed (real streaming)
- [ ] Email / Slack alert integration
- [ ] Docker containerisation
- [ ] REST API endpoint (FastAPI)
- [ ] User authentication layer

---

## 📄 License

MIT © 2024 — Free to use, modify, and distribute.

---

<div align="center">
  <strong>If this project helped you, please ⭐ star the repo!</strong>
</div>
