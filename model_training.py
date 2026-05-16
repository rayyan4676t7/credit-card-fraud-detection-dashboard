"""
model_training.py
==================
Credit Card Fraud Detection - Model Training Script
Trains Logistic Regression and Random Forest models, evaluates them,
and saves the best model along with preprocessing artifacts.

Run this script BEFORE launching the Streamlit app.
Usage:
    python model_training.py
"""

import os
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score,
    classification_report, average_precision_score
)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
RANDOM_STATE   = 42
TEST_SIZE      = 0.20
MODEL_DIR      = "saved_model"
DATA_PATH      = "creditcard.csv"   # Place the Kaggle dataset here
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 2. DATA LOADING  (real or synthetic fallback)
# ─────────────────────────────────────────────
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load dataset; generate synthetic data if the file is not found."""
    if os.path.exists(path):
        print(f"[✓] Loading real dataset from '{path}' …")
        df = pd.read_csv(path)
    else:
        print("[!] Dataset not found — generating synthetic data for demo …")
        df = generate_synthetic_dataset()
    print(f"[✓] Dataset shape : {df.shape}")
    print(f"[✓] Fraud ratio   : {df['Class'].mean()*100:.4f}%")
    return df


def generate_synthetic_dataset(n_samples: int = 50_000) -> pd.DataFrame:
    """
    Create a realistic synthetic credit-card fraud dataset
    that mimics the Kaggle structure (V1-V28 + Amount + Time + Class).
    """
    rng = np.random.default_rng(RANDOM_STATE)

    # ~0.17 % fraud — matching Kaggle distribution
    n_fraud  = int(n_samples * 0.0017)
    n_normal = n_samples - n_fraud

    # Normal transactions
    normal = pd.DataFrame(
        rng.normal(0, 1, (n_normal, 28)),
        columns=[f"V{i}" for i in range(1, 29)]
    )
    normal["Amount"] = rng.exponential(88, n_normal)
    normal["Time"]   = rng.uniform(0, 172_800, n_normal)
    normal["Class"]  = 0

    # Fraudulent transactions (slightly shifted means)
    fraud = pd.DataFrame(
        rng.normal(0.5, 1.5, (n_fraud, 28)),
        columns=[f"V{i}" for i in range(1, 29)]
    )
    fraud["Amount"] = rng.exponential(122, n_fraud)
    fraud["Time"]   = rng.uniform(0, 172_800, n_fraud)
    fraud["Class"]  = 1

    df = pd.concat([normal, fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return X (features) and y (target)."""
    feature_cols = [c for c in df.columns if c != "Class"]
    X = df[feature_cols].copy()
    y = df["Class"].copy()
    return X, y


# ─────────────────────────────────────────────
# 4. EVALUATION HELPER
# ─────────────────────────────────────────────
def evaluate_model(name: str, model, X_test, y_test) -> dict:
    """Compute and print classification metrics; return as dict."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model"        : name,
        "accuracy"     : accuracy_score(y_test, y_pred),
        "precision"    : precision_score(y_test, y_pred, zero_division=0),
        "recall"       : recall_score(y_test, y_pred, zero_division=0),
        "f1"           : f1_score(y_test, y_pred, zero_division=0),
        "roc_auc"      : roc_auc_score(y_test, y_proba),
        "avg_precision": average_precision_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  Accuracy       : {metrics['accuracy']:.4f}")
    print(f"  Precision      : {metrics['precision']:.4f}")
    print(f"  Recall         : {metrics['recall']:.4f}")
    print(f"  F1-Score       : {metrics['f1']:.4f}")
    print(f"  ROC-AUC        : {metrics['roc_auc']:.4f}")
    print(f"  Avg Precision  : {metrics['avg_precision']:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Normal','Fraud'])}")
    return metrics


# ─────────────────────────────────────────────
# 5. MAIN TRAINING ROUTINE
# ─────────────────────────────────────────────
def train_and_save():
    # ── Load data ──────────────────────────────
    df   = load_data()
    X, y = engineer_features(df)

    feature_names = list(X.columns)

    # ── Train / Test split (stratified) ────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )
    print(f"\n[✓] Train size : {X_train.shape[0]:,}  |  Test size : {X_test.shape[0]:,}")

    # ── Scale Amount / Time only (V-features are already PCA-scaled) ───
    scaler = StandardScaler()

    # ── SMOTE on training data (handle class imbalance) ────────────────
    print("\n[~] Applying SMOTE to balance training classes …")
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    print(f"[✓] After SMOTE — Fraud: {y_train_res.sum():,}  Normal: {(y_train_res==0).sum():,}")

    results = {}

    # ── Model A: Logistic Regression ───────────
    print("\n[~] Training Logistic Regression …")
    lr = LogisticRegression(
        max_iter=1000,
        C=0.1,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_STATE
    )
    lr.fit(X_train_res, y_train_res)
    results["Logistic Regression"] = evaluate_model(
        "Logistic Regression", lr, X_test_scaled, y_test
    )
    results["Logistic Regression"]["model_obj"] = lr

    # ── Model B: Random Forest ─────────────────
    print("\n[~] Training Random Forest …")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    rf.fit(X_train_res, y_train_res)
    results["Random Forest"] = evaluate_model(
        "Random Forest", rf, X_test_scaled, y_test
    )
    results["Random Forest"]["model_obj"] = rf

    # ── Pick Best Model (by F1-score) ──────────
    best_name = max(
        ["Logistic Regression", "Random Forest"],
        key=lambda k: results[k]["f1"]
    )
    best_model = results[best_name]["model_obj"]
    print(f"\n[★] Best model : {best_name}  (F1 = {results[best_name]['f1']:.4f})")

    # ── Feature Importance (RF) or Coefficients (LR) ───────────────────
    if best_name == "Random Forest":
        importance_vals = best_model.feature_importances_
    else:
        importance_vals = np.abs(best_model.coef_[0])

    importance_df = pd.DataFrame({
        "feature"   : feature_names,
        "importance": importance_vals
    }).sort_values("importance", ascending=False)

    # ── Save artefacts ─────────────────────────
    print(f"\n[~] Saving artefacts to '{MODEL_DIR}/' …")

    joblib.dump(best_model,  f"{MODEL_DIR}/best_model.pkl")
    joblib.dump(lr,           f"{MODEL_DIR}/logistic_regression.pkl")
    joblib.dump(rf,           f"{MODEL_DIR}/random_forest.pkl")
    joblib.dump(scaler,       f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(feature_names,f"{MODEL_DIR}/feature_names.pkl")
    importance_df.to_csv(f"{MODEL_DIR}/feature_importance.csv", index=False)

    # Save metrics (strip non-serialisable model objects)
    metrics_to_save = {}
    for k, v in results.items():
        metrics_to_save[k] = {kk: vv for kk, vv in v.items() if kk != "model_obj"}
    metrics_to_save["best_model_name"] = best_name
    joblib.dump(metrics_to_save, f"{MODEL_DIR}/metrics.pkl")

    # Save a small sample of test data for the dashboard demo feed
    test_sample = X_test.copy()
    test_sample["Class"] = y_test.values
    test_sample.to_csv(f"{MODEL_DIR}/test_sample.csv", index=False)

    print("\n[✓] All artefacts saved successfully!")
    print(f"    ├── best_model.pkl")
    print(f"    ├── logistic_regression.pkl")
    print(f"    ├── random_forest.pkl")
    print(f"    ├── scaler.pkl")
    print(f"    ├── feature_names.pkl")
    print(f"    ├── feature_importance.csv")
    print(f"    ├── metrics.pkl")
    print(f"    └── test_sample.csv")
    print("\n[✓] Training complete. You can now run:  streamlit run app.py")


if __name__ == "__main__":
    train_and_save()
