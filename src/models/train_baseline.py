"""
Baseline Model Module
Trains a standard Logistic Regression model to establish benchmark performance.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, average_precision_score, 
    confusion_matrix, classification_report
)
import joblib

def evaluate_baseline(data_dir: Path, models_dir: Path):
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Processed Data
    X_train = pd.read_csv(data_dir / "X_train.csv")
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").values.ravel()
    y_test = pd.read_csv(data_dir / "y_test.csv").values.ravel()
    
    # 2. Fit Baseline Logistic Regression
    baseline_model = LogisticRegression(max_iter=1000, random_state=42)
    baseline_model.fit(X_train, y_train)
    
    # 3. Predict on Test Set
    y_pred = baseline_model.predict(X_test)
    y_proba = baseline_model.predict_proba(X_test)[:, 1]
    
    # 4. Calculate Evaluation Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    # 5. Display Metrics
    print("=" * 60)
    print("PHASE 5: BASELINE LOGISTIC REGRESSION BENCHMARK")
    print("=" * 60)
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")
    print("-" * 60)
    print("Confusion Matrix:")
    print(f"TN: {cm[0, 0]} | FP: {cm[0, 1]}")
    print(f"FN: {cm[1, 0]} | TP: {cm[1, 1]}")
    print("=" * 60)
    
    # Business Note on False Negatives
    print("\n[BUSINESS INTERPRETATION]")
    print(f"Out of {cm[1, 0] + cm[1, 1]} actual churners, the baseline missed {cm[1, 0]} customers (False Negatives).")
    print("High False Negatives are risky for retention because churners leave undetected.")
    
    # 6. Save Model Artifact
    joblib.dump(baseline_model, models_dir / "baseline_logreg.joblib")
    print(f"\n Baseline model saved to: {models_dir / 'baseline_logreg.joblib'}")

if __name__ == "__main__":
    data_directory = Path("data/processed")
    models_directory = Path("models")
    evaluate_baseline(data_directory, models_directory)