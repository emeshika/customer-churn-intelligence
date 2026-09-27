"""
Advanced Modeling & Hyperparameter Tuning Module
Trains Random Forest and XGBoost with imbalance handling, compares against baseline,
and persists the best performing model.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix
)

def evaluate_predictions(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
        "PR-AUC": average_precision_score(y_true, y_proba)
    }

def train_and_tune(data_dir: Path, models_dir: Path):
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    X_train = pd.read_csv(data_dir / "X_train.csv")
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").values.ravel()
    y_test = pd.read_csv(data_dir / "y_test.csv").values.ravel()
    
    # Calculate scale_pos_weight for XGBoost
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    imbalance_ratio = neg_count / pos_count
    print(f"--> Class Imbalance Ratio (Train Negative/Positive): {imbalance_ratio:.2f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    # 2. Random Forest Tuning
    print("\n[1/2] Tuning Random Forest...")
    rf_param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 8, 12, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    rf_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=rf_param_dist,
        n_iter=10,
        scoring='roc_auc',
        cv=cv,
        random_state=42,
        n_jobs=-1
    )
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_

    # 3. XGBoost Tuning
    print("[2/2] Tuning XGBoost...")
    xgb_param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'scale_pos_weight': [1.0, imbalance_ratio]
    }
    xgb_search = RandomizedSearchCV(
        XGBClassifier(random_state=42, eval_metric='logloss'),
        param_distributions=xgb_param_dist,
        n_iter=10,
        scoring='roc_auc',
        cv=cv,
        random_state=42,
        n_jobs=-1
    )
    xgb_search.fit(X_train, y_train)
    best_xgb = xgb_search.best_estimator_

    # 4. Evaluate on Test Set
    # Load Baseline for direct comparison
    baseline = joblib.load(models_dir / "baseline_logreg.joblib")
    results['Baseline Logistic Regression'] = evaluate_predictions(
        y_test, baseline.predict(X_test), baseline.predict_proba(X_test)[:, 1]
    )
    results['Tuned Random Forest'] = evaluate_predictions(
        y_test, best_rf.predict(X_test), best_rf.predict_proba(X_test)[:, 1]
    )
    results['Tuned XGBoost'] = evaluate_predictions(
        y_test, best_xgb.predict(X_test), best_xgb.predict_proba(X_test)[:, 1]
    )

    # 5. Display Benchmark Comparison Table
    df_comparison = pd.DataFrame(results).T
    print("\n" + "=" * 75)
    print("MODEL PERFORMANCE COMPARISON (TEST DATA)")
    print("=" * 75)
    print(df_comparison.round(4))
    print("=" * 75)

    # Select Best Model based on ROC-AUC & Recall balance
    best_model_name = "Tuned XGBoost" if results['Tuned XGBoost']['ROC-AUC'] >= results['Tuned Random Forest']['ROC-AUC'] else "Tuned Random Forest"
    selected_model = best_xgb if best_model_name == "Tuned XGBoost" else best_rf
    
    print(f"\nSelected Best Model: {best_model_name}")
    joblib.dump(selected_model, models_dir / "best_model.joblib")
    print(f"Best model saved to: {models_dir / 'best_model.joblib'}")

if __name__ == "__main__":
    data_directory = Path("data/processed")
    models_directory = Path("models")
    train_and_tune(data_directory, models_directory)