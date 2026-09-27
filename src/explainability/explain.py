"""
Explainable AI Module (SHAP)
Computes global feature importances and customer-level attribution factors.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib

def generate_shap_explanations(models_dir: Path, data_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Model & Test Features
    model = joblib.load(models_dir / "best_model.joblib")
    X_test = pd.read_csv(data_dir / "X_test.csv")
    
    print("--> Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    
    # Random Forest probability explanation uses index 1 (Churn=1 class)
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list):
        # Scikit-learn Random Forest returns a list of [class_0, class_1]
        shap_vals_churn = shap_values[1]
    elif len(shap_values.shape) == 3:
        # Some versions return 3D array (samples, features, classes)
        shap_vals_churn = shap_values[:, :, 1]
    else:
        shap_vals_churn = shap_values

    # 2. Global Feature Importance Summary
    mean_abs_shap = np.abs(shap_vals_churn).mean(axis=0)
    importance_df = pd.DataFrame({
        'Feature': X_test.columns,
        'Mean_SHAP_Impact': mean_abs_shap
    }).sort_values(by='Mean_SHAP_Impact', ascending=False)
    
    print("\n" + "=" * 60)
    print("TOP 10 GLOBAL CHURN PREDICTORS (SHAP ATTRIBUTION)")
    print("=" * 60)
    print(importance_df.head(10).to_string(index=False))
    print("=" * 60)

    # 3. Save Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals_churn, X_test, max_display=10, show=False)
    plt.title("SHAP Feature Importance (Impact on Churn Prediction)", fontsize=14)
    plt.tight_layout()
    summary_plot_path = output_dir / "shap_summary_plot.png"
    plt.savefig(summary_plot_path, bbox_inches='tight')
    plt.close()
    print(f"\nSHAP summary plot saved to: {summary_plot_path}")

    # 4. Save Explainer Artifact for API / Dashboard
    joblib.dump(explainer, models_dir / "shap_explainer.joblib")
    print(f"SHAP explainer artifact saved to: {models_dir / 'shap_explainer.joblib'}")

if __name__ == "__main__":
    models_path = Path("models")
    data_path = Path("data/processed")
    figures_path = Path("reports/figures")
    
    generate_shap_explanations(models_path, data_path, figures_path)