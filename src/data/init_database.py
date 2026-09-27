"""
Database Initialization & Ingestion Module
Populates SQLite database with full customer profiles, model predictions, and retention actions.
"""

import sys
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
import joblib

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.features.build_features import engineer_features
from src.recommendations.retention_engine import generate_retention_recommendations

def initialize_database(raw_csv_path: Path, models_dir: Path, db_path: Path, schema_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Connect and Execute Schema
    conn = sqlite3.connect(db_path)
    with open(schema_path, 'r') as f:
        conn.executescript(f.read())
    print(f"Database schema applied to: {db_path}")

    # 2. Load Raw Data
    df_raw = pd.read_csv(raw_csv_path)
    df_raw['TotalCharges'] = pd.to_numeric(df_raw['TotalCharges'].astype(str).str.strip(), errors='coerce').fillna(0.0)
    df_raw['actual_churn'] = (df_raw['Churn'] == 'Yes').astype(int)

    # 3. Load ML Artifacts
    preprocessor = joblib.load(models_dir / "preprocessor.joblib")
    model = joblib.load(models_dir / "best_model.joblib")
    seg_artifact = joblib.load(models_dir / "kmeans_segmentation.joblib")
    
    # 4. Ingest Customers Table
    cust_columns = {
        'customerID': 'customer_id', 'SeniorCitizen': 'senior_citizen', 'Partner': 'partner',
        'Dependents': 'dependents', 'PhoneService': 'phone_service', 'MultipleLines': 'multiple_lines',
        'InternetService': 'internet_service', 'OnlineSecurity': 'online_security', 'OnlineBackup': 'online_backup',
        'DeviceProtection': 'device_protection', 'TechSupport': 'tech_support', 'StreamingTV': 'streaming_tv',
        'StreamingMovies': 'streaming_movies', 'Contract': 'contract', 'PaperlessBilling': 'paperless_billing',
        'PaymentMethod': 'payment_method', 'MonthlyCharges': 'monthly_charges', 'TotalCharges': 'total_charges'
    }
    df_cust_sql = df_raw.rename(columns=cust_columns)
    cols_to_keep = list(cust_columns.values()) + ['gender', 'tenure', 'actual_churn']
    df_cust_sql[cols_to_keep].to_sql('customers', conn, if_exists='append', index=False)
    print(f"Ingested {len(df_cust_sql)} records into 'customers' table.")

    # 5. Compute Features for Prediction and Segmentation
    df_engineered = engineer_features(df_raw)
    X = df_engineered.drop(columns=['customerID', 'Churn', 'actual_churn'], errors='ignore')
    X_transformed = preprocessor.transform(X)
    
    # Feature names DataFrame එකක් ලෙස ලබා දීම (UserWarning වැළැක්වීමට)
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'charges_ratio', 'total_services']
    categorical_features = [col for col in X.columns if col not in numeric_features]
    feature_names = numeric_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)
    
    probabilities = model.predict_proba(X_transformed_df)[:, 1]

    # Compute Clusters using engineered features (total_services අඩංගු df_engineered භාවිතයෙන්)
    X_seg = df_engineered[seg_artifact['features']].copy()
    X_seg_scaled = seg_artifact['scaler'].transform(X_seg)
    cluster_labels = seg_artifact['kmeans'].predict(X_seg_scaled)

    # 6. Populate Predictions and Retention Actions
    pred_records = []
    action_records = []

    for idx, row in df_raw.iterrows():
        cid = row['customerID']
        prob = float(probabilities[idx])
        cluster = int(cluster_labels[idx])
        
        rec_data = generate_retention_recommendations(
            row.to_dict(), prob, cluster, ["Automated Batch Risk Evaluation"]
        )
        
        pred_records.append((
            cid, rec_data['churn_probability'], rec_data['risk_level'],
            cluster, rec_data['segment_name']
        ))

        for act in rec_data['suggested_retention_actions']:
            action_records.append((cid, act))

    conn.executemany(
        """INSERT INTO predictions (customer_id, churn_probability, risk_level, segment_id, segment_name)
           VALUES (?, ?, ?, ?, ?)""",
        pred_records
    )
    conn.executemany(
        """INSERT INTO retention_actions (customer_id, suggested_action) VALUES (?, ?)""",
        action_records
    )
    conn.commit()
    conn.close()

    print(f"Ingested {len(pred_records)} predictions and {len(action_records)} retention recommendations.")
    print("Phase 11 Database initialization completed successfully!")

if __name__ == "__main__":
    raw_csv = Path("data/raw/telco_churn.csv")
    models_p = Path("models")
    db_p = Path("data/churn_platform.db")
    schema_p = Path("sql/schema.sql")
    
    initialize_database(raw_csv, models_p, db_p, schema_p)