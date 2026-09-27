"""
FastAPI Microservice for AI-Powered Churn & Retention Intelligence
Endpoints:
- POST /predict: Real-time inference, risk-scoring, SHAP drivers, recommendations
- GET /customer/{customer_id}: Retrieve persisted customer intelligence from DB
- GET /health: Healthcheck endpoint
"""

import sys
from pathlib import Path
import sqlite3
from typing import List, Optional
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.features.build_features import engineer_features
from src.recommendations.retention_engine import generate_retention_recommendations

app = FastAPI(
    title="Customer Churn & Retention Intelligence API",
    description="End-to-End ML Inference, SHAP Drivers & Retention Recommendation Engine",
    version="1.0.0"
)

# Global artifacts holders
MODELS_DIR = ROOT_DIR / "models"
DB_PATH = ROOT_DIR / "data" / "churn_platform.db"

preprocessor = None
model = None
explainer = None
seg_artifact = None

@app.on_event("startup")
def load_artifacts():
    global preprocessor, model, explainer, seg_artifact
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    model = joblib.load(MODELS_DIR / "best_model.joblib")
    explainer = joblib.load(MODELS_DIR / "shap_explainer.joblib")
    seg_artifact = joblib.load(MODELS_DIR / "kmeans_segmentation.joblib")
    print("✓ All production artifacts (Pipeline, Model, SHAP, K-Means) loaded successfully.")

# Request Schema
class CustomerInput(BaseModel):
    customerID: str = Field(default="CUST-NEW-01", description="Unique Customer Identifier")
    gender: str = Field(default="Female", description="Male / Female")
    SeniorCitizen: int = Field(default=0, ge=0, le=1)
    Partner: str = Field(default="No", description="Yes / No")
    Dependents: str = Field(default="No", description="Yes / No")
    tenure: int = Field(default=5, ge=0, description="Months active")
    PhoneService: str = Field(default="Yes")
    MultipleLines: str = Field(default="No")
    InternetService: str = Field(default="Fiber optic")
    OnlineSecurity: str = Field(default="No")
    OnlineBackup: str = Field(default="No")
    DeviceProtection: str = Field(default="No")
    TechSupport: str = Field(default="No")
    StreamingTV: str = Field(default="Yes")
    StreamingMovies: str = Field(default="Yes")
    Contract: str = Field(default="Month-to-month")
    PaperlessBilling: str = Field(default="Yes")
    PaymentMethod: str = Field(default="Electronic check")
    MonthlyCharges: float = Field(default=89.5, ge=0.0)
    TotalCharges: float = Field(default=447.5, ge=0.0)

# Response Schemas
class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    risk_level: str
    customer_segment: str
    top_risk_factors: List[str]
    suggested_retention_actions: List[str]

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy", "service": "Churn Intelligence API", "version": "1.0.0"}

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_churn(customer: CustomerInput):
    cust_dict = customer.model_dump()
    raw_df = pd.DataFrame([cust_dict])

    # 1. Feature Engineering
    engineered_df = engineer_features(raw_df)
    features_only = engineered_df.drop(columns=['customerID', 'Churn'], errors='ignore')

    # 2. Preprocessing & Alignment
    X_transformed = preprocessor.transform(features_only)
    
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'charges_ratio', 'total_services']
    categorical_features = [c for c in features_only.columns if c not in numeric_features]
    encoded_cat_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
    feature_names = numeric_features + encoded_cat_names
    
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

    # 3. Model Inference
    churn_proba = float(model.predict_proba(X_transformed_df)[0, 1])

    # 4. Local SHAP Attribution
    shap_vals = explainer.shap_values(X_transformed_df)
    if isinstance(shap_vals, list):
        sample_shap = shap_vals[1][0]
    elif len(shap_vals.shape) == 3:
        sample_shap = shap_vals[0, :, 1]
    else:
        sample_shap = shap_vals[0]

    # Get top 3 features increasing churn probability
    impact_series = pd.Series(sample_shap, index=feature_names)
    top_churn_drivers = impact_series.sort_values(ascending=False).head(3).index.tolist()

    # 5. Customer Segmentation
    X_seg = engineered_df[seg_artifact['features']].copy()
    X_seg_scaled = seg_artifact['scaler'].transform(X_seg)
    cluster_id = int(seg_artifact['kmeans'].predict(X_seg_scaled)[0])

    # 6. Retention Recommendations
    decision = generate_retention_recommendations(
        cust_dict, churn_proba, cluster_id, top_churn_drivers
    )

    return {
        "customer_id": customer.customerID,
        "churn_probability": decision['churn_probability'],
        "risk_level": decision['risk_level'],
        "customer_segment": decision['segment_name'],
        "top_risk_factors": decision['top_drivers'],
        "suggested_retention_actions": decision['suggested_retention_actions']
    }

@app.get("/customer/{customer_id}", tags=["Database Retrieval"])
def get_persisted_customer(customer_id: str):
    if not DB_PATH.exists():
        raise HTTPException(status_code=500, detail="Database file not found.")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT c.customer_id, c.contract, c.monthly_charges, c.tenure,
               p.churn_probability, p.risk_level, p.segment_name
        FROM customers c
        JOIN predictions p ON c.customer_id = p.customer_id
        WHERE c.customer_id = ?
    """, (customer_id,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Customer ID {customer_id} not found in database.")

    cursor.execute("SELECT suggested_action FROM retention_actions WHERE customer_id = ?", (customer_id,))
    actions = [r[0] for r in cursor.fetchall()]
    conn.close()

    return {
        "customer_id": row[0],
        "contract": row[1],
        "monthly_charges": row[2],
        "tenure_months": row[3],
        "churn_probability": row[4],
        "risk_level": row[5],
        "customer_segment": row[6],
        "suggested_retention_actions": actions
    }