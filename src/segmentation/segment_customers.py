"""
Customer Segmentation Module
Applies K-Means clustering on behavioral and spending attributes (without churn target)
to identify actionable customer personas.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib

def run_segmentation(raw_data_path: Path, models_dir: Path):
    models_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(raw_data_path)
    
    # 1. Clean TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce').fillna(0.0)
    
    # 2. Derive behavioral features
    services = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['total_services'] = df[services].apply(lambda row: sum(row == 'Yes'), axis=1)
    
    # Clustering Features: Tenure, Spending, and Service breadth
    cluster_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'total_services']
    X_cluster = df[cluster_cols].copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # 3. Evaluate Silhouette Scores for K=3, 4, 5
    print("--> Evaluating Cluster counts (K=3 to 5)...")
    for k in [3, 4, 5]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        print(f"K={k} | Silhouette Score: {sil:.4f}")
        
    # We select K=3 as it yields distinct, interpretable telecom personas
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # 4. Interpret Clusters (Profiling)
    profile = df.groupby('cluster')[cluster_cols].mean()
    profile['customer_count'] = df.groupby('cluster')['customerID'].count()
    profile['churn_rate_%'] = df.groupby('cluster')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
    
    print("\n" + "=" * 80)
    print("CUSTOMER SEGMENT PROFILES (K=3)")
    print("=" * 80)
    print(profile.round(2))
    print("=" * 80)
    
    # 5. Save Artifacts
    segmentation_artifact = {
        'scaler': scaler,
        'kmeans': kmeans,
        'features': cluster_cols,
        'profile': profile
    }
    joblib.dump(segmentation_artifact, models_dir / "kmeans_segmentation.joblib")
    print(f"\nSegmentation artifact saved to: {models_dir / 'kmeans_segmentation.joblib'}")

if __name__ == "__main__":
    raw_csv = Path("data/raw/telco_churn.csv")
    models_path = Path("models")
    run_segmentation(raw_csv, models_path)