"""
Feature Engineering and Preprocessing Pipeline
Builds business features, splits data safely, and fits preprocessor on training partition.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib

def load_and_clean_data(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    
    # 1. Remove whitespaces in TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
    # make total charges 0 for new customers with tenure 0
    df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
    
    # 2. Target Encoding
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Tenure cohorts 
    df['tenure_cohort'] = pd.cut(
        df['tenure'], 
        bins=[-1, 12, 24, 48, 72], 
        labels=['0-12M', '13-24M', '25-48M', '49-72M']
    ).astype(str)
    
    # Charges ratio
    expected_spend = df['tenure'] * df['MonthlyCharges']
    df['charges_ratio'] = np.where(expected_spend > 0, df['TotalCharges'] / expected_spend, 1.0)
    
    # Service density count 
    services = [
        'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    df['total_services'] = df[services].apply(lambda row: sum(row == 'Yes'), axis=1)
    
    # Redundant categorical labels 
    multi_no_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in multi_no_cols:
        df[col] = df[col].replace({'No internet service': 'No'})
    df['MultipleLines'] = df['MultipleLines'].replace({'No phone service': 'No'})

    return df

def build_pipeline_and_split(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    #devide Features and Target 
    X = df.drop(columns=['customerID', 'Churn'])
    y = df['Churn']
    
    # Stratified Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'charges_ratio', 'total_services']
    categorical_features = [col for col in X.columns if col not in numeric_features]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # only fit on training data to prevent Data leakage 
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # get Feature names 
    feature_names = numeric_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
    
    # save Processed data 
    pd.DataFrame(X_train_transformed, columns=feature_names).to_csv(processed_dir / "X_train.csv", index=False)
    pd.DataFrame(X_test_transformed, columns=feature_names).to_csv(processed_dir / "X_test.csv", index=False)
    y_train.to_csv(processed_dir / "y_train.csv", index=False)
    y_test.to_csv(processed_dir / "y_test.csv", index=False)
    
    # save Preprocessor artifact 
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")
    
    print(f"Stratified Split Complete: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    print(f"Engineered Features Shape: {len(feature_names)} columns after one-hot encoding")
    print(f"Preprocessor saved to: {output_dir / 'preprocessor.joblib'}")
    print(f"Processed datasets saved to: {processed_dir}")

if __name__ == "__main__":
    raw_data_path = Path("data/raw/telco_churn.csv")
    models_dir = Path("models")
    
    df_clean = load_and_clean_data(raw_data_path)
    df_engineered = engineer_features(df_clean)
    build_pipeline_and_split(df_engineered, models_dir)