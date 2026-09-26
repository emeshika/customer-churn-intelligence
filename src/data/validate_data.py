"""
Data Validation Module
Validates the schema, row counts, unique constraints, and missingness of the raw Telco dataset.
"""

from pathlib import Path
import pandas as pd
import sys

EXPECTED_COLUMNS = [
    'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
]

def load_and_validate_data(file_path: Path) -> pd.DataFrame:
    print(f"--> Checking file existence at: {file_path}")
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please place telco_churn.csv in data/raw/")

    df = pd.read_csv(file_path)
    print(f"--> Successfully loaded dataset. Shape: {df.shape[0]} rows, {df.shape[1]} columns.")

    # 1. Column Verification
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Schema mismatch! Missing expected columns: {missing_cols}")
    print("Schema validation passed: All expected columns are present.")

    # 2. Check Primary Key / Uniqueness
    duplicate_ids = df['customerID'].duplicated().sum()
    if duplicate_ids > 0:
        raise ValueError(f"Data integrity error: Found {duplicate_ids} duplicate customerIDs.")
    print("Uniqueness check passed: No duplicate customerIDs found.")

    # 3. Check Known Data Anomaly (Whitespace in TotalCharges)
    whitespace_count = (df['TotalCharges'].astype(str).str.strip() == '').sum()
    print(f"--> Found {whitespace_count} rows with whitespace/empty string in 'TotalCharges'. (Expected behavior for new customers with tenure=0)")

    # 4. Check Target Variable
    target_dist = df['Churn'].value_counts(dropna=False).to_dict()
    print(f"--> Target variable ('Churn') distribution: {target_dist}")

    return df

if __name__ == "__main__":
    raw_data_path = Path("data/raw/telco_churn.csv")
    try:
        df = load_and_validate_data(raw_data_path)
        print("\n[SUCCESS] Phase 2 Data Acquisition and Validation completed successfully!")
    except Exception as e:
        print(f"\n[ERROR] Validation failed: {e}", file=sys.stderr)
        sys.exit(1)