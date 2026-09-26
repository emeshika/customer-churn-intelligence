"""
EDA Module: Generates core statistical summaries and answers business questions regarding churn.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(file_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(file_path)

    # Clean whitespace in TotalCharges temporarily for EDA numeric analysis
    df['TotalCharges_clean'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce').fillna(0.0)
    df['Churn_binary'] = (df['Churn'] == 'Yes').astype(int)

    print("=" * 60)
    print("1. OVERALL CHURN METRICS")
    print("=" * 60)
    overall_churn = df['Churn_binary'].mean() * 100
    print(f"Overall Churn Rate: {overall_churn:.2f}%\n")

    print("=" * 60)
    print("2. CHURN RATE BY CONTRACT TYPE")
    print("=" * 60)
    contract_churn = df.groupby('Contract')['Churn_binary'].agg(['count', 'mean'])
    contract_churn['churn_rate_%'] = contract_churn['mean'] * 100
    print(contract_churn[['count', 'churn_rate_%']].sort_values(by='churn_rate_%', ascending=False))
    print()

    print("=" * 60)
    print("3. CHURN RATE BY INTERNET SERVICE")
    print("=" * 60)
    internet_churn = df.groupby('InternetService')['Churn_binary'].agg(['count', 'mean'])
    internet_churn['churn_rate_%'] = internet_churn['mean'] * 100
    print(internet_churn[['count', 'churn_rate_%']].sort_values(by='churn_rate_%', ascending=False))
    print()

    print("=" * 60)
    print("4. CHURN RATE BY PAYMENT METHOD")
    print("=" * 60)
    payment_churn = df.groupby('PaymentMethod')['Churn_binary'].agg(['count', 'mean'])
    payment_churn['churn_rate_%'] = payment_churn['mean'] * 100
    print(payment_churn[['count', 'churn_rate_%']].sort_values(by='churn_rate_%', ascending=False))
    print()

    print("=" * 60)
    print("5. NUMERICAL SUMMARY BY CHURN")
    print("=" * 60)
    num_summary = df.groupby('Churn')[['tenure', 'MonthlyCharges', 'TotalCharges_clean']].mean()
    print(num_summary)
    print("=" * 60)

    # Visualizations
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Tenure vs Churn
    sns.boxplot(x='Churn', y='tenure', data=df, ax=axes[0], palette=['#4CAF50', '#F44336'])
    axes[0].set_title("Customer Tenure vs Churn")
    
    # Monthly Charges vs Churn
    sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', common_norm=False, ax=axes[1], fill=True, palette=['#4CAF50', '#F44336'])
    axes[1].set_title("Monthly Charges Distribution by Churn")

    # Contract Type Churn Percentage
    contract_churn['churn_rate_%'].plot(kind='bar', ax=axes[2], color='#FF7043')
    axes[2].set_title("Churn Rate (%) by Contract Type")
    axes[2].set_ylabel("Churn Rate (%)")
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45)

    plt.tight_layout()
    plot_path = output_dir / "eda_summary.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"\n[INFO] Summary visual plots saved to: {plot_path}")

if __name__ == "__main__":
    raw_path = Path("data/raw/telco_churn.csv")
    fig_dir = Path("reports/figures")
    run_eda(raw_path, fig_dir)