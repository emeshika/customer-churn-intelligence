"""
Retention Recommendation Engine
Translates model predictions, segments, and behavioral drivers into business decision-support actions.
"""

from typing import Dict, List, Any

# Configurable business thresholds
RISK_THRESHOLDS = {
    'low': 0.30,
    'medium': 0.60,
    'high': 0.80
}

def assign_risk_tier(churn_probability: float) -> str:
    """Classifies probability into business risk tiers."""
    if churn_probability < RISK_THRESHOLDS['low']:
        return "Low Risk"
    elif churn_probability < RISK_THRESHOLDS['medium']:
        return "Medium Risk"
    elif churn_probability < RISK_THRESHOLDS['high']:
        return "High Risk"
    else:
        return "Very High Risk"

def generate_retention_recommendations(
    customer_info: Dict[str, Any],
    churn_prob: float,
    segment_label: int,
    top_drivers: List[str]
) -> Dict[str, Any]:
    """
    Generates rule-based, decision-support retention actions.
    (Recommendations are actionable suggestions, not guarantees).
    """
    risk_level = assign_risk_tier(churn_prob)
    recommendations: List[str] = []

    # Heuristic 1: Contract vulnerability
    contract = customer_info.get('Contract', '')
    if contract == 'Month-to-month' and churn_prob >= 0.40:
        recommendations.append("Incentivize 1-year contract migration with a discount on the first 3 months.")

    # Heuristic 2: Service/Fiber issues
    internet = customer_info.get('InternetService', '')
    tech_support = customer_info.get('TechSupport', '')
    if internet == 'Fiber optic' and tech_support == 'No' and churn_prob >= 0.50:
        recommendations.append("Provide 6 months of complimentary Tech Support / Device Protection to address service friction.")

    # Heuristic 3: High payment friction
    payment = customer_info.get('PaymentMethod', '')
    if payment == 'Electronic check' and churn_prob >= 0.40:
        recommendations.append("Promote auto-pay enrollment (Credit Card / Bank Transfer) with a one-time bill credit.")

    # Heuristic 4: High-value customer at risk (Segment 1 or 2 with high monthly spend)
    monthly_charges = float(customer_info.get('MonthlyCharges', 0.0))
    if monthly_charges > 70.0 and churn_prob >= 0.60:
        recommendations.append("Assign high-priority customer relationship manager check-in to review plan value.")

    # Heuristic 5: Early lifecycle risk
    tenure = float(customer_info.get('tenure', 0.0))
    if tenure <= 12 and churn_prob >= 0.50:
        recommendations.append("Trigger automated 90-day onboarding engagement sequence to demonstrate product value.")

    # Default action if no specific heuristics triggered but risk is moderate/high
    if not recommendations and churn_prob >= 0.30:
        recommendations.append("Include customer in standard quarterly loyalty appreciation program.")
    elif not recommendations:
        recommendations.append("Maintain standard service level; customer is stable.")

    # Segment naming map based on Phase 9 insights
    segment_names = {
        0: "Low-Cost Basic User",
        1: "High-Value Loyal Champion",
        2: "High-Risk New High-Spender"
    }

    return {
        "churn_probability": round(churn_prob, 4),
        "risk_level": risk_level,
        "segment_id": segment_label,
        "segment_name": segment_names.get(segment_label, f"Segment {segment_label}"),
        "top_drivers": top_drivers,
        "suggested_retention_actions": recommendations
    }

if __name__ == "__main__":
    # Test sample simulating an at-risk customer
    sample_customer = {
        "customerID": "TEST-1024",
        "tenure": 3,
        "MonthlyCharges": 85.50,
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic",
        "TechSupport": "No",
        "PaymentMethod": "Electronic check"
    }
    
    simulated_prob = 0.82
    simulated_segment = 2
    simulated_drivers = ["Month-to-month contract", "Fiber optic service", "Short tenure"]
    
    result = generate_retention_recommendations(
        sample_customer, simulated_prob, simulated_segment, simulated_drivers
    )
    
    import json
    print("=" * 60)
    print("PHASE 10: SAMPLE RETENTION ENGINE OUTPUT")
    print("=" * 60)
    print(json.dumps(result, indent=2))