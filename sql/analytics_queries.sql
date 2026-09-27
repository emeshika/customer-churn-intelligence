-- Analytical Reporting Queries

-- 1. High-Value, High-Risk Customers Requiring Urgent Intervention
SELECT 
    c.customer_id,
    c.contract,
    c.monthly_charges,
    p.churn_probability,
    p.risk_level,
    p.segment_name
FROM customers c
JOIN predictions p ON c.customer_id = p.customer_id
WHERE p.risk_level IN ('High Risk', 'Very High Risk')
  AND c.monthly_charges >= 75.0
ORDER BY p.churn_probability DESC
LIMIT 10;

-- 2. Churn Risk Distribution by Customer Segment
SELECT 
    p.segment_name,
    COUNT(c.customer_id) AS total_customers,
    ROUND(AVG(p.churn_probability) * 100, 2) AS avg_churn_probability_pct,
    SUM(CASE WHEN p.risk_level IN ('High Risk', 'Very High Risk') THEN 1 ELSE 0 END) AS at_risk_count,
    ROUND(AVG(c.monthly_charges), 2) AS avg_monthly_revenue
FROM customers c
JOIN predictions p ON c.customer_id = p.customer_id
GROUP BY p.segment_name
ORDER BY avg_churn_probability_pct DESC;