-- Schema DDL for Customer Retention Intelligence Platform

DROP TABLE IF EXISTS retention_actions;
DROP TABLE IF EXISTS predictions;
DROP TABLE IF EXISTS customers;

-- 1. Customers Table (Master Record)
CREATE TABLE customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    gender VARCHAR(10),
    senior_citizen INT,
    partner VARCHAR(5),
    dependents VARCHAR(5),
    tenure INT,
    phone_service VARCHAR(5),
    multiple_lines VARCHAR(20),
    internet_service VARCHAR(20),
    online_security VARCHAR(20),
    online_backup VARCHAR(20),
    device_protection VARCHAR(20),
    tech_support VARCHAR(20),
    streaming_tv VARCHAR(20),
    streaming_movies VARCHAR(20),
    contract VARCHAR(30),
    paperless_billing VARCHAR(5),
    payment_method VARCHAR(40),
    monthly_charges REAL,
    total_charges REAL,
    actual_churn INT
);

-- 2. Predictions & Risk Intelligence Table
CREATE TABLE predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id VARCHAR(50),
    churn_probability REAL,
    risk_level VARCHAR(20),
    segment_id INT,
    segment_name VARCHAR(50),
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- 3. Retention Actions Table
CREATE TABLE retention_actions (
    action_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id VARCHAR(50),
    suggested_action TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);