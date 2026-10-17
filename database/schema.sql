-- ============================================================
-- ChurnIQ — Relational Database Schema
-- Compatible with PostgreSQL and SQLite
-- ============================================================

-- 1. Customers Table (stores profile attributes, upserted on prediction)
CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    customer_name VARCHAR(100) NOT NULL,
    gender VARCHAR(10) NOT NULL,
    age INTEGER NOT NULL,
    tenure INTEGER NOT NULL,
    subscription_type VARCHAR(20) NOT NULL,
    contract_type VARCHAR(20) NOT NULL,
    monthly_charges NUMERIC(10, 2) NOT NULL,
    total_charges NUMERIC(12, 2) NOT NULL,
    payment_method VARCHAR(30) NOT NULL,
    payment_status VARCHAR(20) NOT NULL,
    usage_frequency NUMERIC(5, 1) NOT NULL,
    service_usage NUMERIC(5, 1) NOT NULL,
    complaint_count INTEGER NOT NULL,
    support_calls INTEGER NOT NULL,
    engagement_score NUMERIC(5, 1) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Predictions Table (appended on every POST /predict)
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id VARCHAR(50) NOT NULL,
    prediction VARCHAR(30) NOT NULL,
    churn_probability NUMERIC(5, 2) NOT NULL,
    risk_level VARCHAR(10) NOT NULL,
    top_factors TEXT NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id) ON DELETE CASCADE
);

-- 3. Recommendations Table (persists Gemini or rule-based retention plans)
CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id VARCHAR(50) NOT NULL,
    prediction_id INTEGER NOT NULL,
    explanation TEXT NOT NULL,
    recommended_action TEXT NOT NULL,
    reason TEXT NOT NULL,
    priority VARCHAR(10) NOT NULL,
    source VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id) ON DELETE CASCADE,
    FOREIGN KEY (prediction_id) REFERENCES predictions (id) ON DELETE CASCADE
);

-- 4. Messages Table (persists tone-adjusted outreach messages)
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id VARCHAR(50) NOT NULL,
    prediction_id INTEGER NOT NULL,
    message TEXT NOT NULL,
    tone VARCHAR(20) NOT NULL,
    source VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id) ON DELETE CASCADE,
    FOREIGN KEY (prediction_id) REFERENCES predictions (id) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_predictions_customer_id ON predictions(customer_id);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_risk_level ON predictions(risk_level);
CREATE INDEX IF NOT EXISTS idx_recommendations_customer_id ON recommendations(customer_id);
CREATE INDEX IF NOT EXISTS idx_messages_customer_id ON messages(customer_id);
