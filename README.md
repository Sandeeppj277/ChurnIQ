# ChurnIQ — Customer Churn Prediction & Retention Platform

An end-to-end, production-grade churn intelligence system combining supervised machine learning (**XGBoost**), model explainability (**SHAP**), and **Generative AI** (**Google Gemini 3.8 Flash** with deterministic rule-based fallbacks).

---

## ⚡ Quick Start

### 1. Activate Virtual Environment
```bash
# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and verify your configuration:
```bash
copy .env.example .env
```
In `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
AI_MODEL=gemini-3.8-flash
LOW_RISK_MAX=39
MEDIUM_RISK_MAX=69
DATABASE_URL=sqlite:///churn_dev.db
FRONTEND_URL=http://127.0.0.1:5500
```
*(Note: If `GEMINI_API_KEY` is omitted, ChurnIQ automatically operates in deterministic rule-based fallback mode with zero downtime).*

### 3. Generate Data & Train Model
Run the setup pipeline to generate the calibrated dataset, train the XGBoost classifier, compute SHAP explainers, and output evaluation metrics:
```bash
python setup_project.py
```
**Model Results on 1,000 Test Customers:**
- **Accuracy:** 91.70% (Acceptance target: > 77.0%)
- **ROC-AUC:** 0.9721 (Acceptance target: > 0.800)
- **F1-Score:** 82.15%

### 4. Launch the Server
```bash
uvicorn api.main:app --reload --port 8000
```
Open **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)** in your browser to access the complete application directly.

---

## 🧪 Running Automated Tests
Run the 34-test hermetic test suite:
```bash
pytest -v
```
All tests run with network calls disabled to ensure hermetic verification of endpoints, ML pipelines, and rule-based fallbacks (Rule 10).

---

## 🏛️ Project Structure
```text
churniq/
├── api/
│   └── main.py                     # FastAPI application, CORS, static mounting, error handlers
├── backend/
│   └── app/
│       ├── config.py               # Pydantic BaseSettings from .env
│       ├── api/                    # REST routes (/health, /predict, /model, /dashboard, /ai)
│       ├── models/                 # SQLAlchemy ORM (Customer, Prediction, Recommendation, Message)
│       ├── schemas/                # Pydantic models (CustomerInput, PredictionResponse, etc.)
│       └── services/               # Singleton MLService & PersistenceService
├── ml/
│   ├── data/                       # Synthetic dataset generator (5,015 rows)
│   ├── preprocessing/              # Unified clean_data() function
│   ├── features/                   # ColumnTransformer (25 features) & SHAP explainability
│   ├── training/                   # XGBoost trainer with Random Forest fallback
│   ├── evaluation/                 # Test metrics (accuracy, ROC-AUC, confusion matrix)
│   └── artifacts/                  # churn_model.joblib & metrics.json
├── ai/
│   ├── client.py                   # Google GenAI (gemini-3.8-flash) defensive wrapper
│   ├── prompts/                    # Prompts forbidding prediction changes (Rule 1)
│   ├── recommendation/             # Dual-engine retention strategy & tone-adjusted messaging
│   └── assistant/                  # Grounded customer assistant chat
├── frontend/
│   ├── index.html                  # 4-view dashboard with presets, SVG gauge & chat
│   ├── style.css                   # Responsive SaaS styling
│   ├── app.js                      # View switching, dynamic DOM rendering, XSS escaping (Rule 9)
│   └── api.js                      # Async fetch client
├── database/
│   └── schema.sql                  # Portable SQLite & PostgreSQL schema DDL
├── tests/
│   ├── conftest.py                 # Isolated test DB & cleared GEMINI_API_KEY
│   ├── test_risk_levels.py         # Boundary threshold tests
│   ├── test_ml_pipeline.py         # Data cleaning & transformer tests
│   ├── test_predict.py             # Inference & 422 validation tests
│   ├── test_health.py              # Health check tests
│   ├── test_ai_service.py          # AI & fallback verification
│   └── test_assistant.py           # Grounded assistant tests
├── docs/
│   └── architecture.md             # Detailed architectural specification
├── setup_project.py                # One-command project setup
└── requirements.txt                # Pinned dependencies (Python 3.12)
```

---

## 🛡️ Core Governing Principles
1. **Model Authority:** Only XGBoost predicts churn probability; Gemini strictly explains and recommends.
2. **Zero Training-Serving Skew:** Single `clean_data()` function with frozen training medians/modes.
3. **Graceful Fallback:** Automatic rule-based advice generation if Gemini key is missing or fails.
4. **Auditability:** Every generated output is labeled `source: "ai_generated"` or `source: "rule_based"`.
5. **Defensive Security:** Strict Pydantic validation upfront, XSS escaping via `textContent`, and generic error masking.
