# ChurnIQ — Architecture & System Documentation

## 1. System Overview

**ChurnIQ** is an end-to-end customer churn prediction, explainability, and AI-driven retention platform. It integrates:
- A high-accuracy supervised **XGBoost Classifier** (with automatic Random Forest fallback) to compute churn probability and classify customers into LOW, MEDIUM, or HIGH risk tiers.
- Individual-level **SHAP (TreeExplainer)** explainability determining the top 5 directional drivers influencing each customer's risk.
- A **Google Gemini Generative AI** layer (`gemini-3.8-flash`) translating probabilities and SHAP drivers into executive explanations, prioritized retention action plans, tone-adjustable outreach messages, and an interactive grounded customer assistant.
- A deterministic **Rule-Based Fallback Engine** ensuring 100% application uptime without external API keys or network availability.
- A **FastAPI REST API** with Pydantic validation, SQLite/PostgreSQL persistence, and a responsive **HTML5/CSS3/Vanilla JS** dashboard.

---

## 2. Core Architectural Rules

1. **Only the Model Predicts (Rule 1):** The XGBoost model computes churn probability and binary classification. Gemini prompts strictly forbid altering or recalculating predictions.
2. **One Cleaning Function (Rule 2):** Training and single-customer scoring call the exact same `clean_data()` function with frozen training medians and modes, eliminating training-serving skew.
3. **Key Stays on Server (Rule 3):** `GEMINI_API_KEY` resides strictly in `.env`. Never committed to git, never exposed to clients.
4. **AI Failure Never Breaks App (Rule 4):** If the Gemini key is unconfigured, rate-limited, or returns invalid JSON, the deterministic fallback executes seamlessly without throwing 500 errors.
5. **Output Source Auditing (Rule 5):** Every recommendation and message is tagged with `source: "ai_generated"` or `source: "rule_based"`.
6. **Validation First (Rule 6):** Pydantic models validate input upfront (HTTP 422). Missing model files return HTTP 503. Internal server exceptions return a sanitized error message without leaking stack traces.
7. **Singleton Cache (Rule 7):** Model bundle and `shap.TreeExplainer` load once on application startup.
8. **Configurable Thresholds (Rule 8):** Risk boundaries (`LOW_RISK_MAX=39`, `MEDIUM_RISK_MAX=69`) and 50% churn cutoff are driven by settings.
9. **XSS Escaping (Rule 9):** User values rendered in the frontend use `textContent` and standard escaping, avoiding `innerHTML` injection.
10. **Hermetic Test Suite (Rule 10):** Tests run with network disabled for Gemini and use in-memory SQLite instances.

---

## 3. Machine Learning & Feature Engineering

- **Raw Inputs:** 16 attributes (9 numeric, 5 categorical).
- **Preprocessing:** Duplicate removal, median/mode imputation, age boundary replacement (`[1, 110]`), and clipping.
- **Encoding:** `ColumnTransformer` with `StandardScaler` on 9 numerics and `OneHotEncoder(handle_unknown='ignore')` on 5 categoricals yielding **25 model features**.
- **Classifier:** `XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, random_state=42)`.
- **Performance:**
  - Test Accuracy: **91.70%** (Target: $> 77.0\%$)
  - ROC-AUC: **0.9721** (Target: $> 0.800$)
  - Precision: **84.89%** | Recall: **79.58%** | F1-Score: **82.15%**

---

## 4. REST API Contracts

| Method | Path | Request Body | Response Payload | Description |
|---|---|---|---|---|
| `GET` | `/health` | None | `status`, `model_ready`, `ai_enabled` | Operational health and readiness check |
| `POST` | `/predict` | `CustomerInput` | `PredictionResponse` | Scores customer, runs SHAP, upserts customer, logs prediction |
| `GET` | `/model/metrics` | None | Metrics JSON | Serves accuracy, ROC-AUC, confusion matrix, top-10 global features |
| `GET` | `/dashboard/summary` | None | `DashboardSummary` | Aggregates scored counts, risk tiers, avg prob, recent 10 predictions |
| `POST` | `/ai/recommendation` | `{"customer_id": "..."}` | `RecommendationResponse` | Generates & persists retention explanation, action, and priority |
| `POST` | `/ai/message` | `{"customer_id": "...", "tone": "..."}` | `MessageResponse` | Generates & persists outreach text (`Professional`, `Friendly`, etc.) |
| `POST` | `/ai/chat` | `{"customer_id": "...", "question": "..."}` | `ChatResponse` | Grounded interactive assistant answering questions about customer |

---

## 5. Database Schema

- **`customers`**: Stores profile demographics, billing info, and engagement scores (upserted on prediction).
- **`predictions`**: Stores prediction history, churn probabilities, risk levels, and serialized top 5 SHAP drivers.
- **`recommendations`**: Stores executive explanations, retention recommendations, priorities, and origin tags (`ai_generated` / `rule_based`).
- **`messages`**: Stores drafted customer communication with selected tone and source tags.
