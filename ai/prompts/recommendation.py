"""
ChurnIQ — Retention Recommendation Prompt Template

Strictly enforces Rule 1: The prompt forbids modifying or disputing the ML model's prediction.
Directs Gemini to explain why the customer was assigned this probability and recommend a targeted,
actionable retention strategy.
"""

RECOMMENDATION_SYSTEM_INSTRUCTION = """You are ChurnIQ's Chief Retention Strategist.
Your duty is to explain the XGBoost model's churn prediction and devise a high-impact retention plan.

CRITICAL RULES:
1. The XGBoost model alone computes the churn probability and risk tier. You MUST NOT alter, contradict, recalculate, or doubt the model's prediction or probability.
2. Formulate your reasoning specifically around the customer's top SHAP factors and behavioral profile.
3. Your output MUST be strict JSON matching this schema:
{
  "explanation": "2-3 concise sentences summarizing why the customer is likely or not likely to churn based on the top SHAP drivers.",
  "recommended_action": "Specific, practical retention offer, intervention, or engagement step.",
  "reason": "Direct business rationale connecting the recommendation to the customer's risk factors.",
  "priority": "High" | "Medium" | "Low"
}
"""


def build_recommendation_prompt(customer: dict, prediction: dict) -> str:
    """Format the customer profile and prediction into the prompt."""
    factors_summary = "\n".join([
        f"- {f['feature']}: impact magnitude {f['impact']} ({f['direction']})"
        for f in prediction.get("top_factors", [])
    ])

    return f"""Analyze this customer profile and churn prediction:

CUSTOMER PROFILE:
- ID: {customer.get('customer_id')}
- Name: {customer.get('customer_name')}
- Gender: {customer.get('gender')}, Age: {customer.get('age')}
- Tenure: {customer.get('tenure')} months
- Plan: {customer.get('subscription_type')}, Contract: {customer.get('contract_type')}
- Monthly Charges: ${customer.get('monthly_charges'):.2f}, Total: ${customer.get('total_charges'):.2f}
- Payment: {customer.get('payment_method')} (Status: {customer.get('payment_status')})
- Usage Frequency: {customer.get('usage_frequency')} sessions/week, Service Usage: {customer.get('service_usage')}%
- Complaints: {customer.get('complaint_count')}, Support Calls: {customer.get('support_calls')}
- Engagement Score: {customer.get('engagement_score')} / 100

XGBOOST PREDICTION (FROZEN & AUTHORITATIVE):
- Prediction: {prediction.get('prediction')}
- Churn Probability: {prediction.get('churn_probability')}%
- Assigned Risk Level: {prediction.get('risk_level')}

TOP INFLUENTIAL SHAP FACTORS:
{factors_summary}

Generate the JSON retention recommendation now.
"""
