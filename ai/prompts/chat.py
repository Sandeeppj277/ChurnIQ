"""
ChurnIQ — Grounded Customer Assistant Prompt Template

Answers natural-language questions about a specific customer's churn data.
Strictly grounded in the customer's attributes, prediction probability, and SHAP drivers.
"""

CHAT_SYSTEM_INSTRUCTION = """You are ChurnIQ Assistant, an intelligent customer analytics co-pilot.
You answer questions regarding a specific customer's churn risk, behavioral metrics, and retention strategies.

STRICT GROUNDING RULES:
1. Answer ONLY using the facts present in the provided Customer Profile and Prediction Data.
2. If the user asks something not verifiable from the data, politely state that the data does not contain that information.
3. NEVER contradict the XGBoost prediction or probability.
4. Keep answers concise, factual, and actionable.
"""


def build_chat_prompt(customer: dict, prediction: dict, question: str) -> str:
    """Combine customer context, prediction metrics, and user question."""
    factors_summary = ", ".join([
        f"{f['feature']} ({f['impact']} {f['direction']})"
        for f in prediction.get("top_factors", [])
    ])

    return f"""CUSTOMER CONTEXT:
- ID: {customer.get('customer_id')}, Name: {customer.get('customer_name')}
- Gender: {customer.get('gender')}, Age: {customer.get('age')}
- Tenure: {customer.get('tenure')} months
- Plan: {customer.get('subscription_type')}, Contract: {customer.get('contract_type')}
- Monthly Charges: ${customer.get('monthly_charges'):.2f}, Total: ${customer.get('total_charges'):.2f}
- Payment: {customer.get('payment_method')} (Status: {customer.get('payment_status')})
- Usage Frequency: {customer.get('usage_frequency')} sessions/wk, Service Usage: {customer.get('service_usage')}%
- Complaints: {customer.get('complaint_count')}, Support Calls: {customer.get('support_calls')}
- Engagement Score: {customer.get('engagement_score')} / 100

PREDICTION & DRIVERS:
- Prediction: {prediction.get('prediction')}
- Churn Probability: {prediction.get('churn_probability')}%
- Risk Tier: {prediction.get('risk_level')}
- Key SHAP Drivers: {factors_summary}

USER QUESTION:
"{question}"

Provide a grounded, professional response:
"""
