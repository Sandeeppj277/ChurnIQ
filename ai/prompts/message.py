"""
ChurnIQ — Customer Outreach Message Prompt Template

Drafts personalized customer communication tailored to the specified tone:
- Professional
- Friendly
- Empathetic
- Short
"""

MESSAGE_SYSTEM_INSTRUCTION = """You are ChurnIQ's Customer Communications Director.
Your job is to draft a personalized, proactive outreach message to a customer.

GUIDELINES:
1. Address the customer by name.
2. Align the message with the requested tone:
   - "Professional": Courteous, articulate, business-focused.
   - "Friendly": Warm, approachable, collaborative.
   - "Empathetic": Understanding, appreciative of their experience, focused on solving difficulties.
   - "Short": Direct, punchy, 2-3 sentences max.
3. Subtly address pain points (e.g. recent complaints, payment transitions, high charges) without explicitly saying "our AI predicted you will churn".
4. Output strict JSON with:
{
  "message": "The full text of the outreach message."
}
"""


def build_message_prompt(customer: dict, prediction: dict, tone: str) -> str:
    """Format customer context and requested tone into message prompt."""
    return f"""Draft an outreach message for this customer:

CUSTOMER:
- Name: {customer.get('customer_name')}
- Plan: {customer.get('subscription_type')}, Contract: {customer.get('contract_type')}
- Tenure: {customer.get('tenure')} months
- Complaints: {customer.get('complaint_count')}, Support Calls: {customer.get('support_calls')}
- Risk Level: {prediction.get('risk_level')}
- Primary Drivers: {', '.join([f['feature'] for f in prediction.get('top_factors', [])[:3]])}

REQUESTED TONE: {tone}

Generate the JSON message now.
"""
