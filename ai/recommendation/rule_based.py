"""
ChurnIQ — Deterministic Rule-Based Fallback Generator

Fulfills Rules 4 & 5:
- Guarantees the system operates 100% reliably even without a Gemini API key or on network failure.
- Explicitly tags output as source='rule_based'.
- Analyzes risk level and top SHAP drivers to synthesize structured retention advice and messages.
"""

from typing import Any


def generate_rule_based_recommendation(
    customer: dict[str, Any],
    prediction: dict[str, Any],
) -> dict[str, Any]:
    """Generate deterministic retention recommendations from customer attributes and SHAP factors."""
    risk_level = prediction.get("risk_level", "LOW")
    prob = prediction.get("churn_probability", 0.0)
    top_factors = prediction.get("top_factors", [])
    cust_name = customer.get("customer_name", "Customer")
    contract = customer.get("contract_type", "Month-to-Month")
    complaints = customer.get("complaint_count", 0)
    engagement = customer.get("engagement_score", 50.0)
    pay_status = customer.get("payment_status", "On Time")

    # Extract dominant risk-increasing factors
    increasing_factors = [
        f["feature"] for f in top_factors if f.get("direction") == "increases_risk"
    ]
    top_driver = increasing_factors[0] if increasing_factors else "General usage pattern"

    if risk_level == "HIGH":
        priority = "High"
        if "Contract: Month-to-Month" in increasing_factors or contract == "Month-to-Month":
            explanation = (
                f"{cust_name} demonstrates an elevated {prob}% churn risk, driven primarily by a flexible Month-to-Month contract "
                f"and low commitment signals."
            )
            action = (
                "Offer a 15% promotional rate discount conditioned on migrating to a 1-Year or 2-Year annual agreement, "
                "paired with a complimentary feature add-on."
            )
            reason = "Longer contract terms structurally mitigate churn and establish sustained customer lifetime value."
        elif complaints > 0 or "Complaint Count" in increasing_factors:
            explanation = (
                f"{cust_name} shows a {prob}% churn risk, heavily correlated with {complaints} logged complaint(s) "
                "and service friction."
            )
            action = (
                "Initiate immediate proactive outreach by a Senior Customer Success Manager to resolve outstanding issues "
                "and provide a one-time bill credit."
            )
            reason = "Rapid resolution of unresolved friction points restores trust before total disengagement occurs."
        elif pay_status in ["Failed", "Delayed"]:
            explanation = (
                f"{cust_name} is at critical {prob}% risk due to payment friction ({pay_status} status) "
                "disrupting seamless account renewal."
            )
            action = (
                "Send an automated payment assistance notification offering multiple fallback payment methods (UPI/Credit Card) "
                "with a 7-day grace period."
            )
            reason = "Resolving involuntary payment failures retains customers without discounting core service."
        else:
            explanation = (
                f"{cust_name} has high churn probability ({prob}%), impacted by {top_driver} "
                f"and declining engagement ({engagement}/100)."
            )
            action = "Enroll the account in an executive VIP re-engagement campaign with priority support access."
            reason = "Direct intervention reverses declining utilization patterns."

    elif risk_level == "MEDIUM":
        priority = "Medium"
        explanation = (
            f"{cust_name} exhibits moderate churn risk ({prob}%). Primary influential drivers include {top_driver}."
        )
        action = (
            "Schedule a quarterly business check-in and share customized training resources highlighting unused plan benefits."
        )
        reason = "Early engagement prevents moderate-risk accounts from deteriorating into high churn probability."

    else:
        priority = "Low"
        explanation = (
            f"{cust_name} is in good standing with a low churn probability of {prob}%. Engagement ({engagement}/100) and tenure indicate loyalty."
        )
        action = (
            "Deliver a personalized loyalty appreciation note and invite the customer to join our beta feature advisory program."
        )
        reason = "Reinforces positive brand advocacy and long-term brand loyalty."

    return {
        "customer_id": customer.get("customer_id"),
        "explanation": explanation,
        "recommended_action": action,
        "reason": reason,
        "priority": priority,
        "source": "rule_based",
    }


def generate_rule_based_message(
    customer: dict[str, Any],
    prediction: dict[str, Any],
    tone: str = "Professional",
) -> dict[str, Any]:
    """Generate tone-adjusted customer outreach message deterministically."""
    name = customer.get("customer_name", "Valued Customer")
    plan = customer.get("subscription_type", "Standard")
    risk_level = prediction.get("risk_level", "LOW")

    if tone == "Friendly":
        if risk_level == "HIGH":
            msg = (
                f"Hi {name}! 😊 We noticed you might not be getting the smoothest experience with your {plan} plan lately. "
                "We’d love to help! How about a quick chat or a special discount on an annual upgrade so you can enjoy all features worry-free? Let us know!"
            )
        else:
            msg = (
                f"Hi {name}! Hope you're having a wonderful week! We just wanted to say thank you for being with us on the {plan} plan. "
                "If there's ever anything we can do to make your experience even better, we're just a message away! 🌟"
            )

    elif tone == "Empathetic":
        if risk_level == "HIGH":
            msg = (
                f"Dear {name}, we know your time and experience are valuable, and we apologize if anything about your service has fallen short of your expectations. "
                "Your satisfaction means everything to us. Please let us know how we can make things right, and our dedicated support leads will assist you immediately."
            )
        else:
            msg = (
                f"Dear {name}, thank you for your ongoing partnership. We appreciate your trust in us and are always here to listen and support your evolving needs."
            )

    elif tone == "Short":
        if risk_level == "HIGH":
            msg = f"Hi {name}, we have exclusive savings and dedicated support ready for your account. Please reach out to explore your custom renewal options."
        else:
            msg = f"Hi {name}, thank you for choosing us! Let us know if you need any assistance with your {plan} service."

    else:  # Professional (Default)
        if risk_level == "HIGH":
            msg = (
                f"Dear {name}, as part of our commitment to delivering optimal service quality on your {plan} plan, "
                "we would like to review your current configuration and discuss customized renewal incentives designed to maximize your value and efficiency."
            )
        else:
            msg = (
                f"Dear {name}, we value your business and ongoing engagement with our {plan} tier. "
                "Our team remains at your disposal should you wish to review upcoming platform enhancements or optimize your current workflow."
            )

    return {
        "customer_id": customer.get("customer_id"),
        "message": msg,
        "tone": tone,
        "source": "rule_based",
    }


def generate_rule_based_chat_answer(
    customer: dict[str, Any],
    prediction: dict[str, Any],
    question: str,
) -> str:
    """Generate grounded answer to common questions about customer churn data."""
    q_lower = question.lower()
    name = customer.get("customer_name", "Customer")
    prob = prediction.get("churn_probability", 0.0)
    risk = prediction.get("risk_level", "LOW")
    tenure = customer.get("tenure", 0)
    charges = customer.get("monthly_charges", 0.0)
    contract = customer.get("contract_type", "Month-to-Month")
    complaints = customer.get("complaint_count", 0)
    engagement = customer.get("engagement_score", 0.0)
    top_factors = prediction.get("top_factors", [])

    if "why" in q_lower or "reason" in q_lower or "risk" in q_lower:
        factors_str = "; ".join([
            f"{f['feature']} ({f['direction'].replace('_', ' ')})" for f in top_factors[:3]
        ])
        return (
            f"{name} is classified as {risk} risk with a {prob}% churn probability. "
            f"The primary model drivers influencing this score are: {factors_str}."
        )
    elif "contract" in q_lower or "term" in q_lower:
        return f"{name} is currently on a {contract} contract with a tenure of {tenure} month(s)."
    elif "charge" in q_lower or "bill" in q_lower or "cost" in q_lower or "price" in q_lower:
        return f"{name}'s monthly charges are ${charges:.2f}, totaling ${customer.get('total_charges', 0.0):.2f} over {tenure} months."
    elif "complaint" in q_lower or "support" in q_lower:
        return f"{name} has logged {complaints} formal complaint(s) and {customer.get('support_calls', 0)} support call(s)."
    elif "engagement" in q_lower or "usage" in q_lower:
        return f"{name} has an engagement score of {engagement}/100 and averages {customer.get('usage_frequency', 0.0)} sessions per week."
    else:
        return (
            f"Summary for {name}: {risk} Risk ({prob}% churn probability). "
            f"Tenure: {tenure} months, Plan: {customer.get('subscription_type')}, Contract: {contract}, "
            f"Monthly: ${charges:.2f}, Engagement: {engagement}/100."
        )
