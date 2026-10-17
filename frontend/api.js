/**
 * ChurnIQ — Frontend API Client
 * Clean fetch wrappers for all 7 backend REST endpoints.
 */

const API_BASE = "http://127.0.0.1:8000";

export const api = {
    async getHealth() {
        const res = await fetch(`${API_BASE}/health`);
        return res.json();
    },

    async predictCustomer(customerData) {
        const res = await fetch(`${API_BASE}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(customerData),
        });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail ? JSON.stringify(err.detail) : "Prediction failed");
        }
        return res.json();
    },

    async getDashboardSummary() {
        const res = await fetch(`${API_BASE}/dashboard/summary`);
        if (!res.ok) throw new Error("Failed to load dashboard summary");
        return res.json();
    },

    async getModelMetrics() {
        const res = await fetch(`${API_BASE}/model/metrics`);
        if (!res.ok) throw new Error("Failed to load model metrics");
        return res.json();
    },

    async getAiRecommendation(customerId) {
        const res = await fetch(`${API_BASE}/ai/recommendation`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ customer_id: customerId }),
        });
        if (!res.ok) throw new Error("Failed to generate AI recommendation");
        return res.json();
    },

    async getAiMessage(customerId, tone = "Professional") {
        const res = await fetch(`${API_BASE}/ai/message`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ customer_id: customerId, tone }),
        });
        if (!res.ok) throw new Error("Failed to generate outreach message");
        return res.json();
    },

    async askAiChat(customerId, question) {
        const res = await fetch(`${API_BASE}/ai/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ customer_id: customerId, question }),
        });
        if (!res.ok) throw new Error("Failed to get answer from assistant");
        return res.json();
    },
};
