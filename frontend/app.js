/**
 * ChurnIQ — Frontend Application Logic
 * Implements tab switching, sample presets, prediction, AI orchestration,
 * grounded chat assistant, and XSS escaping (Rule 9).
 */

import { api } from "./api.js";

// Global application state
let currentCustomer = null;
let currentPrediction = null;

// Preset sample profiles for 1-click evaluation
const PRESETS = {
    high: {
        customer_id: "C90001",
        customer_name: "Carlos HighRisk",
        gender: "Male",
        age: 45,
        tenure: 2,
        subscription_type: "Basic",
        contract_type: "Month-to-Month",
        monthly_charges: 95.0,
        total_charges: 190.0,
        payment_method: "Credit Card",
        payment_status: "Failed",
        usage_frequency: 4.0,
        service_usage: 15.0,
        complaint_count: 3,
        support_calls: 5,
        engagement_score: 15.0,
    },
    low: {
        customer_id: "C90002",
        customer_name: "Linda LowRisk",
        gender: "Female",
        age: 55,
        tenure: 48,
        subscription_type: "Premium",
        contract_type: "Two-Year",
        monthly_charges: 65.0,
        total_charges: 3120.0,
        payment_method: "Credit Card",
        payment_status: "On Time",
        usage_frequency: 28.0,
        service_usage: 85.0,
        complaint_count: 0,
        support_calls: 0,
        engagement_score: 92.0,
    },
    med: {
        customer_id: "C90003",
        customer_name: "Elena Moderate",
        gender: "Female",
        age: 38,
        tenure: 12,
        subscription_type: "Standard",
        contract_type: "Month-to-Month",
        monthly_charges: 70.0,
        total_charges: 840.0,
        payment_method: "UPI",
        payment_status: "Delayed",
        usage_frequency: 14.0,
        service_usage: 55.0,
        complaint_count: 1,
        support_calls: 2,
        engagement_score: 55.0,
    },
};

// ── DOM Elements ─────────────────────────────────────────────
const navTabs = document.querySelectorAll(".nav-tab");
const viewSections = document.querySelectorAll(".view-section");
const predictForm = document.getElementById("predict-form");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const chatMessages = document.getElementById("chat-messages");
const btnGenerateMsg = document.getElementById("btn-generate-msg");
const msgToneSelect = document.getElementById("msg-tone-select");

// ── Tab Navigation ───────────────────────────────────────────
function switchView(viewName) {
    navTabs.forEach((tab) => {
        tab.classList.toggle("active", tab.dataset.view === viewName);
    });
    viewSections.forEach((section) => {
        section.classList.toggle("active", section.id === `view-${viewName}`);
    });

    if (viewName === "dashboard") loadDashboard();
    if (viewName === "metrics") loadMetrics();
}

navTabs.forEach((tab) => {
    tab.addEventListener("click", () => switchView(tab.dataset.view));
});

// ── Preset Populator ─────────────────────────────────────────
function populateForm(data) {
    Object.entries(data).forEach(([key, val]) => {
        const input = document.getElementById(`inp-${key}`);
        if (input) input.value = val;
    });
}

document.getElementById("btn-load-high").addEventListener("click", () => populateForm(PRESETS.high));
document.getElementById("btn-load-low").addEventListener("click", () => populateForm(PRESETS.low));
document.getElementById("btn-load-med").addEventListener("click", () => populateForm(PRESETS.med));

// ── Prediction Form Submission ───────────────────────────────
predictForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = document.getElementById("btn-submit-predict");
    btn.disabled = true;
    btn.textContent = "Predicting...";

    const payload = {
        customer_id: document.getElementById("inp-customer_id").value.trim(),
        customer_name: document.getElementById("inp-customer_name").value.trim(),
        gender: document.getElementById("inp-gender").value,
        age: parseInt(document.getElementById("inp-age").value, 10),
        tenure: parseInt(document.getElementById("inp-tenure").value, 10),
        subscription_type: document.getElementById("inp-subscription_type").value,
        contract_type: document.getElementById("inp-contract_type").value,
        monthly_charges: parseFloat(document.getElementById("inp-monthly_charges").value),
        total_charges: parseFloat(document.getElementById("inp-total_charges").value),
        payment_method: document.getElementById("inp-payment_method").value,
        payment_status: document.getElementById("inp-payment_status").value,
        usage_frequency: parseFloat(document.getElementById("inp-usage_frequency").value),
        service_usage: parseFloat(document.getElementById("inp-service_usage").value),
        complaint_count: parseInt(document.getElementById("inp-complaint_count").value, 10),
        support_calls: parseInt(document.getElementById("inp-support_calls").value, 10),
        engagement_score: parseFloat(document.getElementById("inp-engagement_score").value),
    };

    try {
        const result = await api.predictCustomer(payload);
        currentCustomer = payload;
        currentPrediction = result;
        renderPredictionResult(result, payload);
        switchView("result");
    } catch (err) {
        alert("Prediction error: " + err.message);
    } finally {
        btn.disabled = false;
        btn.textContent = "Predict Churn Risk";
    }
});

// ── Render Prediction Result & Retention Studio ──────────────
function renderPredictionResult(res, cust) {
    // 1. Gauge & Risk Badge
    const prob = res.churn_probability;
    const gaugeFill = document.getElementById("gauge-fill");
    const circumference = 408.4;
    const offset = circumference - (prob / 100) * circumference;
    gaugeFill.style.strokeDashoffset = offset;

    const riskBadge = document.getElementById("res-risk-badge");
    riskBadge.textContent = `${res.risk_level} RISK`;
    riskBadge.className = "badge";

    if (res.risk_level === "HIGH") {
        gaugeFill.style.stroke = "var(--high-risk)";
        riskBadge.classList.add("badge-high");
    } else if (res.risk_level === "MEDIUM") {
        gaugeFill.style.stroke = "var(--med-risk)";
        riskBadge.classList.add("badge-medium");
    } else {
        gaugeFill.style.stroke = "var(--low-risk)";
        riskBadge.classList.add("badge-low");
    }

    document.getElementById("res-prob").textContent = `${prob}%`;
    document.getElementById("res-cust-name").textContent = cust.customer_name;
    document.getElementById("res-cust-id").textContent = `ID: ${cust.customer_id} | Plan: ${cust.subscription_type}`;
    document.getElementById("res-pred-label").textContent = `Prediction: ${res.prediction}`;
    document.getElementById("res-model-name").textContent = `Model: ${res.model}`;

    // 2. Render Top 5 SHAP Drivers (Rule 9: escape via textContent)
    const shapContainer = document.getElementById("shap-factors-container");
    shapContainer.innerHTML = "";

    const maxImpact = Math.max(...res.top_factors.map((f) => f.impact), 1.0);

    res.top_factors.forEach((factor) => {
        const row = document.createElement("div");
        row.className = "factor-row";

        const header = document.createElement("div");
        header.className = "factor-header";

        const label = document.createElement("span");
        label.textContent = factor.feature;

        const val = document.createElement("span");
        const dirSymbol = factor.direction === "increases_risk" ? "+▲" : "-▼";
        val.textContent = `${dirSymbol} ${factor.impact.toFixed(4)}`;
        val.style.color = factor.direction === "increases_risk" ? "var(--high-risk)" : "var(--low-risk)";
        val.style.fontWeight = "600";

        header.appendChild(label);
        header.appendChild(val);

        const track = document.createElement("div");
        track.className = "bar-track";

        const fill = document.createElement("div");
        fill.className = `bar-fill ${factor.direction === "increases_risk" ? "bar-increase" : "bar-decrease"}`;
        const pctWidth = Math.min((factor.impact / maxImpact) * 100, 100);
        fill.style.width = `${pctWidth}%`;

        track.appendChild(fill);
        row.appendChild(header);
        row.appendChild(track);
        shapContainer.appendChild(row);
    });

    // 3. Reset and Trigger AI Recommendation & Message
    fetchAiRecommendation(cust.customer_id);
    fetchAiMessage(cust.customer_id, msgToneSelect.value);

    // 4. Reset Chat Box
    chatMessages.innerHTML = `
        <div class="chat-bubble chat-assistant">
            Hello! I am grounded in ${escapeHtml(cust.customer_name)}'s data. Ask me anything about their ${prob}% churn risk, contract, complaints, or retention ideas.
        </div>
    `;
}

// ── AI Recommendation Loader ─────────────────────────────────
async function fetchAiRecommendation(customerId) {
    const container = document.getElementById("rec-content-container");
    const sourceBadge = document.getElementById("rec-source-badge");
    container.innerHTML = "<p style='color: var(--text-muted);'>Generating personalized retention strategy...</p>";
    sourceBadge.textContent = "Source: Loading...";

    try {
        const rec = await api.getAiRecommendation(customerId);
        sourceBadge.textContent = `Source: ${rec.source.replace("_", " ")}`;
        sourceBadge.className = `badge badge-source ${rec.source === "rule_based" ? "badge-medium" : ""}`;

        container.innerHTML = "";

        const explP = document.createElement("p");
        explP.style.marginBottom = "0.75rem";
        explP.style.fontSize = "0.95rem";
        explP.textContent = rec.explanation;

        const actionBox = document.createElement("div");
        actionBox.style.background = "var(--bg)";
        actionBox.style.padding = "0.85rem";
        actionBox.style.borderRadius = "var(--radius)";
        actionBox.style.border = "1px solid var(--border)";
        actionBox.style.marginBottom = "0.75rem";

        const actionTitle = document.createElement("div");
        actionTitle.style.fontWeight = "600";
        actionTitle.style.color = "var(--primary)";
        actionTitle.style.marginBottom = "0.3rem";
        actionTitle.textContent = `Recommended Action (Priority: ${rec.priority})`;

        const actionText = document.createElement("p");
        actionText.style.fontSize = "0.9rem";
        actionText.textContent = rec.recommended_action;

        actionBox.appendChild(actionTitle);
        actionBox.appendChild(actionText);

        const reasonP = document.createElement("p");
        reasonP.style.fontSize = "0.85rem";
        reasonP.style.color = "var(--text-muted)";
        reasonP.textContent = `Rationale: ${rec.reason}`;

        container.appendChild(explP);
        container.appendChild(actionBox);
        container.appendChild(reasonP);
    } catch (err) {
        container.innerHTML = `<p style="color: var(--high-risk);">Failed to load recommendation: ${escapeHtml(err.message)}</p>`;
    }
}

// ── AI Message Loader ────────────────────────────────────────
async function fetchAiMessage(customerId, tone) {
    const box = document.getElementById("msg-content-box");
    const sourceBadge = document.getElementById("msg-source-badge");
    box.textContent = "Drafting outreach message...";
    sourceBadge.textContent = "Source: Loading...";

    try {
        const res = await api.getAiMessage(customerId, tone);
        sourceBadge.textContent = `Source: ${res.source.replace("_", " ")}`;
        box.textContent = res.message;
    } catch (err) {
        box.textContent = "Failed to draft message: " + err.message;
    }
}

btnGenerateMsg.addEventListener("click", () => {
    if (currentCustomer) {
        fetchAiMessage(currentCustomer.customer_id, msgToneSelect.value);
    }
});

msgToneSelect.addEventListener("change", () => {
    if (currentCustomer) {
        fetchAiMessage(currentCustomer.customer_id, msgToneSelect.value);
    }
});

// ── Grounded Chat Assistant ──────────────────────────────────
chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = chatInput.value.trim();
    if (!q || !currentCustomer) return;

    // Append user bubble (Rule 9: textContent)
    const userBubble = document.createElement("div");
    userBubble.className = "chat-bubble chat-user";
    userBubble.textContent = q;
    chatMessages.appendChild(userBubble);
    chatInput.value = "";

    // Temporary loading bubble
    const tempBubble = document.createElement("div");
    tempBubble.className = "chat-bubble chat-assistant";
    tempBubble.textContent = "Analyzing customer context...";
    chatMessages.appendChild(tempBubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
        const chatRes = await api.askAiChat(currentCustomer.customer_id, q);
        tempBubble.textContent = chatRes.answer;
        if (chatRes.source === "rule_based") {
            const badge = document.createElement("span");
            badge.style.display = "block";
            badge.style.fontSize = "0.7rem";
            badge.style.color = "var(--text-muted)";
            badge.style.marginTop = "0.3rem";
            badge.textContent = "(Generated via Rule-Based fallback)";
            tempBubble.appendChild(badge);
        }
    } catch (err) {
        tempBubble.textContent = "Error: " + err.message;
    }
    chatMessages.scrollTop = chatMessages.scrollHeight;
});

// ── Dashboard Loader ─────────────────────────────────────────
async function loadDashboard() {
    try {
        const data = await api.getDashboardSummary();
        document.getElementById("dash-total-scored").textContent = data.total_scored;
        document.getElementById("dash-avg-prob").textContent = `${data.average_probability.toFixed(1)}%`;
        document.getElementById("dash-count-low").textContent = `Low: ${data.risk_counts.LOW}`;
        document.getElementById("dash-count-med").textContent = `Med: ${data.risk_counts.MEDIUM}`;
        document.getElementById("dash-count-high").textContent = `High: ${data.risk_counts.HIGH}`;

        const tbody = document.getElementById("dash-recent-tbody");
        tbody.innerHTML = "";

        if (data.recent_predictions.length === 0) {
            tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-muted);">No predictions recorded yet. Score a customer first!</td></tr>`;
            return;
        }

        data.recent_predictions.forEach((row) => {
            const tr = document.createElement("tr");

            const tdDate = document.createElement("td");
            tdDate.textContent = row.created_at;

            const tdId = document.createElement("td");
            tdId.textContent = row.customer_id;
            tdId.style.fontWeight = "600";

            const tdName = document.createElement("td");
            tdName.textContent = row.customer_name;

            const tdProb = document.createElement("td");
            tdProb.textContent = `${row.churn_probability}%`;

            const tdRisk = document.createElement("td");
            const badge = document.createElement("span");
            badge.className = `badge badge-${row.risk_level.toLowerCase()}`;
            badge.textContent = row.risk_level;
            tdRisk.appendChild(badge);

            const tdPred = document.createElement("td");
            tdPred.textContent = row.prediction;

            tr.appendChild(tdDate);
            tr.appendChild(tdId);
            tr.appendChild(tdName);
            tr.appendChild(tdProb);
            tr.appendChild(tdRisk);
            tr.appendChild(tdPred);
            tbody.appendChild(tr);
        });
    } catch (err) {
        console.error("Dashboard load failed:", err);
    }
}

// ── Model Metrics Loader ─────────────────────────────────────
async function loadMetrics() {
    try {
        const metrics = await api.getModelMetrics();
        document.getElementById("metric-accuracy").textContent = `${(metrics.accuracy * 100).toFixed(1)}%`;
        document.getElementById("metric-roc-auc").textContent = metrics.roc_auc.toFixed(4);
        document.getElementById("metric-f1").textContent = `${(metrics.f1_score * 100).toFixed(1)}%`;
        document.getElementById("metric-p-r").textContent = `Precision: ${(metrics.precision * 100).toFixed(1)}% | Recall: ${(metrics.recall * 100).toFixed(1)}%`;

        const cm = metrics.confusion_matrix;
        document.getElementById("cm-tn").textContent = cm.true_negatives;
        document.getElementById("cm-fp").textContent = cm.false_positives;
        document.getElementById("cm-fn").textContent = cm.false_negatives;
        document.getElementById("cm-tp").textContent = cm.true_positives;

        const container = document.getElementById("global-importance-container");
        container.innerHTML = "";

        const maxImp = Math.max(...metrics.global_feature_importance.map((f) => f.importance), 0.01);

        metrics.global_feature_importance.forEach((item) => {
            const row = document.createElement("div");
            row.className = "factor-row";

            const header = document.createElement("div");
            header.className = "factor-header";

            const label = document.createElement("span");
            label.textContent = item.feature;

            const val = document.createElement("span");
            val.textContent = (item.importance * 100).toFixed(2) + "%";
            val.style.fontWeight = "600";

            header.appendChild(label);
            header.appendChild(val);

            const track = document.createElement("div");
            track.className = "bar-track";

            const fill = document.createElement("div");
            fill.className = "bar-fill";
            fill.style.background = "var(--primary)";
            fill.style.width = `${(item.importance / maxImp) * 100}%`;

            track.appendChild(fill);
            row.appendChild(header);
            row.appendChild(track);
            container.appendChild(row);
        });
    } catch (err) {
        console.error("Metrics load failed:", err);
    }
}

// ── Utility Helper (Rule 9) ──────────────────────────────────
function escapeHtml(str) {
    if (!str) return "";
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

// Initial load
loadDashboard();
