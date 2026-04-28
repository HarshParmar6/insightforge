import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
import textwrap
import re
import numpy as np
from datetime import date

# ============================================================
# InsightForge V9.0 - Agentic Industrial AI Decision OS
# Clean copy-paste Streamlit app
# ============================================================

st.set_page_config(page_title="InsightForge", page_icon="🔥", layout="wide")
MODEL_NAME = "gpt-4o-mini"
api_key = st.secrets.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=api_key) if api_key else None

# ---------------- SESSION STATE ----------------
for key, value in {
    "ai_text": "",
    "action_text": "",
    "executive_text": "",
    "chat_history": [],
    "last_dataset_signature": "",
    "use_demo_data": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ---------------- CSS ----------------
st.markdown("""
<style>
.stApp {background: radial-gradient(circle at top left, #E0F2FE 0%, #F8FAFC 34%, #FFFFFF 100%);}
.block-container {padding-top: 1.25rem; padding-bottom: 3rem;}
@keyframes floatCard {0% {transform: translateY(0px);} 50% {transform: translateY(-10px);} 100% {transform: translateY(0px);}}
.hero-shell {position: relative; overflow: hidden; background: linear-gradient(135deg, #020617 0%, #0F172A 45%, #1E3A8A 100%); color:white; padding:48px; border-radius:32px; box-shadow:0 24px 70px rgba(15,23,42,0.34); margin-bottom:24px;}
.hero-shell:before {content:""; position:absolute; width:390px; height:390px; right:-140px; top:-140px; border-radius:50%; background:radial-gradient(circle, rgba(96,165,250,0.45), rgba(37,99,235,0.05));}
.hero-grid {position:relative; z-index:2; display:grid; grid-template-columns:1.28fr 0.72fr; gap:28px; align-items:center;}
.hero-badge {display:inline-block; background:rgba(37,99,235,0.95); padding:8px 16px; border-radius:999px; font-size:13px; font-weight:800; margin-bottom:18px;}
.logo-mark {display:inline-flex; width:58px; height:58px; align-items:center; justify-content:center; border-radius:18px; background:linear-gradient(135deg,#F97316,#EF4444); box-shadow:0 10px 24px rgba(249,115,22,0.35); font-size:32px; margin-right:14px;}
.hero-title {font-size:54px; font-weight:950; line-height:1.05; margin-bottom:12px;}
.hero-title span {color:#60A5FA;}
.hero-subtitle {font-size:22px; color:#E5E7EB; margin-bottom:14px;}
.hero-text {font-size:16px; color:#CBD5E1; line-height:1.65;}
.floating-panel {background:rgba(255,255,255,0.11); border:1px solid rgba(255,255,255,0.22); border-radius:26px; padding:24px; animation:floatCard 4s ease-in-out infinite;}
.panel-row {display:flex; justify-content:space-between; font-size:14px; margin-bottom:16px;}
.panel-value {font-weight:900; color:#93C5FD;}
.trust-strip {display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:24px;}
.trust-pill {background:rgba(255,255,255,0.93); border:1px solid #DBEAFE; border-radius:999px; padding:12px 16px; text-align:center; color:#1E3A8A; font-weight:800; font-size:14px; box-shadow:0 8px 22px rgba(15,23,42,0.06);}
.feature-strip {display:grid; grid-template-columns:repeat(4,1fr); gap:18px; margin-bottom:30px;}
.feature-card {background:white; border:1px solid #E5E7EB; border-radius:24px; padding:24px; box-shadow:0 10px 28px rgba(15,23,42,0.08);}
.feature-icon {font-size:32px; margin-bottom:12px;}
.feature-title {font-weight:900; font-size:17px; color:#111827; margin-bottom:7px;}
.feature-text {color:#6B7280; font-size:14px; line-height:1.45;}
.section-title {font-size:30px; font-weight:950; color:#111827; margin-top:18px; margin-bottom:8px;}
.section-subtitle {color:#6B7280; font-size:15px; margin-bottom:18px;}
.kpi-card {background:white; border-left:7px solid #2563EB; border-radius:20px; padding:20px; margin-bottom:16px; box-shadow:0 8px 24px rgba(15,23,42,0.08);}
.kpi-title {font-size:13px; color:#6B7280; margin-bottom:6px;}
.kpi-value {font-size:27px; font-weight:900; color:#111827;}
.kpi-subtitle {font-size:12px; color:#9CA3AF; margin-top:4px;}
.info-card,.fit-card,.ai-card,.risk-card,.consulting-card {border-radius:22px; padding:24px; margin-bottom:22px; box-shadow:0 8px 24px rgba(15,23,42,0.08);}
.info-card {background:linear-gradient(135deg,#EEF2FF,#FFFFFF); border:1px solid #C7D2FE;}
.fit-card {background:linear-gradient(135deg,#F0F9FF,#FFFFFF); border:1px solid #BAE6FD;}
.ai-card {background:linear-gradient(135deg,#F5F3FF,#FFFFFF); border:1px solid #DDD6FE;}
.risk-card {background:linear-gradient(135deg,#FEF2F2,#FFFFFF); border:1px solid #FECACA;}
.consulting-card {background:linear-gradient(135deg,#EFF6FF,#FFFFFF); border:1px solid #BFDBFE;}
.footer-card {background:#0F172A; color:#CBD5E1; padding:24px; border-radius:20px; margin-top:30px;}
@media (max-width:900px){.hero-grid,.feature-strip,.trust-strip{grid-template-columns:1fr;} .hero-title{font-size:40px;}}
</style>
""", unsafe_allow_html=True)

# ---------------- UI HELPERS ----------------
def kpi_card(title, value, subtitle):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def level_from_score(score):
    if score >= 75:
        return "High"
    if score >= 45:
        return "Medium"
    return "Low"


def classify_risk(score):
    if score >= 75:
        return "High"
    if score >= 50:
        return "Medium"
    return "Low"

# ---------------- DEMO DATASET ----------------
@st.cache_data
def generate_demo_dataset():
    np.random.seed(42)
    n = 1200
    machine_id = np.random.choice(["M1", "M2", "M3", "M4", "M5"], size=n)
    scenario_type = np.random.choice(["normal", "high_load", "tool_wear", "minor_fault"], size=n, p=[0.55, 0.20, 0.15, 0.10])
    machine_state = np.random.choice(["running", "idle", "maintenance", "fault"], size=n, p=[0.68, 0.16, 0.08, 0.08])
    material_type = np.random.choice(["aluminum", "steel", "titanium"], size=n, p=[0.45, 0.40, 0.15])
    vibration = np.random.normal(1.6, 0.45, n)
    temperature = np.random.normal(72, 9, n)
    tool_wear = np.random.gamma(2.4, 2.1, n)
    spindle_load = np.random.normal(62, 13, n)
    power_consumption = spindle_load * np.random.normal(1.85, 0.18, n)
    cycle_time = np.random.normal(118, 18, n)
    queue_length = np.random.poisson(4, n)
    fault_boost = np.where(machine_state == "fault", 0.18, 0)
    load_boost = np.where(scenario_type == "high_load", 0.09, 0)
    wear_boost = np.where(scenario_type == "tool_wear", 0.12, 0)
    breakdown_probability = np.clip(0.06 + vibration * 0.04 + tool_wear * 0.012 + load_boost + wear_boost + fault_boost, 0, 0.95)
    defect_probability = np.clip(0.04 + tool_wear * 0.01 + vibration * 0.025 + np.where(material_type == "titanium", 0.07, 0), 0, 0.9)
    downtime = np.clip(np.random.normal(8, 4, n) + breakdown_probability * 45 + np.where(machine_state == "fault", 25, 0), 0, None)
    health_score = np.clip(np.random.normal(86, 8, n) - tool_wear * 1.2 - breakdown_probability * 25, 20, 100)
    return pd.DataFrame({
        "machine_id": machine_id,
        "scenario_type": scenario_type,
        "machine_state": machine_state,
        "material_type": material_type,
        "health_score": health_score.round(2),
        "vibration": vibration.round(3),
        "temperature": temperature.round(2),
        "tool_wear": tool_wear.round(2),
        "spindle_load": spindle_load.round(2),
        "power_consumption": power_consumption.round(2),
        "cycle_time": cycle_time.round(2),
        "queue_length": queue_length,
        "breakdown_probability": breakdown_probability.round(3),
        "defect_probability": defect_probability.round(3),
        "downtime": downtime.round(2),
        "job_completed": np.random.choice([0, 1], size=n, p=[0.12, 0.88]),
    })

# ---------------- DATA HELPERS ----------------
def detect_sensitive_columns(df):
    patterns = ["name", "email", "phone", "mobile", "address", "ssn", "dob", "birth", "account", "customer", "client", "employee", "supplier", "vendor", "price", "cost", "salary", "revenue", "profit", "id"]
    return [col for col in df.columns if any(re.search(p, str(col).lower()) for p in patterns)]


def find_column(df, possible_names):
    lower_map = {str(c).lower(): c for c in df.columns}
    for name in possible_names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    for col in df.columns:
        col_lower = str(col).lower()
        for name in possible_names:
            if name.lower() in col_lower:
                return col
    return None


def get_group_candidates(df):
    preferred = ["machine_id", "machine_state", "scenario_type", "material_type", "job_type", "supplier", "vendor", "line", "cell", "station", "plant", "site"]
    candidates = []
    for col in df.columns:
        unique_count = df[col].nunique(dropna=True)
        if any(p in str(col).lower() for p in preferred) and 1 < unique_count <= 100:
            candidates.append(col)
    if not candidates:
        for col in df.columns:
            unique_count = df[col].nunique(dropna=True)
            if df[col].dtype == "object" and 1 < unique_count <= 50:
                candidates.append(col)
    return candidates


def calculate_decision_readiness(df, numeric_cols, missing_values, duplicate_rows):
    total_cells = df.shape[0] * df.shape[1]
    if total_cells == 0:
        return 0
    missing_penalty = min(35, (missing_values / total_cells) * 100)
    duplicate_penalty = min(20, (duplicate_rows / max(df.shape[0], 1)) * 100)
    numeric_penalty = 0 if numeric_cols else 25
    size_penalty = 0 if df.shape[0] >= 30 else 15
    return round(max(0, min(100, 100 - missing_penalty - duplicate_penalty - numeric_penalty - size_penalty)), 1)

# ---------------- DATASET FIT ENGINE ----------------
def problem_column_requirements(problem_type):
    req = {
        "Reduce Downtime": {
            "Downtime field": ["downtime", "down_time", "stoppage_time", "idle_time"],
            "Machine identifier": ["machine_id", "machine", "asset_id", "equipment_id"],
            "Machine state/status": ["machine_state", "state", "status"],
            "Breakdown/failure probability": ["breakdown_probability", "failure_probability", "failure_risk"],
            "Cycle time": ["cycle_time", "processing_time"],
            "Queue/backlog": ["queue_length", "queue", "backlog"],
            "Health score": ["health_score", "machine_health"],
            "Vibration": ["vibration", "vibration_level"],
            "Temperature": ["temperature", "temp"],
            "Tool wear": ["tool_wear", "wear"],
        },
        "Improve Quality": {
            "Defect field": ["defect_probability", "defect_rate", "defects", "scrap_rate"],
            "Dimensional error": ["dimensional_error", "error", "deviation"],
            "Material type": ["material_type", "material"],
            "Job/product type": ["job_type", "product_type", "part_type"],
            "Process condition": ["spindle_speed", "feed_rate", "depth_of_cut", "temperature", "vibration"],
            "Tool wear": ["tool_wear", "wear"],
            "Machine identifier": ["machine_id", "machine", "asset_id"],
        },
        "Predict Maintenance Risk": {
            "Health score": ["health_score", "machine_health"],
            "Breakdown/failure probability": ["breakdown_probability", "failure_probability", "failure_risk"],
            "Maintenance history": ["maintenance", "service_history", "maintenance_flag", "repair"],
            "Tool wear": ["tool_wear", "wear"],
            "Vibration": ["vibration", "vibration_level"],
            "Temperature": ["temperature", "temp"],
            "Load": ["spindle_load", "machine_load", "load"],
            "Downtime": ["downtime", "down_time", "stoppage_time"],
            "Machine identifier": ["machine_id", "machine", "asset_id"],
        },
        "Reduce Cost": {
            "Power/energy": ["power_consumption", "energy_consumption", "power", "energy"],
            "Downtime": ["downtime", "down_time", "stoppage_time"],
            "Cycle time": ["cycle_time", "processing_time"],
            "Tool wear": ["tool_wear", "wear"],
            "Defect/scrap": ["defect_probability", "defect_rate", "scrap_rate"],
            "Machine load": ["spindle_load", "machine_load", "load"],
            "Cost field": ["cost", "unit_cost", "operating_cost"],
        },
        "Improve Delivery Performance": {
            "Tardiness/delay": ["tardiness", "delay", "late", "tardiness_increment"],
            "Job completion": ["job_completed", "completed", "completion_status"],
            "Queue/backlog": ["queue_length", "queue", "backlog"],
            "Cycle time": ["cycle_time", "processing_time"],
            "Downtime": ["downtime", "down_time"],
            "Machine identifier": ["machine_id", "machine", "asset_id"],
        },
        "Identify Bottlenecks": {
            "Queue/backlog": ["queue_length", "queue", "backlog"],
            "Cycle time": ["cycle_time", "processing_time"],
            "Downtime": ["downtime", "down_time"],
            "Job completion": ["job_completed", "completed"],
            "Machine state/status": ["machine_state", "state", "status"],
            "Machine identifier": ["machine_id", "machine", "asset_id"],
            "Line/cell/station": ["line", "cell", "station", "workcenter"],
        },
        "Improve Process Stability": {
            "Vibration": ["vibration", "vibration_level"],
            "Temperature": ["temperature", "temp"],
            "Cycle time": ["cycle_time", "processing_time"],
            "Dimensional error": ["dimensional_error", "error", "deviation"],
            "Process parameters": ["spindle_speed", "feed_rate", "depth_of_cut"],
            "Defect field": ["defect_probability", "defect_rate", "scrap_rate"],
            "Machine load": ["spindle_load", "machine_load", "load"],
        },
        "General Industrial Analysis": {
            "Machine identifier": ["machine_id", "machine", "asset_id"],
            "Downtime": ["downtime", "down_time"],
            "Quality": ["defect_probability", "defect_rate", "dimensional_error"],
            "Maintenance": ["health_score", "tool_wear", "breakdown_probability"],
            "Process": ["cycle_time", "temperature", "vibration"],
            "Delivery": ["queue_length", "tardiness", "job_completed"],
        },
    }
    return req.get(problem_type, req["General Industrial Analysis"])


def calculate_dataset_fit(df, problem_type):
    reqs = problem_column_requirements(problem_type)
    matched, missing = {}, []
    for label, synonyms in reqs.items():
        found = find_column(df, synonyms)
        if found is not None:
            matched[label] = found
        else:
            missing.append(label)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    group_cols = get_group_candidates(df)
    score = (len(matched) / max(len(reqs), 1)) * 75 + (10 if len(numeric_cols) >= 2 else 0) + (10 if df.shape[0] >= 30 else 0) + (5 if group_cols else 0)
    critical_labels = {
        "Reduce Downtime": ["Downtime field", "Machine state/status", "Breakdown/failure probability", "Cycle time"],
        "Improve Quality": ["Defect field", "Dimensional error", "Process condition"],
        "Predict Maintenance Risk": ["Health score", "Breakdown/failure probability", "Tool wear", "Vibration"],
        "Reduce Cost": ["Power/energy", "Downtime", "Cycle time", "Cost field"],
        "Improve Delivery Performance": ["Tardiness/delay", "Job completion", "Queue/backlog"],
        "Identify Bottlenecks": ["Queue/backlog", "Cycle time", "Machine identifier", "Line/cell/station"],
        "Improve Process Stability": ["Vibration", "Temperature", "Cycle time", "Process parameters"],
        "General Industrial Analysis": [],
    }
    critical = critical_labels.get(problem_type, [])
    if critical and not any(label in matched for label in critical):
        score = min(score, 55)
    score = round(max(0, min(100, score)), 1)
    confidence = level_from_score(score)
    context = f"""
Dataset Fit Assessment:
Selected problem: {problem_type}
Dataset Fit Score: {score}/100
Analysis Confidence: {confidence}

Matched fields:
{matched if matched else 'No strong problem-specific fields matched.'}

Missing recommended fields:
{missing if missing else 'No major recommended fields missing.'}

Interpretation:
- High fit means the uploaded data is suitable for the selected industrial diagnosis.
- Medium fit means the analysis is useful but should be interpreted carefully.
- Low fit means the dataset does not contain enough relevant fields for reliable diagnosis.
"""
    return score, confidence, matched, missing, context

# ---------------- AI READINESS ENGINE ----------------
def score_requirement_group(df, requirements, max_match_score=80):
    matched, missing = {}, []
    for label, synonyms in requirements.items():
        found = find_column(df, synonyms)
        if found is not None:
            matched[label] = found
        else:
            missing.append(label)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    score = (len(matched) / max(len(requirements), 1)) * max_match_score + (10 if len(numeric_cols) >= 3 else 5 if numeric_cols else 0) + (10 if df.shape[0] >= 100 else 5 if df.shape[0] >= 30 else 0)
    score = round(max(0, min(100, score)), 1)
    return score, level_from_score(score), matched, missing


def calculate_ai_readiness(df):
    total_cells = df.shape[0] * df.shape[1]
    missing_values = int(df.isna().sum().sum())
    completeness = 100 if total_cells == 0 else round(100 - (missing_values / total_cells * 100), 1)
    req_sets = {
        "Predictive Maintenance Readiness": {
            "Machine identifier": ["machine_id", "machine", "asset_id"],
            "Health score": ["health_score", "machine_health"],
            "Failure/breakdown label or probability": ["breakdown_probability", "failure_probability", "failure", "failure_risk"],
            "Vibration": ["vibration", "vibration_level"],
            "Temperature": ["temperature", "temp"],
            "Tool wear": ["tool_wear", "wear"],
            "Load": ["spindle_load", "machine_load", "load"],
            "Maintenance history": ["maintenance", "service_history", "maintenance_flag", "repair"],
        },
        "Quality Prediction Readiness": {
            "Defect label or probability": ["defect_probability", "defect_rate", "defects", "scrap_rate", "quality_risk"],
            "Dimensional error": ["dimensional_error", "error", "deviation"],
            "Material": ["material_type", "material"],
            "Machine identifier": ["machine_id", "machine", "asset_id"],
            "Tool wear": ["tool_wear", "wear"],
            "Process parameters": ["spindle_speed", "feed_rate", "depth_of_cut", "temperature", "vibration"],
        },
        "Downtime Diagnosis Readiness": {
            "Downtime": ["downtime", "down_time", "stoppage_time"],
            "Machine state": ["machine_state", "state", "status"],
            "Machine identifier": ["machine_id", "machine", "asset_id"],
            "Cycle time": ["cycle_time", "processing_time"],
            "Queue/backlog": ["queue_length", "queue", "backlog"],
            "Failure probability": ["breakdown_probability", "failure_probability", "failure_risk"],
        },
        "Digital Twin Readiness": {
            "Machine identifier": ["machine_id", "machine", "asset_id"],
            "Machine state": ["machine_state", "state", "status"],
            "Cycle time": ["cycle_time", "processing_time"],
            "Load": ["spindle_load", "machine_load", "load"],
            "Power": ["power_consumption", "energy_consumption", "power"],
            "Temperature": ["temperature", "temp"],
            "Vibration": ["vibration", "vibration_level"],
            "Job/product/material": ["job_type", "product_type", "part_type", "material_type", "material"],
            "Time variable": ["timestamp", "time", "date", "datetime", "time_step"],
        },
    }
    rows = []
    area_scores = {}
    for area, reqs in req_sets.items():
        score, level, matched, missing = score_requirement_group(df, reqs)
        area_scores[area] = score
        matched_text = ", ".join([f"{k}: {v}" for k, v in matched.items()]) if matched else "None"
        missing_text = ", ".join(missing) if missing else "None"
        rows.append({"Readiness Area": area, "Score": score, "Level": level, "Matched Fields": matched_text, "Missing Fields": missing_text})
    data_score = completeness
    if df.shape[0] >= 100 and len(df.select_dtypes(include=["number"]).columns) >= 3:
        data_score = min(100, data_score + 5)
    data_score = round(data_score, 1)
    overall = round(area_scores["Predictive Maintenance Readiness"] * 0.25 + area_scores["Quality Prediction Readiness"] * 0.20 + area_scores["Downtime Diagnosis Readiness"] * 0.25 + area_scores["Digital Twin Readiness"] * 0.20 + data_score * 0.10, 1)
    table = pd.DataFrame([{"Readiness Area": "Overall AI Readiness", "Score": overall, "Level": level_from_score(overall), "Matched Fields": "See area scores", "Missing Fields": "See area scores"}] + rows + [{"Readiness Area": "Data Completeness Readiness", "Score": data_score, "Level": level_from_score(data_score), "Matched Fields": f"Rows: {df.shape[0]}, Columns: {df.shape[1]}", "Missing Fields": f"Missing cells: {missing_values}"}])
    context = f"""
AI Readiness Assessment:
Overall AI Readiness Score: {overall}/100
Overall AI Readiness Level: {level_from_score(overall)}

Readiness Breakdown:
{table.to_string(index=False)}

Interpretation:
- High readiness means the dataset is strong enough for deeper AI modeling or advanced decision support after validation.
- Medium readiness means the dataset can support diagnosis, but predictive modeling should be validated carefully.
- Low readiness means the dataset is not yet ready for reliable AI modeling. More relevant fields should be collected first.
"""
    return overall, level_from_score(overall), table, context

# ---------------- PROBLEM FOCUS ----------------
def get_problem_focus(problem_type):
    return {
        "Reduce Downtime": "Focus on downtime, breakdown probability, cycle time, queue length, machine state, machine health, and maintenance flags.",
        "Improve Quality": "Focus on defect probability, dimensional error, process variation, material type, job type, and abnormal operating conditions.",
        "Predict Maintenance Risk": "Focus on health score, vibration, temperature, spindle load, tool wear, breakdown probability, maintenance flags, and downtime.",
        "Reduce Cost": "Focus on power consumption, cycle time, downtime, tool wear, defect probability, queue length, and resource inefficiency.",
        "Improve Delivery Performance": "Focus on tardiness, queue length, job completion, cycle time, downtime, and production delay signals.",
        "Identify Bottlenecks": "Focus on queue length, cycle time, downtime, job completion, machine state, and process flow delays.",
        "Improve Process Stability": "Focus on variability, outliers, standard deviations, correlations, vibration, temperature, feed rate, spindle speed, defect probability, and dimensional error.",
        "General Industrial Analysis": "Focus on data health, operational risks, performance drivers, quality risks, maintenance signals, and business impact.",
    }.get(problem_type, "Focus on data health and operational risk.")

# ---------------- RISK ENGINE ----------------
def numeric_series(df, col):
    if col is None or col not in df.columns:
        return pd.Series(0, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(0)


def minmax_score(series, inverse=False):
    s = pd.to_numeric(series, errors="coerce")
    med = s.median()
    s = s.fillna(med if pd.notna(med) else 0)
    if s.max() == s.min():
        base = pd.Series(0, index=s.index, dtype=float)
    else:
        base = ((s - s.min()) / (s.max() - s.min())) * 100
    return (100 - base if inverse else base).clip(0, 100)


def safe_weighted_score(index, components):
    valid = [(score, weight) for score, weight in components if score is not None and weight > 0]
    if not valid:
        return pd.Series(0, index=index, dtype=float)
    total_weight = sum(weight for _, weight in valid)
    return (sum(score * weight for score, weight in valid) / total_weight).clip(0, 100)


def compute_risk_engine(df, problem_type):
    risk_df = pd.DataFrame(index=df.index)
    health_col = find_column(df, ["health_score", "machine_health"])
    downtime_col = find_column(df, ["downtime", "down_time", "stoppage_time"])
    breakdown_col = find_column(df, ["breakdown_probability", "failure_probability", "failure_risk"])
    defect_col = find_column(df, ["defect_probability", "defect_rate", "scrap_rate", "quality_risk"])
    vibration_col = find_column(df, ["vibration", "vibration_level"])
    temperature_col = find_column(df, ["temperature", "temp"])
    tool_wear_col = find_column(df, ["tool_wear", "wear"])
    cycle_time_col = find_column(df, ["cycle_time", "processing_time"])
    queue_col = find_column(df, ["queue_length", "queue", "backlog"])
    power_col = find_column(df, ["power_consumption", "energy_consumption", "power"])
    load_col = find_column(df, ["spindle_load", "machine_load", "load"])
    dimensional_error_col = find_column(df, ["dimensional_error", "error", "deviation"])
    tardiness_col = find_column(df, ["tardiness_increment", "tardiness", "delay"])
    state_col = find_column(df, ["machine_state", "state", "status"])

    health_risk = minmax_score(numeric_series(df, health_col), inverse=True) if health_col else None
    downtime_risk = minmax_score(numeric_series(df, downtime_col)) if downtime_col else None
    breakdown_risk = minmax_score(numeric_series(df, breakdown_col)) if breakdown_col else None
    defect_risk = minmax_score(numeric_series(df, defect_col)) if defect_col else None
    vibration_risk = minmax_score(numeric_series(df, vibration_col)) if vibration_col else None
    temperature_risk = minmax_score(numeric_series(df, temperature_col)) if temperature_col else None
    tool_wear_risk = minmax_score(numeric_series(df, tool_wear_col)) if tool_wear_col else None
    cycle_time_risk = minmax_score(numeric_series(df, cycle_time_col)) if cycle_time_col else None
    queue_risk = minmax_score(numeric_series(df, queue_col)) if queue_col else None
    power_risk = minmax_score(numeric_series(df, power_col)) if power_col else None
    load_risk = minmax_score(numeric_series(df, load_col)) if load_col else None
    dimensional_risk = minmax_score(numeric_series(df, dimensional_error_col)) if dimensional_error_col else None
    tardiness_risk = minmax_score(numeric_series(df, tardiness_col)) if tardiness_col else None
    state_risk = None
    if state_col:
        state_map = {"fault": 100, "failed": 100, "breakdown": 100, "maintenance": 70, "idle": 45, "running": 10, "normal": 10}
        state_risk = df[state_col].astype(str).str.lower().map(state_map).fillna(35)

    risk_df["Downtime Risk"] = safe_weighted_score(df.index, [(downtime_risk, 0.35), (breakdown_risk, 0.25), (queue_risk, 0.15), (cycle_time_risk, 0.15), (state_risk, 0.10)])
    risk_df["Maintenance Risk"] = safe_weighted_score(df.index, [(breakdown_risk, 0.25), (health_risk, 0.20), (vibration_risk, 0.18), (tool_wear_risk, 0.18), (temperature_risk, 0.12), (load_risk, 0.07)])
    risk_df["Quality Risk"] = safe_weighted_score(df.index, [(defect_risk, 0.35), (dimensional_risk, 0.25), (vibration_risk, 0.15), (tool_wear_risk, 0.15), (health_risk, 0.10)])
    risk_df["Cost Risk"] = safe_weighted_score(df.index, [(power_risk, 0.30), (downtime_risk, 0.25), (cycle_time_risk, 0.20), (tool_wear_risk, 0.15), (defect_risk, 0.10)])
    risk_df["Delivery Risk"] = safe_weighted_score(df.index, [(tardiness_risk, 0.35), (queue_risk, 0.25), (cycle_time_risk, 0.20), (downtime_risk, 0.20)])
    stability_components = [(score, 1.0) for score in [vibration_risk, temperature_risk, cycle_time_risk, load_risk, power_risk] if score is not None]
    risk_df["Process Stability Risk"] = safe_weighted_score(df.index, stability_components)

    weights_by_problem = {
        "Reduce Downtime": {"Downtime Risk": 0.45, "Maintenance Risk": 0.25, "Process Stability Risk": 0.15, "Delivery Risk": 0.15},
        "Improve Quality": {"Quality Risk": 0.50, "Process Stability Risk": 0.25, "Maintenance Risk": 0.15, "Cost Risk": 0.10},
        "Predict Maintenance Risk": {"Maintenance Risk": 0.55, "Downtime Risk": 0.25, "Process Stability Risk": 0.20},
        "Reduce Cost": {"Cost Risk": 0.45, "Downtime Risk": 0.25, "Quality Risk": 0.15, "Maintenance Risk": 0.15},
        "Improve Delivery Performance": {"Delivery Risk": 0.50, "Downtime Risk": 0.25, "Process Stability Risk": 0.15, "Cost Risk": 0.10},
        "Identify Bottlenecks": {"Downtime Risk": 0.35, "Delivery Risk": 0.30, "Cost Risk": 0.20, "Process Stability Risk": 0.15},
        "Improve Process Stability": {"Process Stability Risk": 0.45, "Quality Risk": 0.25, "Maintenance Risk": 0.20, "Downtime Risk": 0.10},
        "General Industrial Analysis": {"Downtime Risk": 0.25, "Maintenance Risk": 0.25, "Quality Risk": 0.20, "Cost Risk": 0.15, "Delivery Risk": 0.10, "Process Stability Risk": 0.05},
    }
    weights = weights_by_problem.get(problem_type, weights_by_problem["General Industrial Analysis"])
    risk_df["Overall Risk Score"] = sum(risk_df[col] * weight for col, weight in weights.items())
    risk_df["Risk Level"] = risk_df["Overall Risk Score"].apply(classify_risk)
    return risk_df.round(2)


def make_group_risk_table(df, risk_df, group_col):
    if group_col is None or group_col not in df.columns:
        return pd.DataFrame()
    combined = pd.concat([df[[group_col]].copy(), risk_df.copy()], axis=1)
    risk_cols = ["Overall Risk Score", "Downtime Risk", "Maintenance Risk", "Quality Risk", "Cost Risk", "Delivery Risk", "Process Stability Risk"]
    table = combined.groupby(group_col)[risk_cols].mean().round(2)
    table["Records"] = combined.groupby(group_col).size()
    table["Risk Level"] = table["Overall Risk Score"].apply(classify_risk)
    return table.sort_values("Overall Risk Score", ascending=False).reset_index()


def get_problem_target_column(df, problem_type):
    mapping = {
        "Reduce Downtime": ["downtime", "down_time", "stoppage_time"],
        "Improve Quality": ["defect_probability", "defect_rate", "dimensional_error", "scrap_rate"],
        "Predict Maintenance Risk": ["breakdown_probability", "failure_probability", "tool_wear", "health_score"],
        "Reduce Cost": ["power_consumption", "energy_consumption", "cycle_time", "downtime", "cost"],
        "Improve Delivery Performance": ["tardiness_increment", "tardiness", "delay", "queue_length"],
        "Identify Bottlenecks": ["queue_length", "cycle_time", "downtime"],
        "Improve Process Stability": ["vibration", "temperature", "cycle_time", "dimensional_error"],
    }
    return find_column(df, mapping.get(problem_type, []))


def root_cause_finder(df, risk_df, problem_type):
    rows = []
    target_col = get_problem_target_column(df, problem_type)
    if target_col is not None and target_col in df.columns:
        target = pd.to_numeric(df[target_col], errors="coerce")
        for col in df.select_dtypes(include=["number"]).columns:
            if col == target_col:
                continue
            candidate = pd.to_numeric(df[col], errors="coerce")
            if candidate.nunique(dropna=True) <= 1 or target.nunique(dropna=True) <= 1:
                continue
            corr = target.corr(candidate)
            if pd.notna(corr):
                rows.append({"Driver": col, "Type": "Numeric correlation", "Evidence": f"Correlation with {target_col}: {corr:.3f}", "Strength": abs(corr), "Direction": "Positive" if corr > 0 else "Negative"})
        for col in get_group_candidates(df):
            try:
                group_means = df.groupby(col)[target_col].mean(numeric_only=True).sort_values(ascending=False)
                if len(group_means) > 1:
                    top_group = group_means.index[0]
                    top_value = group_means.iloc[0]
                    overall = target.mean()
                    uplift = 0 if overall == 0 else ((top_value - overall) / abs(overall)) * 100
                    rows.append({"Driver": f"{col} = {top_group}", "Type": "Group difference", "Evidence": f"Average {target_col}: {top_value:.2f} vs overall {overall:.2f}", "Strength": abs(uplift) / 100, "Direction": "Higher than average" if uplift > 0 else "Lower than average"})
            except Exception:
                pass
    risk_target = risk_df["Overall Risk Score"]
    for col in df.select_dtypes(include=["number"]).columns:
        candidate = pd.to_numeric(df[col], errors="coerce")
        if candidate.nunique(dropna=True) <= 1:
            continue
        corr = risk_target.corr(candidate)
        if pd.notna(corr):
            rows.append({"Driver": col, "Type": "Risk score relationship", "Evidence": f"Correlation with Overall Risk Score: {corr:.3f}", "Strength": abs(corr), "Direction": "Positive" if corr > 0 else "Negative"})
    if not rows:
        return pd.DataFrame(columns=["Driver", "Type", "Evidence", "Strength", "Direction"])
    result = pd.DataFrame(rows).sort_values("Strength", ascending=False).drop_duplicates(subset=["Driver"]).head(10)
    result["Strength"] = result["Strength"].round(3)
    return result.reset_index(drop=True)


def build_risk_context(risk_df, group_risk_table, root_cause_df, problem_type, fit_score, confidence):
    avg = risk_df["Overall Risk Score"].mean()
    context = f"""
Risk Engine Summary:
Selected problem: {problem_type}
Dataset Fit Score: {fit_score}/100
Analysis Confidence: {confidence}

Average Overall Risk Score: {avg:.2f}/100
High-risk records: {(risk_df['Risk Level'] == 'High').sum()}
Medium-risk records: {(risk_df['Risk Level'] == 'Medium').sum()}
Low-risk records: {(risk_df['Risk Level'] == 'Low').sum()}

Average Risk Scores:
{risk_df.drop(columns=['Risk Level']).mean().round(2).to_string()}
"""
    if confidence == "Low":
        context += "\nImportant limitation: Dataset fit is low. Risk scores may be incomplete or misleading because key problem-specific fields are missing.\n"
    if group_risk_table is not None and not group_risk_table.empty:
        context += f"\nTop High-Risk Groups:\n{group_risk_table.head(10).to_string(index=False)}\n"
    if root_cause_df is not None and not root_cause_df.empty:
        context += f"\nTop Root Cause Candidates:\n{root_cause_df.head(10).to_string(index=False)}\n"
    return context

# ---------------- PRIVACY PROFILE ----------------
def apply_column_masking(df, text_blocks, mask_columns):
    if not mask_columns:
        return text_blocks, {}
    mapping = {str(col): f"column_{i+1}" for i, col in enumerate(df.columns)}
    masked_blocks = []
    for text in text_blocks:
        masked = str(text)
        for original, masked_name in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
            masked = masked.replace(original, masked_name)
        masked_blocks.append(masked)
    return masked_blocks, mapping


def build_ai_profile(df, privacy_mode=True, mask_columns=False, risk_context="", fit_context="", ai_readiness_context=""):
    profile_df = df.copy()
    column_mapping = {}
    if mask_columns:
        column_mapping = {col: f"column_{i+1}" for i, col in enumerate(profile_df.columns)}
        profile_df = profile_df.rename(columns=column_mapping)
    numeric_cols = profile_df.select_dtypes(include=["number"]).columns.tolist()
    profile = {
        "shape": {"rows": profile_df.shape[0], "columns": profile_df.shape[1]},
        "columns": list(profile_df.columns),
        "data_types": profile_df.dtypes.astype(str).to_dict(),
        "missing_counts": profile_df.isna().sum().to_dict(),
        "duplicate_rows": int(profile_df.duplicated().sum()),
        "summary_statistics": profile_df.describe(include="all").fillna("").to_string(),
        "correlation_matrix": profile_df[numeric_cols].corr().round(3).to_string() if len(numeric_cols) > 1 else "Not enough numeric columns for correlation.",
        "dataset_fit_summary": fit_context,
        "ai_readiness_summary": ai_readiness_context,
        "risk_engine_summary": risk_context,
    }
    privacy_note = """
Privacy Mode is ON. The AI prompt uses only summarized profile information, dataset fit assessment,
AI readiness assessment, and calculated risk summaries. Raw dataset rows are not sent.
""" if privacy_mode else """
Standard Mode is ON. This version still sends only summarized profile information, dataset fit assessment,
AI readiness assessment, and calculated risk summaries. Raw dataset rows are not included in the prompt.
"""
    return profile, column_mapping, privacy_note

# ---------------- PROMPTS ----------------
BASE_CONSULTANT_PROMPT = """
You are InsightForge, a privacy-first agentic industrial AI consulting engine.
You help manufacturing, quality, maintenance, supply chain, lab, and operations teams convert summarized operational data into practical decisions.
You do not see raw dataset rows. You only receive summarized information, dataset fit assessment, AI readiness assessment, calculated risk summary, and root-cause candidates.
Never claim that you inspected raw rows. Never invent unsupported measurements, causes, savings, or business impact. Do not overstate certainty.
Use Dataset Fit Score and AI Readiness Score to control confidence. If either is Low, focus on missing data and do not recommend predictive modeling yet.
Style: practical, business-facing, engineering-aware, clear, direct, consulting-grade, implementation-oriented. Avoid generic advice and markdown symbols like ### or **.
"""


def build_context_pack(project_info, problem_type, problem_focus, product_mode, industry_template, privacy_note, fit_score, confidence, fit_context, ai_readiness_score, ai_readiness_level, ai_readiness_context, ai_profile, risk_context):
    return f"""
Client Project Workspace:
{project_info}

Selected Industrial Problem:
{problem_type}

Problem Focus:
{problem_focus}

Product Mode:
{product_mode}

Industry Template:
{industry_template}

Dataset Fit Score: {fit_score}/100
Analysis Confidence: {confidence}
AI Readiness Score: {ai_readiness_score}/100
AI Readiness Level: {ai_readiness_level}

Privacy Note:
{privacy_note}

Dataset Fit Assessment:
{fit_context}

AI Readiness Assessment:
{ai_readiness_context}

Summarized Dataset Profile:
{ai_profile}

Calculated Risk Engine Summary:
{risk_context}
"""


def build_executive_prompt(*args):
    context_pack = build_context_pack(*args)
    return f"""
{BASE_CONSULTANT_PROMPT}
Task: Write an executive summary for a manager.
{context_pack}
Output format:
Executive Summary
- Dataset suitability:
- AI readiness:
- Main operational signal:
- Main risk or limitation:
- Business or engineering impact:
- Recommended management focus:
Rules: Use 5 to 7 bullets only. Mention Dataset Fit Score, Analysis Confidence, and AI Readiness Score.
"""


def build_action_prompt(*args):
    context_pack = build_context_pack(*args)
    return f"""
{BASE_CONSULTANT_PROMPT}
Task: Provide exactly the top 3 recommended actions.
{context_pack}
Output format:
Action 1:
Why it matters:
Expected impact:
Required owner or data:

Action 2:
Why it matters:
Expected impact:
Required owner or data:

Action 3:
Why it matters:
Expected impact:
Required owner or data:
Rules: If Dataset Fit Score or AI Readiness Score is Low, focus on collecting missing fields and improving data readiness. Do not invent numerical savings.
"""


def build_diagnostic_prompt(*args):
    context_pack = build_context_pack(*args)
    return f"""
{BASE_CONSULTANT_PROMPT}
Task: Write a consulting-style diagnostic analysis.
{context_pack}
Output format:
1. Dataset suitability interpretation
2. AI readiness interpretation
3. Key operational risks
4. Top risk drivers
5. Root-cause candidates
6. Business or engineering impact
7. Recommended implementation roadmap
8. Data gaps and next data to collect
Rules: Do not overstate causality. Use terms like candidate driver, possible relationship, or should be validated.
"""


def build_chat_prompt(user_question, *args):
    context_pack = build_context_pack(*args)
    return f"""
{BASE_CONSULTANT_PROMPT}
Task: Answer the user's question using only the summarized profile, dataset fit assessment, AI readiness assessment, and risk-engine summary.
User question: {user_question}
{context_pack}
Output format:
Direct answer:
Reasoning:
Recommended next step:
"""


def get_ai_response(prompt):
    if client is None:
        return "OpenAI API key is missing. Add OPENAI_API_KEY in Streamlit secrets to run AI analysis."
    response = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], temperature=0.2)
    return response.choices[0].message.content

# ---------------- PDF REPORT ----------------
def clean_markdown_text(text):
    text = str(text)
    for old, new in {"**": "", "###": "", "##": "", "#": "", "`": "", "* ": "- "}.items():
        text = text.replace(old, new)
    return text


def add_wrapped_pdf_pages(pdf, title, body, footer="InsightForge | AI Decision Support Report"):
    body = clean_markdown_text(body)
    lines = []
    for para in body.split("\n"):
        lines.extend(textwrap.wrap(para, width=95)) if para.strip() else lines.append("")
    chunks = [lines[i:i + 42] for i in range(0, len(lines), 42)] or [[""]]
    for page, chunk in enumerate(chunks, start=1):
        fig = plt.figure(figsize=(8.5, 11))
        plt.axis("off")
        page_title = title if page == 1 else f"{title} continued"
        plt.text(0.05, 0.95, page_title, fontsize=17, fontweight="bold", va="top")
        plt.text(0.05, 0.88, "\n".join(chunk), fontsize=9.5, va="top")
        plt.text(0.05, 0.04, footer, fontsize=8, color="gray")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()


def create_pdf_report(df, project_info, executive_text, ai_text, action_text, numeric_cols, readiness_score, privacy_note, problem_type, product_mode, risk_context, fit_context, fit_score, confidence, ai_readiness_score, ai_readiness_level, ai_readiness_context, ai_readiness_table, matched_fields, missing_fields, risk_df, group_risk_table, root_cause_df):
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        title_body = f"""
Prepared by InsightForge
Report Type: {product_mode} Industrial AI Decision Support Report
Selected Industrial Problem: {problem_type}

Client Project Workspace:
{project_info}

Dataset Overview:
Rows: {df.shape[0]}
Columns: {df.shape[1]}
Decision Readiness Score: {readiness_score}/100
Dataset Fit Score: {fit_score}/100
Analysis Confidence: {confidence}
AI Readiness Score: {ai_readiness_score}/100
AI Readiness Level: {ai_readiness_level}

Privacy Note:
{privacy_note}
"""
        add_wrapped_pdf_pages(pdf, "InsightForge Industrial Decision Report", title_body)
        overview_body = f"""
Dataset Columns:
{', '.join(map(str, df.columns))}

Matched Problem Fields:
{matched_fields if matched_fields else 'No strong problem-specific fields matched.'}

Missing Recommended Fields:
{missing_fields if missing_fields else 'No major recommended fields missing.'}
"""
        add_wrapped_pdf_pages(pdf, "1. Dataset Overview and Fit", overview_body)
        add_wrapped_pdf_pages(pdf, "2. Dataset Fit Assessment", fit_context)
        add_wrapped_pdf_pages(pdf, "3. AI Readiness Assessment", ai_readiness_context)
        add_wrapped_pdf_pages(pdf, "4. Executive Summary", executive_text)
        add_wrapped_pdf_pages(pdf, "5. Risk Engine Summary", risk_context)
        add_wrapped_pdf_pages(pdf, "6. Top Recommended Actions", action_text)
        add_wrapped_pdf_pages(pdf, "7. Detailed AI Insights", ai_text)

        if ai_readiness_table is not None and not ai_readiness_table.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_df = ai_readiness_table[ai_readiness_table["Readiness Area"] != "Overall AI Readiness"]
            ax.bar(plot_df["Readiness Area"], plot_df["Score"])
            ax.set_title("AI Readiness by Area")
            ax.set_ylabel("Readiness Score")
            plt.xticks(rotation=45, ha="right")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()
        if risk_df is not None and not risk_df.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            risk_df["Overall Risk Score"].plot(kind="hist", bins=20, ax=ax)
            ax.set_title("Overall Risk Score Distribution")
            ax.set_xlabel("Risk Score")
            ax.set_ylabel("Frequency")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()
        if group_risk_table is not None and not group_risk_table.empty:
            top_groups = group_risk_table.head(10)
            group_col = top_groups.columns[0]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(top_groups[group_col].astype(str), top_groups["Overall Risk Score"])
            ax.set_title("Top High-Risk Groups")
            ax.set_xlabel(group_col)
            ax.set_ylabel("Overall Risk Score")
            plt.xticks(rotation=45, ha="right")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()
        if root_cause_df is not None and not root_cause_df.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            top_root = root_cause_df.head(8)
            ax.barh(top_root["Driver"].astype(str), top_root["Strength"])
            ax.set_title("Top Root Cause Candidates")
            ax.set_xlabel("Relative Strength")
            ax.invert_yaxis()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()
        chart_cols = [col for col in ["health_score", "tool_wear", "temperature", "vibration", "downtime", "defect_probability", "breakdown_probability", "cycle_time", "power_consumption", "queue_length"] if col in numeric_cols]
        for col in chart_cols[:4]:
            fig, ax = plt.subplots(figsize=(8, 5))
            df[col].plot(ax=ax)
            ax.set_title(f"{col} Trend")
            ax.set_xlabel("Index")
            ax.set_ylabel(col)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()
        roadmap = """
Priority Roadmap

0-30 Days:
- If dataset fit is low, collect missing recommended fields before making strong operational decisions.
- If AI readiness is low, improve data structure before predictive modeling.
- If dataset fit is medium or high, investigate high-risk machines, states, groups, or process segments from the Risk Engine.
- Validate root-cause candidates with shop-floor observations, maintenance records, or engineering review.

30-90 Days:
- Build a monitoring dashboard for the highest-risk KPIs.
- Standardize data collection around downtime, maintenance, quality, cycle time, and process conditions.
- Launch targeted process improvement or maintenance actions.

3-6 Months:
- Develop predictive maintenance or quality-risk models only if data readiness and dataset fit are sufficient.
- Integrate automated alerts for risk-score changes.
- Review business impact and refine the improvement roadmap.
"""
        add_wrapped_pdf_pages(pdf, "8. Priority Roadmap", roadmap)
        add_wrapped_pdf_pages(pdf, "9. Consulting and Next Steps", "For deeper implementation support, contact: insightforge.ai@gmail.com")
    buffer.seek(0)
    return buffer

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🔥 InsightForge")
    st.markdown("**Agentic Industrial AI Decision OS**")
    st.markdown("---")
    st.markdown("### 🧾 Client Project Workspace")
    project_name = st.text_input("Project Name", value="Industrial Data Review")
    company_name = st.text_input("Company Name", value="Demo Company")
    plant_site = st.text_input("Plant / Site", value="Plant 1")
    prepared_for = st.text_input("Prepared For", value="Operations Team")
    prepared_by = st.text_input("Prepared By", value="InsightForge")
    analysis_date = st.date_input("Analysis Date", value=date.today())
    st.markdown("---")
    st.markdown("### 🧩 Industrial Problem")
    problem_type = st.selectbox("What problem are you trying to solve?", ["Reduce Downtime", "Improve Quality", "Predict Maintenance Risk", "Reduce Cost", "Improve Delivery Performance", "Identify Bottlenecks", "Improve Process Stability", "General Industrial Analysis"])
    st.markdown("---")
    st.markdown("### 🧭 Product Mode")
    product_mode = st.selectbox("Select analysis mode", ["Demo Mode", "Client Mode", "Expert Review Mode"])
    st.markdown("---")
    st.markdown("### 🔐 Privacy Controls")
    privacy_mode = st.toggle("Privacy Mode", value=True)
    mask_columns = st.checkbox("Mask column names before AI analysis", value=True if product_mode in ["Client Mode", "Expert Review Mode"] else False)
    industry_template = st.selectbox("Industry Template", ["Manufacturing", "Quality Analysis", "Supply Chain", "Lab Testing", "General Business"])
    st.markdown("---")
    st.markdown("### Navigation")
    st.markdown("""
    - 📂 Upload Data  
    - 📊 Executive Dashboard  
    - ✅ Dataset Fit  
    - 🤖 AI Readiness  
    - ⚠️ Risk Engine  
    - 📈 Visual Analytics  
    - 🧾 AI Transparency  
    - 🧠 AI Decision Engine  
    - 💬 Ask Your Data  
    - 🚀 Consulting  
    """)
    st.markdown("---")
    st.caption("Version 9.0 | Clean Code File")

project_info = f"""
Project Name: {project_name}
Company Name: {company_name}
Plant / Site: {plant_site}
Prepared For: {prepared_for}
Prepared By: {prepared_by}
Analysis Date: {analysis_date}
"""

# ---------------- LANDING PAGE ----------------
st.markdown("""
<div class="hero-shell">
    <div class="hero-grid">
        <div>
            <div class="hero-badge">Privacy-First Agentic Industrial AI Decision OS</div>
            <div style="display:flex; align-items:center;"><div class="logo-mark">🔥</div><div class="hero-title">InsightForge</div></div>
            <div class="hero-title">Turn Industrial Data into <span>Actionable Decisions</span>.</div>
            <div class="hero-subtitle">Check data readiness, diagnose risks, identify root-cause candidates, and generate consulting-ready action plans.</div>
            <div class="hero-text">InsightForge converts operational data into dataset-fit scores, AI readiness scores, risk scores, dashboards, agentic recommendations, and professional decision reports.</div>
        </div>
        <div class="floating-panel">
            <div class="panel-row"><span>🧾 Client Workspace</span><span class="panel-value">Enabled</span></div>
            <div class="panel-row"><span>🤖 AI Readiness</span><span class="panel-value">Enabled</span></div>
            <div class="panel-row"><span>✅ Dataset Fit</span><span class="panel-value">Enabled</span></div>
            <div class="panel-row"><span>⚠️ Risk Engine</span><span class="panel-value">Enabled</span></div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="trust-strip">
    <div class="trust-pill">🔐 Privacy-first analysis</div><div class="trust-pill">🤖 AI readiness scoring</div><div class="trust-pill">✅ Dataset fit validation</div><div class="trust-pill">⚠️ Risk scoring engine</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="feature-strip">
    <div class="feature-card"><div class="feature-icon">🧾</div><div class="feature-title">Client Workspace</div><div class="feature-text">Add project, company, site, prepared-for, and report metadata.</div></div>
    <div class="feature-card"><div class="feature-icon">🤖</div><div class="feature-title">AI Readiness Engine</div><div class="feature-text">Check readiness for predictive maintenance, quality AI, downtime diagnosis, and digital twins.</div></div>
    <div class="feature-card"><div class="feature-icon">✅</div><div class="feature-title">Dataset Fit Score</div><div class="feature-text">Checks whether the uploaded data supports the selected industrial problem.</div></div>
    <div class="feature-card"><div class="feature-icon">📄</div><div class="feature-title">Consulting Reports</div><div class="feature-text">Export structured decision reports for managers, engineers, and clients.</div></div>
</div>
""", unsafe_allow_html=True)

# ---------------- FILE UPLOAD ----------------
st.markdown('<div class="section-title">📂 Start Privacy-First Diagnosis</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Upload your CSV/Excel file or try the built-in demo manufacturing dataset.</div>', unsafe_allow_html=True)

demo_col, upload_col = st.columns([1, 2])
with demo_col:
    st.markdown("### ⚙️ Try Demo")
    st.caption("No file ready? Load a sample industrial dataset.")
    if st.button("Load Demo Dataset", type="secondary"):
        st.session_state.use_demo_data = True
with upload_col:
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        st.session_state.use_demo_data = False

# ---------------- MAIN APP ----------------
df = None
data_source_name = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        data_source_name = uploaded_file.name
    except Exception as exc:
        st.error(f"Could not read the uploaded file: {exc}")
elif st.session_state.use_demo_data:
    df = generate_demo_dataset()
    data_source_name = "Demo Manufacturing Dataset"

if df is None:
    st.info("Upload a CSV/Excel file or load the demo dataset to begin analysis.")
    st.stop()

st.success(f"✅ Active dataset: {data_source_name}")
signature = f"{data_source_name}_{df.shape}_{list(df.columns)}_{problem_type}_{product_mode}_{privacy_mode}_{mask_columns}_{industry_template}"
if st.session_state.last_dataset_signature != signature:
    st.session_state.ai_text = ""
    st.session_state.action_text = ""
    st.session_state.executive_text = ""
    st.session_state.chat_history = []
    st.session_state.last_dataset_signature = signature

numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
missing_values = int(df.isna().sum().sum())
duplicate_rows = int(df.duplicated().sum())
total_cells = df.shape[0] * df.shape[1]
data_completeness = 100 if total_cells == 0 else round(100 - (missing_values / total_cells * 100), 2)
sensitive_cols = detect_sensitive_columns(df)
readiness_score = calculate_decision_readiness(df, numeric_cols, missing_values, duplicate_rows)
fit_score, confidence, matched_fields, missing_fields, raw_fit_context = calculate_dataset_fit(df, problem_type)
ai_readiness_score, ai_readiness_level, ai_readiness_table, raw_ai_readiness_context = calculate_ai_readiness(df)
risk_df = compute_risk_engine(df, problem_type)
group_candidates = get_group_candidates(df)
default_group_col = group_candidates[0] if group_candidates else None
group_risk_table = make_group_risk_table(df, risk_df, default_group_col) if default_group_col else pd.DataFrame()
root_cause_df = root_cause_finder(df, risk_df, problem_type)
raw_risk_context = build_risk_context(risk_df, group_risk_table, root_cause_df, problem_type, fit_score, confidence)
masked_blocks, privacy_mapping = apply_column_masking(df, [raw_fit_context, raw_ai_readiness_context, raw_risk_context], mask_columns)
fit_context, ai_readiness_context, risk_context = masked_blocks
ai_profile, column_mapping, privacy_note = build_ai_profile(df, privacy_mode, mask_columns, risk_context, fit_context, ai_readiness_context)
problem_focus = get_problem_focus(problem_type)

# ---------------- STATUS PANELS ----------------
st.markdown('<div class="info-card"><h3>🧭 Product Mode Active</h3><p>InsightForge adjusts its analysis style based on Demo Mode, Client Mode, or Expert Review Mode.</p></div>', unsafe_allow_html=True)
if product_mode == "Demo Mode":
    st.info("🧪 Demo Mode Active — suitable for quick testing and sample datasets.")
elif product_mode == "Client Mode":
    st.success("🔐 Client Mode Active — designed for real company data using summarized profiles instead of raw rows.")
else:
    st.warning("🚀 Expert Review Mode Active — consulting-style diagnosis with business impact and roadmap.")

st.markdown('<div class="info-card"><h3>🔐 Privacy-First Analysis Mode</h3><p>AI receives summarized profile information, dataset fit, AI readiness, and risk summary. Raw dataset rows are not sent.</p></div>', unsafe_allow_html=True)
if sensitive_cols:
    st.warning(f"Sensitive or business-critical column names detected: {', '.join(map(str, sensitive_cols))}. Consider using column masking.")

st.markdown('<div class="fit-card"><h3>✅ Dataset Fit Engine</h3><p>Checks whether your dataset supports the selected industrial problem before diagnosis.</p></div>', unsafe_allow_html=True)
if confidence == "High":
    st.success(f"✅ Dataset Fit Score: {fit_score}/100 — High Confidence")
elif confidence == "Medium":
    st.warning(f"⚠️ Dataset Fit Score: {fit_score}/100 — Medium Confidence")
else:
    st.error(f"🚫 Dataset Fit Score: {fit_score}/100 — Low Confidence. Recommendations will focus on missing data first.")

st.markdown('<div class="ai-card"><h3>🤖 AI Readiness Engine</h3><p>Evaluates readiness for predictive maintenance, quality prediction, downtime diagnosis, and digital twin modeling.</p></div>', unsafe_allow_html=True)
if ai_readiness_level == "High":
    st.success(f"🤖 AI Readiness Score: {ai_readiness_score}/100 — High Readiness")
elif ai_readiness_level == "Medium":
    st.warning(f"🤖 AI Readiness Score: {ai_readiness_score}/100 — Medium Readiness")
else:
    st.error(f"🤖 AI Readiness Score: {ai_readiness_score}/100 — Low Readiness. Improve data structure before predictive AI.")

# ---------------- EXECUTIVE DASHBOARD ----------------
st.markdown('<div class="section-title">📊 Executive Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">A quick health overview of your uploaded dataset.</div>', unsafe_allow_html=True)
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
with c1: kpi_card("Rows", df.shape[0], "Total records")
with c2: kpi_card("Columns", df.shape[1], "Total features")
with c3: kpi_card("Numeric", len(numeric_cols), "Analyzable metrics")
with c4: kpi_card("Missing", missing_values, "Data gaps")
with c5: kpi_card("Readiness", f"{readiness_score}/100", "Data score")
with c6: kpi_card("Fit", f"{fit_score}/100", confidence)
with c7: kpi_card("AI Ready", f"{ai_readiness_score}/100", ai_readiness_level)

st.markdown("### 🧪 Data Health Score")
st.progress(data_completeness / 100)
st.caption("Higher score means the dataset is more complete for analysis and decision-making.")
st.markdown("### 🎯 Decision Readiness Score")
st.progress(readiness_score / 100)
st.caption("Higher score means the dataset is more ready for operational diagnosis and reporting.")
st.markdown("### ✅ Dataset Fit Score")
st.progress(fit_score / 100)
st.caption("Higher score means the dataset is better suited to the selected industrial problem.")
st.markdown("### 🤖 AI Readiness Score")
st.progress(ai_readiness_score / 100)
st.caption("Higher score means the dataset is more ready for advanced AI, predictive modeling, and digital twin use cases.")
st.markdown("---")

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(["📋 Data Overview", "✅ Dataset Fit", "🤖 AI Readiness", "⚠️ Risk Engine", "📈 Visual Dashboard", "🧾 AI Transparency", "🧠 AI Decision Engine", "💬 Ask Your Data", "🚀 Consulting"])

with tab1:
    st.markdown("### 📋 Client Project Workspace")
    st.text(project_info)
    st.markdown("### 📋 Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.markdown("### 📈 Summary Statistics")
    st.dataframe(df.describe(include="all"), use_container_width=True)
    st.markdown("### 🧠 Data Quality Snapshot")
    q1, q2, q3 = st.columns(3)
    with q1: kpi_card("Duplicate Rows", duplicate_rows, "Repeated records")
    with q2: kpi_card("Missing Cells", missing_values, "Blank values")
    with q3: kpi_card("Completeness", f"{data_completeness}%", "Overall health")
    if numeric_cols:
        st.markdown("### 🔢 Numeric Metric Cards")
        metric_cols = st.columns(3)
        for i, col in enumerate(numeric_cols):
            with metric_cols[i % 3]:
                kpi_card(col, f"{df[col].mean():.2f}", f"Min {df[col].min()} | Max {df[col].max()}")
    else:
        st.warning("No numeric columns found.")

with tab2:
    st.markdown("### ✅ Dataset Suitability / Fit Assessment")
    st.write(f"Selected industrial problem: **{problem_type}**")
    st.write(f"Dataset Fit Score: **{fit_score}/100**")
    st.write(f"Analysis Confidence: **{confidence}**")
    st.markdown("#### Matched fields")
    if matched_fields:
        st.dataframe(pd.DataFrame([{"Expected Field": k, "Matched Column": v} for k, v in matched_fields.items()]), use_container_width=True)
    else:
        st.warning("No strong problem-specific fields matched.")
    st.markdown("#### Missing recommended fields")
    if missing_fields:
        st.dataframe(pd.DataFrame({"Missing Recommended Field": missing_fields}), use_container_width=True)
    else:
        st.success("No major recommended fields missing for the selected problem.")
    with st.expander("View dataset fit assessment text sent to AI"):
        st.text(fit_context)

with tab3:
    st.markdown("### 🤖 AI Readiness Assessment")
    st.write(f"Overall AI Readiness Score: **{ai_readiness_score}/100**")
    st.write(f"Overall AI Readiness Level: **{ai_readiness_level}**")
    st.dataframe(ai_readiness_table, use_container_width=True)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    plot_df = ai_readiness_table[ai_readiness_table["Readiness Area"] != "Overall AI Readiness"]
    ax.bar(plot_df["Readiness Area"], plot_df["Score"])
    ax.set_title("AI Readiness by Area")
    ax.set_ylabel("Readiness Score")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)
    with st.expander("View AI readiness assessment text sent to AI"):
        st.text(ai_readiness_context)

with tab4:
    st.markdown("### ⚠️ Risk Scoring Engine")
    st.caption("Calculated locally from uploaded data. AI receives only summarized risk results, not raw rows.")
    if confidence == "Low":
        st.error("Dataset fit is low. Risk scores may be incomplete because selected problem fields are missing.")
    avg_overall = risk_df["Overall Risk Score"].mean()
    r1, r2, r3, r4 = st.columns(4)
    with r1: kpi_card("Avg Overall Risk", f"{avg_overall:.1f}/100", "Calculated score")
    with r2: kpi_card("High Risk", int((risk_df["Risk Level"] == "High").sum()), "High-priority records")
    with r3: kpi_card("Medium Risk", int((risk_df["Risk Level"] == "Medium").sum()), "Watchlist records")
    with r4: kpi_card("Low Risk", int((risk_df["Risk Level"] == "Low").sum()), "Stable records")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Risk Score Averages")
        fig, ax = plt.subplots(figsize=(8, 4))
        risk_df.drop(columns=["Risk Level"]).mean().sort_values(ascending=False).plot(kind="bar", ax=ax)
        ax.set_title("Average Risk Score by Category")
        ax.set_ylabel("Score")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
    with col_b:
        st.markdown("#### Overall Risk Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        risk_df["Overall Risk Score"].plot(kind="hist", bins=20, ax=ax)
        ax.set_title("Overall Risk Score Distribution")
        ax.set_xlabel("Risk Score")
        st.pyplot(fig)
    st.markdown("### 🏭 High-Risk Group Ranking")
    if group_candidates:
        selected_group_col = st.selectbox("Group risk by", group_candidates, index=0)
        group_risk_table = make_group_risk_table(df, risk_df, selected_group_col)
        st.dataframe(group_risk_table, use_container_width=True)
        if not group_risk_table.empty:
            top_groups = group_risk_table.head(10)
            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.bar(top_groups[selected_group_col].astype(str), top_groups["Overall Risk Score"])
            ax.set_title(f"Top High-Risk Groups by {selected_group_col}")
            ax.set_xlabel(selected_group_col)
            ax.set_ylabel("Overall Risk Score")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)
        raw_risk_context = build_risk_context(risk_df, group_risk_table, root_cause_df, problem_type, fit_score, confidence)
        masked_blocks, _ = apply_column_masking(df, [raw_risk_context], mask_columns)
        risk_context = masked_blocks[0]
        ai_profile["risk_engine_summary"] = risk_context
    else:
        st.info("No suitable categorical grouping column found.")
    st.markdown("### 🔍 Root Cause Candidates")
    if not root_cause_df.empty:
        st.dataframe(root_cause_df, use_container_width=True)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        plot_df = root_cause_df.head(10)
        ax.barh(plot_df["Driver"].astype(str), plot_df["Strength"])
        ax.set_title("Top Root Cause Candidate Strength")
        ax.set_xlabel("Relative Strength")
        ax.invert_yaxis()
        st.pyplot(fig)
    else:
        st.warning("Not enough numeric or categorical structure to estimate root-cause candidates.")

with tab5:
    st.markdown("### 📈 Visual Analytics Dashboard")
    if numeric_cols:
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            metric1 = st.selectbox("Trend Metric", numeric_cols, key="line_metric")
            fig, ax = plt.subplots(figsize=(8, 4))
            df[metric1].plot(ax=ax)
            ax.set_title(f"{metric1} Trend")
            ax.set_xlabel("Index")
            ax.set_ylabel(metric1)
            st.pyplot(fig)
        with r1c2:
            metric2 = st.selectbox("Distribution Metric", numeric_cols, key="hist_metric")
            fig, ax = plt.subplots(figsize=(8, 4))
            df[metric2].plot(kind="hist", ax=ax, bins=20)
            ax.set_title(f"{metric2} Distribution")
            ax.set_xlabel(metric2)
            st.pyplot(fig)
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            metric3 = st.selectbox("Outlier / Spread Metric", numeric_cols, key="box_metric")
            fig, ax = plt.subplots(figsize=(8, 4))
            df[metric3].plot(kind="box", ax=ax)
            ax.set_title(f"{metric3} Spread / Outlier View")
            st.pyplot(fig)
        with r2c2:
            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr()
                fig, ax = plt.subplots(figsize=(8, 5))
                heatmap = ax.imshow(corr, aspect="auto")
                ax.set_xticks(range(len(corr.columns)))
                ax.set_yticks(range(len(corr.columns)))
                ax.set_xticklabels(corr.columns, rotation=90)
                ax.set_yticklabels(corr.columns)
                ax.set_title("Correlation Heatmap")
                fig.colorbar(heatmap, ax=ax)
                st.pyplot(fig)
            else:
                st.info("Need at least two numeric columns for correlation heatmap.")
    else:
        st.warning("No numeric columns found for dashboard charts.")

with tab6:
    st.markdown("### 🧾 AI Transparency Panel")
    st.success("Raw dataset rows are not sent to the AI prompt in this version.")
    st.markdown("""
    #### What AI receives
    - Client project metadata
    - Dataset shape
    - Column names or masked labels
    - Data types
    - Missing-value counts
    - Duplicate-row count
    - Summary statistics
    - Correlation matrix
    - Dataset fit assessment
    - AI readiness assessment
    - Calculated risk-engine summary
    - Root-cause candidate summary
    """)
    st.write(f"Industrial Problem: **{problem_type}**")
    st.write(f"Product Mode: **{product_mode}**")
    st.write(f"Privacy Mode: **{'ON' if privacy_mode else 'OFF'}**")
    st.write(f"Column Masking: **{'ON' if mask_columns else 'OFF'}**")
    st.write(f"Dataset Fit Score: **{fit_score}/100**")
    st.write(f"AI Readiness Score: **{ai_readiness_score}/100**")
    if mask_columns and column_mapping:
        with st.expander("View local column masking map"):
            st.write(column_mapping)
    with st.expander("Preview summarized AI profile"):
        st.write(ai_profile)
    with st.expander("Preview dataset fit assessment sent to AI"):
        st.text(fit_context)
    with st.expander("Preview AI readiness assessment sent to AI"):
        st.text(ai_readiness_context)
    with st.expander("Preview risk-engine summary sent to AI"):
        st.text(risk_context)

with tab7:
    st.markdown("### 🧠 AI Decision Engine")
    st.info("Run AI analysis when ready. AI receives summarized profile, dataset fit, AI readiness, and risk summaries.")
    if confidence == "Low":
        st.error("Dataset fit is low. AI output will focus on what data is missing before strong diagnosis.")
    if ai_readiness_level == "Low":
        st.warning("AI readiness is low. InsightForge will avoid recommending predictive modeling until data improves.")
    run_ai = st.button("🚀 Run AI Analysis", type="primary")
    prompt_args = (project_info, problem_type, problem_focus, product_mode, industry_template, privacy_note, fit_score, confidence, fit_context, ai_readiness_score, ai_readiness_level, ai_readiness_context, ai_profile, risk_context)
    if run_ai:
        with st.spinner("Generating executive summary..."):
            st.session_state.executive_text = get_ai_response(build_executive_prompt(*prompt_args))
        with st.spinner("Generating top recommended actions..."):
            st.session_state.action_text = get_ai_response(build_action_prompt(*prompt_args))
        with st.spinner("Generating detailed AI insights..."):
            st.session_state.ai_text = get_ai_response(build_diagnostic_prompt(*prompt_args))
    if st.session_state.executive_text:
        st.markdown("### 📌 Executive Summary")
        st.info(st.session_state.executive_text)
    if st.session_state.action_text and st.session_state.ai_text:
        left, right = st.columns(2)
        with left:
            st.markdown("### 🎯 Top 3 Recommended Actions")
            st.warning(st.session_state.action_text)
        with right:
            st.markdown("### 🧠 Detailed AI Insights")
            st.success(st.session_state.ai_text)
        st.markdown("### 📄 Export Consulting-Style Report")
        pdf_report = create_pdf_report(df, project_info, st.session_state.executive_text, st.session_state.ai_text, st.session_state.action_text, numeric_cols, readiness_score, privacy_note, problem_type, product_mode, risk_context, fit_context, fit_score, confidence, ai_readiness_score, ai_readiness_level, ai_readiness_context, ai_readiness_table, matched_fields, missing_fields, risk_df, group_risk_table, root_cause_df)
        st.download_button(label="📥 Download InsightForge V9 Industrial Report", data=pdf_report.getvalue(), file_name="InsightForge_V9_Industrial_Report.pdf", mime="application/pdf")
    else:
        st.warning("Run AI Analysis first to generate the report.")

with tab8:
    st.markdown("### 💬 Ask Your Data")
    st.caption("Ask questions using the summarized privacy-safe data profile. Raw rows are not sent.")
    suggested_questions = [
        "Is this dataset suitable for the selected problem?",
        "Is this dataset ready for predictive AI?",
        "What fields are missing for reliable diagnosis?",
        "What should we improve first?",
        "Which machine or group appears highest risk?",
        "What is likely causing downtime?",
        "Which root cause should management investigate first?",
        "Is this data ready for predictive maintenance?",
        "Is this data ready for a digital twin?",
        "What should management do next?",
    ]
    selected_question = st.selectbox("Choose a suggested question", [""] + suggested_questions)
    user_question = st.text_input("Or type your own question")
    final_question = user_question.strip() if user_question.strip() else selected_question
    if st.button("Ask InsightForge") and final_question:
        prompt_args = (project_info, problem_type, problem_focus, product_mode, industry_template, privacy_note, fit_score, confidence, fit_context, ai_readiness_score, ai_readiness_level, ai_readiness_context, ai_profile, risk_context)
        with st.spinner("Thinking through your summarized data profile..."):
            answer = get_ai_response(build_chat_prompt(final_question, *prompt_args))
        st.session_state.chat_history.append((final_question, answer))
    if st.session_state.chat_history:
        st.markdown("### Conversation")
        for q, a in reversed(st.session_state.chat_history):
            st.markdown(f"**You:** {q}")
            st.markdown(f"**InsightForge:** {a}")
            st.markdown("---")

with tab9:
    st.markdown("""
    <div class="consulting-card">
        <h2>🚀 Need Expert Help?</h2>
        <p>InsightForge gives an initial AI-based diagnosis. For deeper implementation, expert review can help with root cause analysis, process optimization, cost reduction, quality improvement, predictive maintenance planning, AI readiness planning, digital twin readiness, and custom dashboard development.</p>
    </div>
    """, unsafe_allow_html=True)
    with st.form("consulting_form"):
        f1, f2 = st.columns(2)
        with f1:
            name = st.text_input("Name")
            email = st.text_input("Email")
        with f2:
            industry = st.text_input("Industry")
            company = st.text_input("Company / Organization")
        problem = st.text_area("Describe your challenge")
        submitted = st.form_submit_button("Request Consulting")
        if submitted:
            st.success("✅ Request received! Please email your dataset/problem for expert review.")
            st.markdown("""
            📧 **Contact:** insightforge.ai@gmail.com  
            ⏱️ **Response time:** Within 24 hours  
            📌 **Please include:** your dataset, problem description, and expected outcome.
            """)

st.markdown("""
<div class="footer-card">
    <b>InsightForge V9.0</b> | Privacy-first agentic industrial AI decision OS with client workspace, dataset-fit validation, AI readiness scoring, risk scoring, root-cause analysis, and implementation pathway.
</div>
""", unsafe_allow_html=True)
