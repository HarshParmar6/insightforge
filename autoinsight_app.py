import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
import textwrap
import re

# ---------------- PAGE SETUP ----------------
st.set_page_config(
    page_title="InsightForge",
    page_icon="🔥",
    layout="wide"
)

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ---------------- SESSION STATE ----------------
default_states = {
    "ai_text": "",
    "action_text": "",
    "executive_text": "",
    "privacy_note": "",
    "chat_history": [],
    "last_dataset_signature": ""
}

for key, value in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top left, #E0F2FE 0%, #F8FAFC 35%, #FFFFFF 100%);
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(28px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes floatCard {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-12px); }
    100% { transform: translateY(0px); }
}

@keyframes glowPulse {
    0% { box-shadow: 0 0 20px rgba(37, 99, 235, 0.20); }
    50% { box-shadow: 0 0 42px rgba(37, 99, 235, 0.45); }
    100% { box-shadow: 0 0 20px rgba(37, 99, 235, 0.20); }
}

.logo-mark {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 54px;
    height: 54px;
    border-radius: 16px;
    background: linear-gradient(135deg, #F97316, #EF4444);
    color: white;
    font-size: 30px;
    margin-right: 12px;
    box-shadow: 0 10px 24px rgba(249,115,22,0.35);
}

.hero-shell {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #020617 0%, #0F172A 45%, #1E3A8A 100%);
    padding: 46px;
    border-radius: 30px;
    color: white;
    margin-bottom: 30px;
    box-shadow: 0 20px 60px rgba(15, 23, 42, 0.35);
    animation: fadeUp 0.8s ease-out;
}

.hero-shell:before {
    content: "";
    position: absolute;
    width: 360px;
    height: 360px;
    right: -120px;
    top: -120px;
    background: radial-gradient(circle, rgba(96,165,250,0.45), rgba(37,99,235,0.05));
    border-radius: 50%;
}

.hero-content {
    position: relative;
    z-index: 2;
}

.hero-badge {
    display: inline-block;
    background: rgba(37, 99, 235, 0.9);
    color: white;
    padding: 8px 16px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 800;
    margin-bottom: 18px;
}

.hero-title {
    font-size: 58px;
    font-weight: 950;
    margin-bottom: 12px;
    line-height: 1.05;
}

.hero-title span {
    color: #60A5FA;
}

.hero-subtitle {
    font-size: 23px;
    color: #E5E7EB;
    margin-bottom: 14px;
    max-width: 850px;
}

.hero-text {
    font-size: 16px;
    color: #CBD5E1;
    max-width: 850px;
    line-height: 1.65;
}

.hero-grid {
    display: grid;
    grid-template-columns: 1.2fr 0.8fr;
    gap: 26px;
    align-items: center;
}

.floating-panel {
    background: rgba(255,255,255,0.10);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.20);
    border-radius: 24px;
    padding: 22px;
    animation: floatCard 4s ease-in-out infinite;
}

.panel-row {
    display: flex;
    justify-content: space-between;
    margin-bottom: 14px;
    font-size: 14px;
}

.panel-value {
    font-weight: 900;
    color: #93C5FD;
}

.feature-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 18px;
    margin-bottom: 30px;
    animation: fadeUp 1.0s ease-out;
}

.feature-card {
    background: white;
    border-radius: 22px;
    padding: 22px;
    border: 1px solid #E5E7EB;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    transition: all 0.25s ease;
}

.feature-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 16px 36px rgba(15, 23, 42, 0.13);
}

.feature-icon {
    font-size: 30px;
    margin-bottom: 10px;
}

.feature-title {
    font-weight: 850;
    font-size: 17px;
    color: #111827;
    margin-bottom: 6px;
}

.feature-text {
    color: #6B7280;
    font-size: 14px;
    line-height: 1.45;
}

.upload-zone {
    background: white;
    border-radius: 24px;
    padding: 26px;
    border: 1px solid #E5E7EB;
    box-shadow: 0 8px 28px rgba(15, 23, 42, 0.08);
    margin-bottom: 24px;
    animation: glowPulse 4s infinite;
}

.section-title {
    font-size: 30px;
    font-weight: 900;
    color: #111827;
    margin-top: 18px;
    margin-bottom: 8px;
}

.section-subtitle {
    color: #6B7280;
    font-size: 15px;
    margin-bottom: 18px;
}

.kpi-card {
    background: white;
    padding: 22px;
    border-radius: 20px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    border-left: 7px solid #2563EB;
    margin-bottom: 18px;
    transition: all 0.25s ease;
}

.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 34px rgba(15, 23, 42, 0.12);
}

.kpi-title {
    font-size: 14px;
    color: #6B7280;
    margin-bottom: 6px;
}

.kpi-value {
    font-size: 30px;
    font-weight: 900;
    color: #111827;
}

.kpi-subtitle {
    font-size: 12px;
    color: #9CA3AF;
    margin-top: 4px;
}

.privacy-card {
    background: linear-gradient(135deg, #ECFDF5, #FFFFFF);
    border: 1px solid #A7F3D0;
    border-radius: 22px;
    padding: 24px;
    box-shadow: 0 8px 24px rgba(16, 185, 129, 0.12);
    margin-bottom: 22px;
}

.warning-card {
    background: linear-gradient(135deg, #FFF7ED, #FFFFFF);
    border: 1px solid #FDBA74;
    border-radius: 22px;
    padding: 24px;
    box-shadow: 0 8px 24px rgba(249, 115, 22, 0.10);
    margin-bottom: 22px;
}

.consulting-card {
    background: linear-gradient(135deg, #EFF6FF, #FFFFFF);
    padding: 30px;
    border-radius: 24px;
    border: 1px solid #BFDBFE;
    box-shadow: 0 10px 30px rgba(37, 99, 235, 0.12);
    margin-top: 20px;
}

.footer-card {
    background-color: #0F172A;
    color: #CBD5E1;
    padding: 24px;
    border-radius: 20px;
    margin-top: 30px;
}

@media (max-width: 900px) {
    .hero-grid { grid-template-columns: 1fr; }
    .feature-strip { grid-template-columns: 1fr; }
    .hero-title { font-size: 42px; }
}
</style>
""", unsafe_allow_html=True)

# ---------------- HELPER FUNCTIONS ----------------
def kpi_card(title, value, subtitle):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def detect_sensitive_columns(df):
    patterns = [
        "name", "email", "phone", "mobile", "address", "ssn", "dob",
        "birth", "account", "customer", "client", "employee", "supplier",
        "vendor", "price", "cost", "salary", "revenue", "profit", "id"
    ]

    flagged = []
    for col in df.columns:
        col_lower = str(col).lower()
        for p in patterns:
            if re.search(p, col_lower):
                flagged.append(col)
                break
    return flagged


def calculate_decision_readiness(df, numeric_cols, missing_values, duplicate_rows):
    total_cells = df.shape[0] * df.shape[1]

    if total_cells == 0:
        return 0

    missing_penalty = min(35, (missing_values / total_cells) * 100)
    duplicate_penalty = min(20, (duplicate_rows / max(df.shape[0], 1)) * 100)
    numeric_penalty = 0 if len(numeric_cols) > 0 else 25
    size_penalty = 0 if df.shape[0] >= 30 else 15

    score = 100 - missing_penalty - duplicate_penalty - numeric_penalty - size_penalty
    return round(max(0, min(100, score)), 1)


def get_problem_focus(problem_type):
    focus_map = {
        "Reduce Downtime": """
Focus on downtime, breakdown probability, cycle time, queue length, machine state, machine health, and maintenance flags.
Prioritize machine availability, bottlenecks, and causes of production stoppage.
""",
        "Improve Quality": """
Focus on defect probability, dimensional error, process variation, material type, job type, and abnormal operating conditions.
Prioritize quality risk drivers and corrective actions.
""",
        "Predict Maintenance Risk": """
Focus on health score, initial health, vibration, temperature, spindle load, tool wear, breakdown probability, maintenance flags, and downtime.
Prioritize equipment risk ranking and preventive maintenance actions.
""",
        "Reduce Cost": """
Focus on power consumption, cycle time, downtime, tool wear, defect probability, queue length, and resource inefficiency.
Prioritize cost-saving opportunities and waste reduction.
""",
        "Improve Delivery Performance": """
Focus on tardiness, queue length, job completion, cycle time, downtime, and production delay signals.
Prioritize schedule stability and delivery reliability.
""",
        "Identify Bottlenecks": """
Focus on queue length, cycle time, downtime, job completion, machine state, and process flow delays.
Prioritize bottleneck identification and throughput improvement.
""",
        "Improve Process Stability": """
Focus on variability, outliers, standard deviations, correlations, vibration, temperature, feed rate, spindle speed, defect probability, and dimensional error.
Prioritize stability, repeatability, and process control.
""",
        "General Industrial Analysis": """
Focus on overall data health, operational risks, performance drivers, quality risks, maintenance signals, and business impact.
Prioritize practical actions and implementation roadmap.
"""
    }
    return focus_map.get(problem_type, focus_map["General Industrial Analysis"])


def build_ai_profile(df, privacy_mode=True, mask_columns=False):
    temp_df = df.copy()

    if mask_columns:
        mapping = {col: f"column_{i+1}" for i, col in enumerate(temp_df.columns)}
        temp_df = temp_df.rename(columns=mapping)
    else:
        mapping = {}

    numeric_cols = temp_df.select_dtypes(include=["number"]).columns.tolist()

    profile = {}
    profile["shape"] = {
        "rows": temp_df.shape[0],
        "columns": temp_df.shape[1],
    }
    profile["columns"] = list(temp_df.columns)
    profile["data_types"] = temp_df.dtypes.astype(str).to_dict()
    profile["missing_counts"] = temp_df.isna().sum().to_dict()
    profile["duplicate_rows"] = int(temp_df.duplicated().sum())
    profile["summary_statistics"] = temp_df.describe(include="all").fillna("").to_string()

    if len(numeric_cols) > 1:
        corr = temp_df[numeric_cols].corr().round(3)
        profile["correlation_matrix"] = corr.to_string()
    else:
        profile["correlation_matrix"] = "Not enough numeric columns for correlation."

    # Aggregated group profile: not raw rows, only grouped statistics
    group_candidates = [
        "machine_id", "machine_state", "scenario_type", "material_type",
        "job_type", "supplier", "vendor", "line", "cell", "station"
    ]

    priority_metrics = [
        "downtime", "health_score", "tool_wear", "vibration", "temperature",
        "breakdown_probability", "defect_probability", "cycle_time",
        "queue_length", "power_consumption", "tardiness_increment"
    ]

    group_profiles = {}

    for col in temp_df.columns:
        col_lower = str(col).lower()
        if any(candidate in col_lower for candidate in group_candidates):
            unique_count = temp_df[col].nunique(dropna=True)

            if 1 < unique_count <= 50 and len(numeric_cols) > 0:
                selected_metrics = [m for m in priority_metrics if m in temp_df.columns]
                if not selected_metrics:
                    selected_metrics = numeric_cols[:5]

                try:
                    grouped = temp_df.groupby(col)[selected_metrics].mean(numeric_only=True).round(3)
                    group_profiles[col] = grouped.head(30).to_string()
                except Exception:
                    pass

    profile["aggregated_group_profiles"] = group_profiles

    if privacy_mode:
        privacy_note = """
Privacy Mode is ON. The AI prompt uses only a statistical data profile:
dataset shape, column names or masked column labels, data types, missing counts,
duplicate count, summary statistics, correlation matrix, and aggregated group summaries.
Raw dataset rows are not sent.
"""
    else:
        privacy_note = """
Standard Mode is ON. This version still sends only a summarized data profile to AI.
Raw dataset rows are not included in the prompt.
"""

    return profile, mapping, privacy_note


def add_wrapped_pdf_pages(pdf, title, body, footer="InsightForge | AI Decision Support Report"):
    paragraphs = str(body).split("\n")
    wrapped_lines = []

    for para in paragraphs:
        if para.strip() == "":
            wrapped_lines.append("")
        else:
            wrapped_lines.extend(textwrap.wrap(para, width=95))

    lines_per_page = 42
    chunks = [wrapped_lines[i:i + lines_per_page] for i in range(0, len(wrapped_lines), lines_per_page)]

    if not chunks:
        chunks = [[""]]

    for page_num, chunk in enumerate(chunks, start=1):
        fig = plt.figure(figsize=(8.5, 11))
        plt.axis("off")

        page_title = title if page_num == 1 else f"{title} continued"
        plt.text(0.05, 0.95, page_title, fontsize=17, fontweight="bold", va="top")
        plt.text(0.05, 0.88, "\n".join(chunk), fontsize=9.5, va="top")
        plt.text(0.05, 0.04, footer, fontsize=8, color="gray")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close()


def create_pdf_report(df, executive_text, ai_text, action_text, numeric_cols, readiness_score, privacy_note, problem_type):
    buffer = BytesIO()

    priority_keywords = [
        "health_score", "tool_wear", "temperature", "vibration",
        "downtime", "defect_probability", "breakdown_probability",
        "cycle_time", "power_consumption", "spindle_load", "queue_length"
    ]

    chart_cols = []
    for key in priority_keywords:
        if key in numeric_cols:
            chart_cols.append(key)

    if len(chart_cols) < 3:
        for col in numeric_cols:
            if col not in chart_cols and "id" not in col.lower() and "time_step" not in col.lower():
                chart_cols.append(col)
            if len(chart_cols) >= 3:
                break

    with PdfPages(buffer) as pdf:
        title_body = f"""
Prepared by InsightForge

Report Type: Industrial AI Decision Support Report
Selected Industrial Problem: {problem_type}

Dataset Overview:
Rows: {df.shape[0]}
Columns: {df.shape[1]}
Decision Readiness Score: {readiness_score}/100

Privacy Note:
{privacy_note}

This report summarizes operational data, identifies risks, recommends priority actions,
and supports consulting-style decision-making.
"""
        add_wrapped_pdf_pages(pdf, "InsightForge Industrial Decision Report", title_body)

        overview_body = f"""
Dataset Columns:
{", ".join(map(str, df.columns))}

The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.
This report is designed to support faster operational diagnosis and implementation planning.
"""
        add_wrapped_pdf_pages(pdf, "1. Dataset Overview", overview_body)

        add_wrapped_pdf_pages(pdf, "2. Executive Summary", executive_text)
        add_wrapped_pdf_pages(pdf, "3. Top Recommended Actions", action_text)
        add_wrapped_pdf_pages(pdf, "4. Detailed AI Insights", ai_text)

        if len(chart_cols) > 0:
            for col in chart_cols[:4]:
                fig, ax = plt.subplots(figsize=(8, 5))
                df[col].plot(ax=ax)
                ax.set_title(f"{col} Trend")
                ax.set_xlabel("Index")
                ax.set_ylabel(col)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close()

        closing_body = """
For deeper implementation support, InsightForge can support root cause analysis,
process optimization, quality improvement, predictive maintenance planning,
and custom dashboard development.

Contact: insightforge.ai@gmail.com
"""
        add_wrapped_pdf_pages(pdf, "5. Consulting and Next Steps", closing_body)

    buffer.seek(0)
    return buffer


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🔥 InsightForge")
    st.markdown("**Industrial AI Consulting Engine**")
    st.markdown("---")

    st.markdown("### 🧩 Industrial Problem")
    problem_type = st.selectbox(
        "What problem are you trying to solve?",
        [
            "Reduce Downtime",
            "Improve Quality",
            "Predict Maintenance Risk",
            "Reduce Cost",
            "Improve Delivery Performance",
            "Identify Bottlenecks",
            "Improve Process Stability",
            "General Industrial Analysis"
        ]
    )

    st.markdown("---")
    st.markdown("### 🔐 Privacy Controls")
    privacy_mode = st.toggle("Privacy Mode", value=True)
    mask_columns = st.checkbox("Mask column names before AI analysis", value=False)

    industry_template = st.selectbox(
        "Industry Template",
        [
            "Manufacturing",
            "Quality Analysis",
            "Supply Chain",
            "Lab Testing",
            "General Business"
        ]
    )

    st.markdown("---")
    st.markdown("### Navigation")
    st.markdown("""
    - 📂 Upload Data  
    - 📊 Executive Dashboard  
    - 📈 Visual Analytics  
    - 🧾 AI Transparency  
    - 🧠 AI Decision Engine  
    - 💬 Ask Your Data  
    - 🚀 Consulting  
    """)

    st.markdown("---")
    st.caption("Version 8.0 | Industrial Consulting Engine")


# ---------------- HERO ----------------
st.markdown("""
<div class="hero-shell">
    <div class="hero-content">
        <div class="hero-grid">
            <div>
                <div class="hero-badge">Privacy-First Industrial AI Consulting Engine</div>
                <div style="display:flex; align-items:center;">
                    <div class="logo-mark">🔥</div>
                    <div class="hero-title">InsightForge</div>
                </div>
                <div class="hero-title">Solve Industrial Problems from <span>Operational Data</span>.</div>
                <div class="hero-subtitle">
                    Diagnose downtime, quality risk, maintenance needs, cost drivers,
                    bottlenecks, and process instability.
                </div>
                <div class="hero-text">
                    InsightForge converts operational data into consulting-grade dashboards,
                    action plans, privacy-first AI insights, and decision reports.
                </div>
            </div>
            <div class="floating-panel">
                <div class="panel-row"><span>🧩 Problem Diagnosis</span><span class="panel-value">Built-in</span></div>
                <div class="panel-row"><span>🔐 Privacy Mode</span><span class="panel-value">Available</span></div>
                <div class="panel-row"><span>🧠 AI Consultant</span><span class="panel-value">Active</span></div>
                <div class="panel-row"><span>📄 Report Export</span><span class="panel-value">PDF</span></div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- FEATURE STRIP ----------------
st.markdown("""
<div class="feature-strip">
    <div class="feature-card">
        <div class="feature-icon">🧩</div>
        <div class="feature-title">Problem Solver</div>
        <div class="feature-text">Select downtime, quality, cost, maintenance, bottleneck, or stability problems.</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🔐</div>
        <div class="feature-title">Privacy Mode</div>
        <div class="feature-text">AI uses summarized profiles and aggregated statistics instead of raw rows.</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">💬</div>
        <div class="feature-title">Ask Your Data</div>
        <div class="feature-text">Ask operational questions using the summarized data profile.</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">📄</div>
        <div class="feature-title">Consulting Reports</div>
        <div class="feature-text">Export structured decision reports for managers, engineers, and clients.</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- FILE UPLOAD ----------------
st.markdown('<div class="section-title">📂 Start Your Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Upload a CSV or Excel file to generate your industrial decision dashboard.</div>', unsafe_allow_html=True)

st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- MAIN APP ----------------
if uploaded_file is not None:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    dataset_signature = f"{uploaded_file.name}_{df.shape}_{list(df.columns)}"

    if st.session_state.last_dataset_signature != dataset_signature:
        st.session_state.ai_text = ""
        st.session_state.action_text = ""
        st.session_state.executive_text = ""
        st.session_state.chat_history = []
        st.session_state.last_dataset_signature = dataset_signature

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    missing_values = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    total_cells = df.shape[0] * df.shape[1]
    data_completeness = 100 if total_cells == 0 else round(100 - (missing_values / total_cells * 100), 2)

    sensitive_cols = detect_sensitive_columns(df)
    readiness_score = calculate_decision_readiness(df, numeric_cols, missing_values, duplicate_rows)

    ai_profile, column_mapping, privacy_note = build_ai_profile(
        df,
        privacy_mode=privacy_mode,
        mask_columns=mask_columns
    )

    st.session_state.privacy_note = privacy_note
    problem_focus = get_problem_focus(problem_type)

    # ---------------- PRIVACY PANEL ----------------
    st.markdown("""
    <div class="privacy-card">
        <h3>🔐 Privacy-First Analysis Mode</h3>
        <p>
        InsightForge is designed to reduce exposure of raw data. In AI analysis,
        the app uses a summarized statistical profile and aggregated group summaries
        instead of sending full raw dataset rows.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if sensitive_cols:
        st.markdown("""
        <div class="warning-card">
            <h3>⚠️ Sensitive Column Warning</h3>
            <p>
            Sensitive or business-critical column names were detected. Consider enabling column masking
            before running AI analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.warning(f"Detected columns: {', '.join(map(str, sensitive_cols))}")

    # ---------------- EXECUTIVE DASHBOARD ----------------
    st.markdown('<div class="section-title">📊 Executive Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">A quick health overview of your uploaded dataset.</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        kpi_card("Rows", df.shape[0], "Total records")
    with c2:
        kpi_card("Columns", df.shape[1], "Total features")
    with c3:
        kpi_card("Numeric", len(numeric_cols), "Analyzable metrics")
    with c4:
        kpi_card("Missing", missing_values, "Data gaps")
    with c5:
        kpi_card("Readiness", f"{readiness_score}/100", "Decision score")

    st.markdown("### 🧪 Data Health Score")
    st.progress(data_completeness / 100)
    st.caption("Higher score means the dataset is more complete for analysis and decision-making.")

    st.markdown("### 🎯 Decision Readiness Score")
    st.progress(readiness_score / 100)
    st.caption("Higher score means the dataset is more ready for operational diagnosis and reporting.")

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Data Overview",
        "📈 Visual Dashboard",
        "🧾 AI Transparency",
        "🧠 AI Decision Engine",
        "💬 Ask Your Data",
        "🚀 Consulting"
    ])

    # ---------------- TAB 1 ----------------
    with tab1:
        st.markdown("### 📋 Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)

        st.markdown("### 📈 Summary Statistics")
        st.dataframe(df.describe(include="all"), use_container_width=True)

        st.markdown("### 🧠 Data Quality Snapshot")

        q1, q2, q3 = st.columns(3)

        with q1:
            kpi_card("Duplicate Rows", duplicate_rows, "Repeated records")
        with q2:
            kpi_card("Missing Cells", missing_values, "Blank values")
        with q3:
            kpi_card("Completeness", f"{data_completeness}%", "Overall health")

        if len(numeric_cols) > 0:
            st.markdown("### 🔢 Numeric Metric Cards")
            metric_cols = st.columns(3)

            for i, col in enumerate(numeric_cols):
                with metric_cols[i % 3]:
                    kpi_card(
                        col,
                        f"{df[col].mean():.2f}",
                        f"Min {df[col].min()} | Max {df[col].max()}"
                    )
        else:
            st.warning("No numeric columns found.")

    # ---------------- TAB 2 ----------------
    with tab2:
        st.markdown("### 📈 Visual Analytics Dashboard")

        if len(numeric_cols) > 0:
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
                    st.markdown("#### 🔥 Correlation Heatmap")
                    corr = df[numeric_cols].corr()
                    fig, ax = plt.subplots(figsize=(8, 5))
                    heatmap = ax.imshow(corr, aspect="auto")

                    ax.set_xticks(range(len(corr.columns)))
                    ax.set_yticks(range(len(corr.columns)))
                    ax.set_xticklabels(corr.columns, rotation=90)
                    ax.set_yticklabels(corr.columns)

                    for i in range(len(corr.columns)):
                        for j in range(len(corr.columns)):
                            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)

                    ax.set_title("Correlation Heatmap")
                    fig.colorbar(heatmap, ax=ax)
                    st.pyplot(fig)
                else:
                    st.info("Need at least two numeric columns for correlation heatmap.")

            st.markdown("### 🎛️ Custom Chart Builder")
            custom1, custom2 = st.columns(2)

            with custom1:
                selected_column = st.selectbox("Select metric", numeric_cols, key="custom_metric")

            with custom2:
                plot_type = st.selectbox("Choose chart type", ["Line", "Histogram", "Box"], key="custom_chart")

            fig, ax = plt.subplots(figsize=(10, 4.5))

            if plot_type == "Line":
                df[selected_column].plot(ax=ax)
            elif plot_type == "Histogram":
                df[selected_column].plot(kind="hist", ax=ax, bins=20)
            else:
                df[selected_column].plot(kind="box", ax=ax)

            ax.set_title(f"{selected_column} - {plot_type}")
            ax.set_xlabel("Index")
            ax.set_ylabel(selected_column)
            st.pyplot(fig)

        else:
            st.warning("No numeric columns found for dashboard charts.")

    # ---------------- TAB 3 ----------------
    with tab3:
        st.markdown("### 🧾 AI Transparency Panel")

        st.success("Raw dataset rows are not sent to the AI prompt in this version.")

        st.markdown("#### What AI receives")
        st.markdown("""
        - Dataset shape  
        - Column names or masked column labels  
        - Data types  
        - Missing-value counts  
        - Duplicate-row count  
        - Summary statistics  
        - Correlation matrix  
        - Aggregated group summaries  
        """)

        st.markdown("#### Current Settings")
        st.write(f"Industrial Problem: **{problem_type}**")
        st.write(f"Privacy Mode: **{'ON' if privacy_mode else 'OFF'}**")
        st.write(f"Column Masking: **{'ON' if mask_columns else 'OFF'}**")
        st.write(f"Industry Template: **{industry_template}**")

        if mask_columns and column_mapping:
            with st.expander("View local column masking map"):
                st.write(column_mapping)

        with st.expander("Preview summarized AI profile"):
            st.write(ai_profile)

    # ---------------- TAB 4 ----------------
    with tab4:
        st.markdown("### 🧠 AI Decision Engine")
        st.info("Run AI analysis when ready. AI receives the summarized profile shown in the transparency panel.")

        run_ai = st.button("🚀 Run AI Analysis", type="primary")

        if run_ai:
            with st.spinner("Generating executive summary..."):
                executive_prompt = f"""
You are a senior industrial operations consultant.

Write a short executive summary for a manager based on this summarized dataset profile.

Selected industrial problem:
{problem_type}

Problem focus:
{problem_focus}

Privacy note:
{privacy_note}

Summarized dataset profile:
{ai_profile}

Output:
- 4 to 6 bullet points
- Clear business language
- No generic claims
- Do not say you saw raw rows
"""

                executive_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": executive_prompt}]
                )

                st.session_state.executive_text = executive_response.choices[0].message.content

            with st.spinner("Generating top recommended actions..."):
                action_prompt = f"""
You are a practical {industry_template} business and engineering consultant.

Based on this summarized dataset profile, provide exactly the TOP 3 actions the user should take next.

Selected industrial problem:
{problem_type}

Problem focus:
{problem_focus}

Rules:
- Make each action specific and practical.
- Avoid generic advice.
- Explain why each action matters.
- Include expected business or engineering impact.
- Do not claim you saw raw rows.
- Base your response only on the summarized profile.

Privacy note:
{privacy_note}

Summarized dataset profile:
{ai_profile}
"""

                action_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": action_prompt}]
                )

                st.session_state.action_text = action_response.choices[0].message.content

            with st.spinner("Generating detailed AI insights..."):
                insight_prompt = f"""
You are an expert {industry_template} analytics consultant.

Analyze the summarized dataset profile and provide:
1. Key problems
2. Possible trends or risks
3. Variability and stability concerns
4. Business or engineering impact
5. Practical next steps
6. Implementation roadmap

Selected industrial problem:
{problem_type}

Problem focus:
{problem_focus}

Do not claim you saw raw rows.
Use the privacy note and summarized profile only.

Privacy note:
{privacy_note}

Summarized dataset profile:
{ai_profile}
"""

                insight_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": insight_prompt}]
                )

                st.session_state.ai_text = insight_response.choices[0].message.content

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
            pdf_report = create_pdf_report(
                df,
                st.session_state.executive_text,
                st.session_state.ai_text,
                st.session_state.action_text,
                numeric_cols,
                readiness_score,
                privacy_note,
                problem_type
            )

            st.download_button(
                label="📥 Download InsightForge Industrial Report",
                data=pdf_report.getvalue(),
                file_name="InsightForge_Industrial_Report.pdf",
                mime="application/pdf"
            )
        else:
            st.warning("Run AI Analysis first to generate the report.")

    # ---------------- TAB 5 ----------------
    with tab5:
        st.markdown("### 💬 Ask Your Data")
        st.caption("Ask questions using the summarized privacy-safe data profile. Raw rows are not sent.")

        suggested_questions = [
            "What should we improve first?",
            "Which machine or group appears highest risk?",
            "What is likely causing downtime?",
            "Which factors seem related to quality problems?",
            "Is this data ready for predictive modeling?",
            "What should management do next?"
        ]

        selected_question = st.selectbox("Choose a suggested question", [""] + suggested_questions)
        user_question = st.text_input("Or type your own question")

        final_question = user_question.strip() if user_question.strip() else selected_question

        ask_button = st.button("Ask InsightForge")

        if ask_button and final_question:
            with st.spinner("Thinking through your data profile..."):
                chat_prompt = f"""
You are InsightForge, an industrial AI consulting assistant.

Answer the user's question using only the summarized dataset profile below.
Do not claim you saw raw rows.
If the profile is insufficient, say what additional data would be needed.

Selected industrial problem:
{problem_type}

Problem focus:
{problem_focus}

User question:
{final_question}

Privacy note:
{privacy_note}

Summarized dataset profile:
{ai_profile}
"""

                chat_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": chat_prompt}]
                )

                answer = chat_response.choices[0].message.content
                st.session_state.chat_history.append((final_question, answer))

        if st.session_state.chat_history:
            st.markdown("### Conversation")
            for q, a in reversed(st.session_state.chat_history):
                st.markdown(f"**You:** {q}")
                st.markdown(f"**InsightForge:** {a}")
                st.markdown("---")

    # ---------------- TAB 6 ----------------
    with tab6:
        st.markdown("""
        <div class="consulting-card">
            <h2>🚀 Need Expert Help?</h2>
            <p>
            InsightForge gives an initial AI-based diagnosis. For deeper implementation,
            expert review can help with root cause analysis, process optimization,
            cost reduction, quality improvement, predictive maintenance planning,
            and custom dashboard development.
            </p>
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
        <b>InsightForge</b> | Privacy-first industrial AI consulting, dashboard intelligence, decision support, and implementation pathway.
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("Upload a CSV or Excel file to begin analysis.")
