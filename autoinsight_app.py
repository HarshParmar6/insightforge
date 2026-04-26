import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
import textwrap

# ---------------- PAGE SETUP ----------------
st.set_page_config(
    page_title="InsightForge",
    page_icon="🔥",
    layout="wide"
)

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

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
    from {
        opacity: 0;
        transform: translateY(28px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes floatCard {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-12px); }
    100% { transform: translateY(0px); }
}

@keyframes glowPulse {
    0% { box-shadow: 0 0 20px rgba(37, 99, 235, 0.25); }
    50% { box-shadow: 0 0 42px rgba(37, 99, 235, 0.50); }
    100% { box-shadow: 0 0 20px rgba(37, 99, 235, 0.25); }
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

.hero-shell:after {
    content: "";
    position: absolute;
    width: 260px;
    height: 260px;
    left: -90px;
    bottom: -120px;
    background: radial-gradient(circle, rgba(14,165,233,0.35), rgba(14,165,233,0.03));
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
    letter-spacing: 0.3px;
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

.info-card {
    background: white;
    border-radius: 20px;
    padding: 24px;
    border: 1px solid #E5E7EB;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.07);
    margin-bottom: 20px;
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
    .hero-grid {
        grid-template-columns: 1fr;
    }
    .feature-strip {
        grid-template-columns: 1fr;
    }
    .hero-title {
        font-size: 42px;
    }
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


def create_pdf_report(df, ai_text, action_text, numeric_cols):
    buffer = BytesIO()

    with PdfPages(buffer) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        plt.axis("off")

        text = f"""
InsightForge AI Report

Dataset Overview
Rows: {df.shape[0]}
Columns: {df.shape[1]}

Column Names:
{", ".join(df.columns)}

Top 3 Recommended Actions:
{action_text}

AI Insights:
{ai_text}
"""

        wrapped_text = "\n".join(textwrap.wrap(text, width=95))
        plt.text(0.05, 0.95, wrapped_text, va="top", fontsize=9)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        if len(numeric_cols) > 0:
            for col in numeric_cols[:3]:
                fig, ax = plt.subplots(figsize=(8, 5))
                df[col].plot(ax=ax)
                ax.set_title(f"{col} Trend")
                ax.set_xlabel("Index")
                ax.set_ylabel(col)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close()

    buffer.seek(0)
    return buffer


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🔥 InsightForge")
    st.markdown("**AI Decision Support Platform**")
    st.markdown("---")

    st.markdown("### Navigation")
    st.markdown("""
    - 📂 Upload Data  
    - 📊 Executive Dashboard  
    - 📈 Visual Analytics  
    - 🤖 AI Decision Engine  
    - 🚀 Consulting  
    """)

    st.markdown("---")
    st.markdown("### Built for")
    st.markdown("""
    - Manufacturing SMEs  
    - Engineering teams  
    - Lab testing data  
    - Quality analysis  
    - Operations data  
    """)

    st.markdown("---")
    st.caption("Version 6.0 | Animated Landing UI")


# ---------------- ANIMATED HERO LANDING ----------------
st.markdown("""
<div class="hero-shell">
    <div class="hero-content">
        <div class="hero-grid">
            <div>
                <div class="hero-badge">AI + Analytics + Decision Support</div>
                <div class="hero-title">Forge Better Decisions from <span>Raw Data</span>.</div>
                <div class="hero-subtitle">
                    InsightForge turns datasets into executive dashboards, AI recommendations,
                    risk signals, and professional reports.
                </div>
                <div class="hero-text">
                    Designed for engineering teams, manufacturing SMEs, labs, and operators who
                    need faster decisions without manually cleaning spreadsheets or building reports.
                </div>
            </div>
            <div class="floating-panel">
                <div class="panel-row"><span>AI Recommendations</span><span class="panel-value">Active</span></div>
                <div class="panel-row"><span>Dashboard Analytics</span><span class="panel-value">Ready</span></div>
                <div class="panel-row"><span>Report Export</span><span class="panel-value">PDF</span></div>
                <div class="panel-row"><span>Consulting Pathway</span><span class="panel-value">Built-in</span></div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- FEATURE STRIP ----------------
st.markdown("""
<div class="feature-strip">
    <div class="feature-card">
        <div class="feature-icon">📊</div>
        <div class="feature-title">Executive Dashboard</div>
        <div class="feature-text">Instant KPIs for rows, columns, missing values, and data health.</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🤖</div>
        <div class="feature-title">AI Decision Engine</div>
        <div class="feature-text">Generate top actions, risks, trends, and decision recommendations.</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">📈</div>
        <div class="feature-title">Visual Analytics</div>
        <div class="feature-text">Explore trends, distributions, spread, and correlations in one place.</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">📄</div>
        <div class="feature-title">PDF Reports</div>
        <div class="feature-text">Export professional decision-support reports for teams and clients.</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- FILE UPLOAD ----------------
st.markdown('<div class="section-title">📂 Start Your Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Upload a CSV or Excel file to generate your decision dashboard.</div>', unsafe_allow_html=True)

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

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    missing_values = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    total_cells = df.shape[0] * df.shape[1]
    data_completeness = 100 if total_cells == 0 else round(100 - (missing_values / total_cells * 100), 2)

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
        kpi_card("Completeness", f"{data_completeness}%", "Data quality score")

    st.markdown("### 🧪 Data Health Score")
    st.progress(data_completeness / 100)
    st.caption("Higher score means the dataset is more complete for analysis and decision-making.")

    st.markdown("---")

    # ---------------- TABS ----------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Data Overview",
        "📈 Visual Dashboard",
        "🤖 AI Decision Engine",
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
                st.markdown('<div class="info-card">', unsafe_allow_html=True)
                metric1 = st.selectbox("Trend Metric", numeric_cols, key="line_metric")
                fig, ax = plt.subplots(figsize=(8, 4))
                df[metric1].plot(ax=ax)
                ax.set_title(f"{metric1} Trend")
                ax.set_xlabel("Index")
                ax.set_ylabel(metric1)
                st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)

            with r1c2:
                st.markdown('<div class="info-card">', unsafe_allow_html=True)
                metric2 = st.selectbox("Distribution Metric", numeric_cols, key="hist_metric")
                fig, ax = plt.subplots(figsize=(8, 4))
                df[metric2].plot(kind="hist", ax=ax, bins=20)
                ax.set_title(f"{metric2} Distribution")
                ax.set_xlabel(metric2)
                st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)

            r2c1, r2c2 = st.columns(2)

            with r2c1:
                st.markdown('<div class="info-card">', unsafe_allow_html=True)
                metric3 = st.selectbox("Outlier / Spread Metric", numeric_cols, key="box_metric")
                fig, ax = plt.subplots(figsize=(8, 4))
                df[metric3].plot(kind="box", ax=ax)
                ax.set_title(f"{metric3} Spread / Outlier View")
                st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)

            with r2c2:
                st.markdown('<div class="info-card">', unsafe_allow_html=True)

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
                            ax.text(
                                j, i, f"{corr.iloc[i, j]:.2f}",
                                ha="center",
                                va="center",
                                fontsize=7
                            )

                    ax.set_title("Correlation Heatmap")
                    fig.colorbar(heatmap, ax=ax)
                    st.pyplot(fig)
                else:
                    st.info("Need at least two numeric columns for correlation heatmap.")

                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("### 🎛️ Custom Chart Builder")

            custom1, custom2 = st.columns(2)

            with custom1:
                selected_column = st.selectbox("Select metric", numeric_cols, key="custom_metric")

            with custom2:
                plot_type = st.selectbox("Choose chart type", ["Line", "Histogram", "Box"], key="custom_chart")

            st.markdown('<div class="info-card">', unsafe_allow_html=True)

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

            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.warning("No numeric columns found for dashboard charts.")

    # ---------------- TAB 3 ----------------
    with tab3:
        st.markdown("### 🤖 AI Decision Engine")
        st.info("Run AI analysis when you are ready. This prevents repeated API calls during dashboard interaction.")

        run_ai = st.button("🚀 Run AI Analysis", type="primary")

        if run_ai:
            with st.spinner("Generating top recommended actions..."):
                action_prompt = f"""
You are a practical business and engineering consultant.

Based on this dataset summary, provide exactly the TOP 3 actions the user should take next.

Rules:
- Make each action specific and practical.
- Avoid generic advice.
- Explain why each action matters.
- Use simple business language.

Dataset columns:
{list(df.columns)}

Summary statistics:
{df.describe(include='all').to_string()}
"""

                action_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": action_prompt}]
                )

                action_text = action_response.choices[0].message.content

            with st.spinner("Generating detailed AI insights..."):
                insight_prompt = f"""
You are an expert manufacturing and data analytics consultant.

Analyze this dataset and provide:
1. Key problems in the data
2. Possible trends or risks
3. Variability and stability concerns
4. Business or engineering impact
5. Practical next steps

Dataset columns:
{list(df.columns)}

Summary statistics:
{df.describe(include='all').to_string()}
"""

                insight_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": insight_prompt}]
                )

                ai_text = insight_response.choices[0].message.content

            left, right = st.columns(2)

            with left:
                st.markdown("### 🎯 Top 3 Recommended Actions")
                st.markdown(f"""
                <div class="info-card">
                {action_text}
                </div>
                """, unsafe_allow_html=True)

            with right:
                st.markdown("### 🧠 Detailed AI Insights")
                st.markdown(f"""
                <div class="info-card">
                {ai_text}
                </div>
                """, unsafe_allow_html=True)

            st.markdown("### 📄 Export Report")
            pdf_report = create_pdf_report(df, ai_text, action_text, numeric_cols)

            st.download_button(
                label="📥 Download InsightForge AI Report",
                data=pdf_report,
                file_name="InsightForge_AI_Report.pdf",
                mime="application/pdf"
            )

    # ---------------- TAB 4 ----------------
    with tab4:
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
        <b>InsightForge</b> | AI-powered analytics, dashboard intelligence, decision support, and consulting pathway.
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("Upload a CSV or Excel file to begin analysis.")
