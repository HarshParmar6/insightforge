import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
import textwrap

st.set_page_config(page_title="InsightForge", layout="wide")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("🔥 InsightForge")
st.subheader("Transform Data into Actionable Intelligence")
st.markdown("---")

uploaded_file = st.file_uploader("📂 Upload your data file", type=["csv", "xlsx"])

def create_pdf_report(df, ai_text, numeric_cols):
    buffer = BytesIO()

    with PdfPages(buffer) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        plt.axis("off")

        text = f"""
InsightForge Report

Dataset Shape:
Rows: {df.shape[0]}
Columns: {df.shape[1]}

Columns:
{", ".join(df.columns)}

AI Insights:
{ai_text}
"""

        wrapped_text = "\n".join(textwrap.wrap(text, width=90))
        plt.text(0.05, 0.95, wrapped_text, va="top", fontsize=10)
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

if uploaded_file is not None:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.markdown("## 📊 Data Preview")
    st.dataframe(df.head())

    st.markdown("## 📈 Summary Statistics")
    st.dataframe(df.describe(include="all"))

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    st.markdown("## 🤖 AI Insights")

    prompt = f"""
You are an expert manufacturing and data analytics consultant.

Analyze this dataset and provide insights like you are advising an engineering team.

Focus on:
1. Process efficiency
2. Variability and stability
3. Possible anomalies or risks
4. Recommendations for improvement
5. Any patterns useful for predictive maintenance

Dataset columns:
{list(df.columns)}

Summary statistics:
{df.describe(include='all').to_string()}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    ai_text = response.choices[0].message.content
    st.write(ai_text)

    st.markdown("## 📄 Download Report")

    pdf_report = create_pdf_report(df, ai_text, numeric_cols)

    st.download_button(
        label="Download PDF Report",
        data=pdf_report,
        file_name="InsightForge_Report.pdf",
        mime="application/pdf"
    )

    st.markdown("## 🧠 Key Numeric Insights")

    if len(numeric_cols) > 0:
        for col in numeric_cols:
            st.success(
                f"{col}: Avg = {df[col].mean():.2f}, "
                f"Max = {df[col].max()}, "
                f"Min = {df[col].min()}"
            )

        st.markdown("## 📉 Data Visualization")

        column = st.selectbox("Select metric", numeric_cols)
        plot_type = st.selectbox("Choose chart type", ["Line", "Histogram", "Box"])

        fig, ax = plt.subplots()

        if plot_type == "Line":
            df[column].plot(ax=ax)
        elif plot_type == "Histogram":
            df[column].plot(kind="hist", ax=ax)
        else:
            df[column].plot(kind="box", ax=ax)

        ax.set_title(f"{column} - {plot_type}")
        ax.set_xlabel("Index")
        ax.set_ylabel(column)

        st.pyplot(fig)

    else:
        st.warning("No numeric columns found for analysis or plotting.")

else:
    st.info("Upload a CSV or Excel file to begin analysis.")