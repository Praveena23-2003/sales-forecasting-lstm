import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# ========== Page Setup ==========
st.set_page_config(page_title="📊 Sales Forecasting Dashboard", layout="wide")
st.title("📈 Sales Forecasting with LSTM + Sentiment Analysis")

# ========== File Paths ==========
# NEW (use these if files are in root directory)
forecast_df = pd.read_csv("lstm_forecast.csv")
forecast_plot = "lstm_forecast_plot.png"

# ========== Load Forecast Data ==========
df = None
if os.path.exists(forecast_csv_path):
    df = pd.read_csv(forecast_csv_path, parse_dates=["ds"])

# ========== Show Forecast Plot ==========
if os.path.exists(forecast_plot_path):
    st.subheader("🖼️ Forecast Plot")
    st.image(forecast_plot_path)
else:
    st.warning("⚠️ Forecast plot not found.")

# ========== Forecast Data & Visuals ==========
if df is not None:
    st.subheader("📄 Forecast Data (Last 10 rows)")
    st.dataframe(df.tail(10))

    st.subheader("📅 Filter by Date Range")
    start_date = st.date_input("Start Date", df["ds"].min().date())
    end_date = st.date_input("End Date", df["ds"].max().date())
    filtered_df = df[(df["ds"] >= pd.to_datetime(start_date)) & (df["ds"] <= pd.to_datetime(end_date))]

    st.subheader("📈 Forecast Visualization (Filtered)")
    fig, ax = plt.subplots(figsize=(12, 5))
    if "actual" in filtered_df.columns:
        ax.plot(filtered_df["ds"], filtered_df["actual"], label="Actual Sales", color="blue")
    ax.plot(filtered_df["ds"], filtered_df["forecast"], label="Predicted Sales", color="orange")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.set_title("Actual vs Forecasted Sales")
    ax.legend()
    st.pyplot(fig)

    # ========== Forecast Summary ==========
    st.subheader("📌 Forecast Summary")
    st.write(f"🔹 Total Days: {len(df)}")
    st.write(f"🔹 Max Forecasted Sales: {df['forecast'].max():,.2f}")
    st.write(f"🔹 Min Forecasted Sales: {df['forecast'].min():,.2f}")
    if "actual" in df.columns:
        mae = abs(df["actual"] - df["forecast"]).mean()
        st.write(f"🔹 Mean Absolute Error (MAE): {mae:,.2f}")

    # ========== AI-style Insight (No API) ==========
    st.markdown("---")
    st.subheader("🤖 AI-Style Forecast Insight (No API)")
    latest_30 = df.tail(30)
    max_val = latest_30["forecast"].max()
    min_val = latest_30["forecast"].min()
    avg_val = latest_30["forecast"].mean()
    trend = "increasing" if latest_30["forecast"].iloc[-1] > latest_30["forecast"].iloc[0] else (
        "decreasing" if latest_30["forecast"].iloc[-1] < latest_30["forecast"].iloc[0] else "stable"
    )

    insight_text = (
        f"📊 The forecast shows a **{trend}** trend in the last 30 days.\n"
        f"- 🔺 Highest predicted sales: **{max_val:.2f}**\n"
        f"- 🔻 Lowest predicted sales: **{min_val:.2f}**\n"
        f"- 📉 Average predicted sales: **{avg_val:.2f}**\n\n"
        "🧠 **Recommendation**: Continue strategies for high-sale periods. Investigate causes of dips."
    )
    st.success("✅ Insight Generated:")
    st.markdown(insight_text)

    st.download_button(
        label="📥 Download Insight",
        data=insight_text,
        file_name="forecast_insight.txt",
        mime="text/plain"
    )

    # ========== Anomaly Detection ==========
    st.markdown("---")
    st.subheader("🚨 Anomaly Detection (Unusual Spikes or Drops)")
    threshold = st.slider("Set Anomaly Threshold (std deviation)", 1.0, 3.0, 2.0)
    mean = df["forecast"].mean()
    std = df["forecast"].std()
    df["anomaly"] = abs(df["forecast"] - mean) > threshold * std
    anomalies = df[df["anomaly"]]

    st.write(f"Detected {len(anomalies)} anomalies (based on threshold: {threshold:.1f} std)")
    st.dataframe(anomalies[["ds", "forecast"]])

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(df["ds"], df["forecast"], label="Forecast", color="gray")
    ax2.scatter(anomalies["ds"], anomalies["forecast"], color="red", label="Anomalies")
    ax2.set_title("Anomaly Detection in Forecast")
    ax2.legend()
    st.pyplot(fig2)

    # ========== Monthly Aggregation ==========
    st.markdown("---")
    st.subheader("📆 Monthly Average Forecast")
    df["month"] = df["ds"].dt.to_period("M")
    monthly_avg = df.groupby("month")["forecast"].mean().reset_index()
    monthly_avg["month"] = monthly_avg["month"].astype(str)

    fig3, ax3 = plt.subplots(figsize=(12, 5))
    sns.barplot(x="month", y="forecast", hue="month", data=monthly_avg, ax=ax3, palette="viridis", legend=False)
    ax3.set_title("Monthly Average Forecasted Sales")
    ax3.set_ylabel("Avg Sales")
    ax3.set_xlabel("Month")
    ax3.tick_params(axis='x', rotation=45)
    st.pyplot(fig3)

    # ========== Top & Bottom 5 Days ==========
    st.markdown("---")
    st.subheader("🏆 Top 5 and Bottom 5 Days by Forecast")
    top5 = df.sort_values("forecast", ascending=False).head(5)
    bottom5 = df.sort_values("forecast", ascending=True).head(5)
    col1, col2 = st.columns(2)
    with col1:
        st.write("🔝 Top 5 Days")
        st.dataframe(top5[["ds", "forecast"]])
    with col2:
        st.write("🔻 Bottom 5 Days")
        st.dataframe(bottom5[["ds", "forecast"]])

    # ========== Volatility ==========
    st.markdown("---")
    st.subheader("📉 Sales Volatility (Rolling Std Dev)")
    df["rolling_std"] = df["forecast"].rolling(window=7).std()
    fig4, ax4 = plt.subplots(figsize=(12, 5))
    ax4.plot(df["ds"], df["rolling_std"], label="7-day Rolling Std Dev", color="purple")
    ax4.set_title("Sales Volatility Over Time")
    ax4.set_ylabel("Rolling Std Dev")
    st.pyplot(fig4)

else:
    st.error("❌ Forecast data not available.")