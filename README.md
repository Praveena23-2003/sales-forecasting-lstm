📈 Sales Forecasting with LSTM + Sentiment Analysis
This project demonstrates a machine learning-based approach for sales forecasting using LSTM (Long Short-Term Memory) neural networks, enhanced with basic sentiment analysis insights. Built with Python and Streamlit, it provides an interactive dashboard to explore, analyze, and visualize time series sales data.

🚀 Features
- 🖼️ Visual Forecast Plot (LSTM Predictions)
- 📅 Filter Forecasts by Date Range
- 📈 Actual vs Predicted Sales Line Graph
- 📌 Forecast Summary (Min, Max, MAE)
- 🤖 AI-Style Forecast Insight (No API required)
- 🚨 Anomaly Detection (Spikes/Drops)
- 📆 Monthly Average Forecast Visualization
- 🏆 Top & Bottom 5 Sales Days
- 📉 Rolling Sales Volatility (Std Deviation)
- 📥 Downloadable Forecast Summary

🛠️ Tech Stack
- Python
- Streamlit
- Pandas, NumPy
- Matplotlib, Seaborn
- LSTM (pre-trained)
- CSV-based Data Pipeline
- 
📁 Project Structure
sales_forecasting_lstm/ │ ├── app.py # Streamlit App ├── outputs/ │ ├── lstm_forecast.csv # Forecast data (with ds, forecast, actual) │ └── lstm_forecast_plot.png # Static Forecast Plot

▶️ How to Run Locally
Clone the Repo

git clone https://github.com/Praveena23-2003/sales-forecasting-lstm.git
 cd sales-forecasting-lstm Create Virtual Environment (Optional)

python -m venv venv venv\Scripts\activate # On Windows Install Dependencies

pip install -r requirements.txt 
Run Streamlit App

streamlit run app.py 🧪 Sample Output Feature Output Example Total Days 364 Max Forecasted Sales 30.47 Min Forecasted Sales 13.01 Mean Absolute Error 4.61 Detected Anomalies 7 Trend Insight Decreasing trend

📌 Insights LSTM effectively models sequential sales patterns.

AI-style summaries provide business-friendly interpretations.

Anomaly detection helps identify sudden sales dips or spikes.

Rolling volatility gives clarity on sales consistency.

🙋‍♀️ Author Praveena R 🎓 MCA Student | 📊 AI & Data Analytics Enthusiast
