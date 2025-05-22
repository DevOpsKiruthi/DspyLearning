import streamlit as st
import os
import dspy
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List
from pydantic import BaseModel
import plotly.graph_objects as go
from config import azure_llm

# Set page configuration
st.set_page_config(page_title="Gold Price Range Predictor", layout="wide")

# -----------------------
# 1. USER LOGIN AUTH
# -----------------------

# Dummy users (username: password)
users = {"admin": "#6bwBcoe&ZuxH4dH38d", "demo": "#6bwBcoe&ZuxH4dH38d"}


def login():
    st.sidebar.title("🔐 Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        if username in users and users[username] == password:
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.rerun()
        else:
            st.sidebar.error("Invalid username or password")


def logout():
    st.sidebar.button("Logout", on_click=lambda: st.session_state.clear())


# Login logic
if "authenticated" not in st.session_state:
    login()
    st.stop()
else:
    st.sidebar.success(f"Logged in as {st.session_state['username']}")
    logout()

# -----------------------
# 2. GOLD PRICE PREDICTOR
# -----------------------


if 'dspy_configured' not in st.session_state:
    try:
        dspy.settings.configure(lm=azure_llm)
    except RuntimeError:
        pass  # Ignore the error
    st.session_state['dspy_configured'] = True


# Pydantic models for structured prediction output
class PriceRange(BaseModel):
    lower_bound: float
    upper_bound: float
    confidence: float  # 0-1 scale


class MarketFactors(BaseModel):
    bullish_factors: List[str]
    bearish_factors: List[str]
    key_indicators: List[str]


class GoldPrediction(BaseModel):
    current_price: float
    prediction_timeframe: str  # e.g., "24 hours", "1 week", "1 month"
    predicted_range: PriceRange
    market_analysis: MarketFactors
    recommendation: str  # e.g., "Buy", "Hold", "Sell"
    explanation: str


# DSPy Signature for gold price prediction
class GoldPricePredictionSignature(dspy.Signature):
    current_price: float = dspy.InputField(desc="Current gold price")
    historical_data: str = dspy.InputField(
        desc="Recent gold price historical data and trends"
    )
    market_context: str = dspy.InputField(
        desc="Current market conditions and news affecting gold"
    )
    timeframe: str = dspy.InputField(desc="Prediction timeframe (24h, 1 week, 1 month)")
    prediction: GoldPrediction = dspy.OutputField(
        desc="Structured gold price prediction with analysis"
    )


# Initialize DSPy predictor
gold_predictor = dspy.ChainOfThought(GoldPricePredictionSignature)

# Function to fetch gold price data
def fetch_gold_data(days=90):
    try:
        # Gold ETF as proxy for gold prices - GLD
        ticker = "GLD"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        data = yf.download(ticker, start=start_date, end=end_date)
        return data
    except Exception as e:
        st.error(f"Error fetching gold data: {e}")
        return pd.DataFrame()


# Function to get latest news and market context
def get_market_context():
    # In a real application, this would fetch actual news and market indicators
    # For now, we'll return placeholder text
    return """
    Recent market factors affecting gold:
    - Federal Reserve's latest interest rate decision
    - US Dollar strength/weakness
    - Global inflation trends
    - Geopolitical tensions
    - Recent gold demand for jewelry and industry
    - Central bank gold purchases
    """


# Streamlit UI
st.title("🥇 Gold Price Range Predictor")
st.write(
    "Predict the next price range for gold based on current price and market conditions."
)

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    prediction_timeframe = st.selectbox(
        "Prediction Timeframe", ["24 hours", "1 week", "1 month"], index=1
    )

    historical_days = st.slider(
        "Historical Data (days)", min_value=30, max_value=365, value=90
    )

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Historical Gold Price")

    # Fetch historical data
    with st.spinner("Fetching historical gold data..."):
        gold_data = fetch_gold_data(days=historical_days)

    if not gold_data.empty:
        # Display chart
        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=gold_data.index,
                open=gold_data["Open"],
                high=gold_data["High"],
                low=gold_data["Low"],
                close=gold_data["Close"],
                name="Gold ETF (GLD)",
            )
        )

        fig.update_layout(
            title="Gold ETF (GLD) Price Chart",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Current price
        if not gold_data.empty:
            current_price = float(
                gold_data["Close"].iloc[-1]
            )  # Convert to float explicitly
            price_change = float(
                gold_data["Close"].iloc[-1] - gold_data["Close"].iloc[-2]
            )
            percent_change = float(
                (gold_data["Close"].iloc[-1] / gold_data["Close"].iloc[-2] - 1) * 100
            )  # Add float conversion here

            st.metric(
                "Current GLD Price",
                f"${current_price:.2f}",
                f"{price_change:.2f} ({percent_change:.2f}%)",
            )
    else:
        st.error("Failed to fetch gold price data. Please try again later.")
        current_price = None

with col2:
    st.subheader("🔮 Predict Next Price Range")

    # Manual price input option
    use_manual_price = st.checkbox("Enter price manually")

    if use_manual_price:
        manual_price = st.number_input(
            "Current Gold Price (USD)",
            value=float(current_price) if current_price else 1800.0,
        )
        current_price = manual_price

    # Market context
    st.text_area(
        "Market Context (editable)",
        value=get_market_context(),
        height=200,
        key="market_context",
    )

    # Prediction button
    if st.button("🔍 Predict Price Range", type="primary"):
        if current_price:
            with st.spinner("Analyzing gold market and generating prediction..."):
                try:
                    # Prepare historical data summary
                    if not gold_data.empty:
                        min_low = float(gold_data['Low'].min())
                        max_high = float(gold_data['High'].max())
                        mean_close = float(gold_data['Close'].mean())
                        
                        recent_trend = f"Past {historical_days} days: Low=${min_low:.2f}, High=${max_high:.2f}, Avg=${mean_close:.2f}"
                        
                        if len(gold_data) >= 30:
                            month_change = float((gold_data["Close"].iloc[-1] / gold_data["Close"].iloc[-30] - 1) * 100)
                        else:
                            month_change = 0.0
                        
                        recent_trend += f", 30-day change: {month_change:.2f}%"
                    else:
                        recent_trend = "No historical data available"

                    # Get prediction from LLM
                    result = gold_predictor(
                        current_price=current_price,
                        historical_data=recent_trend,
                        market_context=st.session_state.market_context,
                        timeframe=prediction_timeframe,
                    )

                    prediction = result.prediction

                    # Display prediction results
                    st.success(
                        f"Prediction Complete for {prediction_timeframe} timeframe"
                    )

                    # Prediction range
                    st.metric(
                        "Predicted Range",
                        f"${prediction.predicted_range.lower_bound:.2f} - ${prediction.predicted_range.upper_bound:.2f}",
                        f"Confidence: {prediction.predicted_range.confidence * 100:.0f}%",
                    )

                    # Recommendation
                    recommendation_color = {
                        "Buy": "green",
                        "Hold": "blue",
                        "Sell": "red",
                    }.get(prediction.recommendation, "black")

                    st.markdown(
                        f"**Recommendation:** <span style='color:{recommendation_color};font-weight:bold'>{prediction.recommendation}</span>",
                        unsafe_allow_html=True,
                    )

                    # Factors affecting prediction
                    with st.expander("Market Factors Analysis", expanded=True):
                        st.markdown("### Bullish Factors")
                        for factor in prediction.market_analysis.bullish_factors:
                            st.markdown(f"✅ {factor}")

                        st.markdown("### Bearish Factors")
                        for factor in prediction.market_analysis.bearish_factors:
                            st.markdown(f"⛔ {factor}")

                        st.markdown("### Key Indicators to Watch")
                        for indicator in prediction.market_analysis.key_indicators:
                            st.markdown(f"👁️ {indicator}")

                    # Detailed explanation
                    with st.expander("Detailed Analysis"):
                        st.write(prediction.explanation)

                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
        else:
            st.warning(
                "Please ensure gold price data is available or enter a manual price."
            )

# Historical predictions tracking
st.subheader("📜 Past Predictions")
st.info(
    "This section would track your previous predictions and their accuracy. Feature coming soon!"
)

# Additional resources
with st.expander("📚 Gold Trading Resources"):
    st.markdown(
        """
    - **Key Gold Price Drivers:**
      - Interest rates and monetary policy
      - US Dollar strength
      - Inflation rates
      - Geopolitical uncertainty
      - Supply and demand fundamentals
    
    - **Common Technical Indicators for Gold:**
      - Moving Averages (50-day, 200-day)
      - Relative Strength Index (RSI)
      - MACD (Moving Average Convergence Divergence)
      - Fibonacci Retracement levels
    """
    )

# Footer
st.markdown("---")
st.caption(
    "Gold Price Range Predictor - Data from Yahoo Finance - Not financial advice"
)
