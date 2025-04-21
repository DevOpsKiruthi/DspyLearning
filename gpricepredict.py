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
users = {
    "admin": "#6bwBcoe&ZuxH4dH38d",
    "demo": "#6bwBcoe&ZuxH4dH38d"
}

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

# Configure DSPy for LLM-based analysis
dspy.settings.configure(lm=azure_llm)

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
    historical_data: str = dspy.InputField(desc="Recent gold price historical data and trends")
    market_context: str = dspy.InputField(desc="Current market conditions and news affecting gold")
    timeframe: str = dspy.InputField(desc="Prediction timeframe (24h, 1 week, 1 month)")
    prediction: GoldPrediction = dspy.OutputField(desc="Structured gold price prediction with analysis")

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
    except
