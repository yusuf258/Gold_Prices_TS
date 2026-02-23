import os
os.environ["TF_USE_LEGACY_KERAS"] = "0"

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import time

TF_AVAILABLE = False
try:
    import tensorflow as tf
    import keras
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    pass

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="GoldPrice AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. DATA & MODEL LAYER ---
@st.cache_resource
def load_ai_assets():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'models', 'gold_lstm_model.h5')
    scaler_path = os.path.join(BASE_DIR, 'models', 'gold_scaler.pkl')

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error(f"Model veya Scaler dosyası bulunamadı! Yol: {model_path}")
        return None, None

    try:
        model = load_model(model_path, compile=False)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"Model yukleme hatasi: {e}")
        return None, None


def load_csv_data(uploaded_file):
    """Load gold price CSV data."""
    try:
        df = pd.read_csv(uploaded_file)
        # Try to find a date column
        date_cols = [c for c in df.columns if 'date' in c.lower()]
        if date_cols:
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
            df.set_index(date_cols[0], inplace=True)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Veri okuma hatasi: {e}")
        return None


def generate_financial_data(window_size=30, volatility=0.5):
    """Generates synthetic financial data - CLEARLY MARKED."""
    start_price = 1500 + np.random.uniform(-200, 200)
    prices = [start_price]
    for _ in range(window_size - 1):
        change = np.random.normal(0, 10 * volatility)
        prices.append(prices[-1] + change)
    return np.array(prices).reshape(-1, 1)


# --- 3. BUSINESS LOGIC ---
def make_prediction(model, scaler, input_data, window_size=30):
    input_scaled = scaler.transform(input_data)
    model_input = input_scaled.reshape(1, window_size, 1)
    pred_scaled = model.predict(model_input, verbose=0)
    prediction = scaler.inverse_transform(pred_scaled)[0][0]
    return prediction


# --- 4. UI LAYER ---
if not TF_AVAILABLE:
    st.error("TensorFlow yuklu degil. Lutfen 'pip install tensorflow' komutuyla yukleyin.")
    st.stop()

model, scaler = load_ai_assets()

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3062/3062634.png", width=100)
    st.title("GoldPrice AI")
    st.markdown("---")

    st.subheader("📂 Veri Kaynagi")
    data_source = st.radio(
        "Veri kaynagini secin:",
        ["CSV Dosyasi Yukle", "Simulasyon (Demo)"],
        index=1
    )

    uploaded_file = None
    volatility = 0.5
    if data_source == "CSV Dosyasi Yukle":
        uploaded_file = st.file_uploader(
            "Altin fiyat verisi yukleyin (CSV)",
            type=["csv"],
            help="USD fiyat kolonu iceren CSV dosyasi bekleniyor"
        )
        price_col = st.text_input("Fiyat kolonu", value="USD (PM)")
    else:
        volatility = st.slider("Market Volatility", 0.1, 1.0, 0.5)
        st.warning("⚠️ Simulasyon modu: Yapay veri kullanilmaktadir.")

    st.markdown("---")
    if model:
        st.success("🟢 Model Online")
        st.caption("Architecture: LSTM\nInput Window: 30 Days")
    else:
        st.error("🔴 Model Offline")
        st.stop()

# Main Page
st.title("💰 Gold Price Prediction System")
st.markdown("AI-powered forecasting tool using **LSTM** to predict the next day's Gold (USD/oz) price based on 30-day trends.")

if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- Data Loading ---
real_data = None
window_data = None
if data_source == "CSV Dosyasi Yukle" and uploaded_file:
    real_data = load_csv_data(uploaded_file)
    if real_data is not None:
        st.success(f"Veri yuklendi: {real_data.shape[0]} satir")

        if price_col not in real_data.columns:
            # Try to find a USD column
            usd_cols = [c for c in real_data.columns if 'usd' in c.lower() or 'price' in c.lower() or 'close' in c.lower()]
            price_col = usd_cols[0] if usd_cols else real_data.select_dtypes(include=[np.number]).columns[0]
            st.info(f"'{price_col}' kolonu kullaniliyor.")

        values = real_data[price_col].dropna().values
        total_days = len(values)
        st.write(f"Toplam {total_days} gunluk veri mevcut.")

        start_idx = st.slider("Baslangic gunu secin (son 30 gun kullanilacak):",
                              30, total_days, total_days)
        window_data = values[start_idx-30:start_idx].reshape(-1, 1)

# --- Action ---
col_action, col_blank = st.columns([1, 4])
with col_action:
    predict_btn = st.button("🔄 Analyze & Predict", type="primary")

if predict_btn:
    with st.spinner("Analyzing market trends..."):
        time.sleep(0.3)

        if data_source == "CSV Dosyasi Yukle" and window_data is not None:
            input_data = window_data
            is_simulation = False
        else:
            input_data = generate_financial_data(volatility=volatility)
            is_simulation = True

        prediction = make_prediction(model, scaler, input_data)
        last_val = input_data[-1][0]

        st.session_state['current_input'] = input_data
        st.session_state['current_pred'] = prediction
        st.session_state['last_val'] = last_val
        st.session_state['is_simulation'] = is_simulation

# --- Results ---
if 'current_pred' in st.session_state:
    pred = st.session_state['current_pred']
    last = st.session_state['last_val']
    input_seq = st.session_state['current_input']
    is_sim = st.session_state.get('is_simulation', True)

    if is_sim:
        st.warning("⚠️ Bu tahmin SIMULASYON verisi ile yapilmistir. Gercek veri icin CSV yukleyin.")

    st.markdown("---")

    # KPI Cards
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        delta = pred - last
        st.metric(label="Predicted Price (Next Day)", value=f"${pred:.2f}",
                  delta=f"${delta:.2f}", delta_color="normal")
    with kpi2:
        st.metric(label="Last Closing Price", value=f"${last:.2f}")
    with kpi3:
        trend = "Bullish 📈" if pred > last else "Bearish 📉"
        st.metric(label="Market Sentiment", value=trend)

    # Chart
    st.subheader("📈 Trend Analysis")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, 31)), y=input_seq.flatten(),
        mode='lines+markers', name='Past 30 Days',
        line=dict(color='#FFD700', width=3),
        fill='tozeroy', fillcolor='rgba(255, 215, 0, 0.1)'
    ))
    fig.add_trace(go.Scatter(
        x=[30, 31], y=[input_seq[-1][0], pred],
        mode='lines', line=dict(color='white', width=2, dash='dash'),
        name='Projection'
    ))
    fig.add_trace(go.Scatter(
        x=[31], y=[pred],
        mode='markers+text',
        marker=dict(color='#00FF00' if pred > last else '#FF0000', size=15),
        text=[f"${pred:.1f}"], textposition="top right",
        name='AI Forecast'
    ))
    fig.update_layout(
        xaxis_title="Days", yaxis_title="Price (USD)",
        template="plotly_dark", height=500, hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔍 View Raw Market Data"):
        df_view = pd.DataFrame(input_seq, columns=["Price (USD)"])
        df_view.index.name = "Day"
        st.dataframe(df_view.T)
else:
    st.info("👈 Click the button to analyze and predict gold prices.")
