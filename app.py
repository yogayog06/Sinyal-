import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ====================================================================
# 1. INIT & CONFIG (Clean Layout & Performance)
# ====================================================================
st.set_page_config(layout="wide", page_title="AI Crypto Intelligence | XGBoost", page_icon="🤖", initial_sidebar_state="expanded")

# ====================================================================
# 2. DESIGN SYSTEM & CSS INJECTION (Premium Dark Theme)
# ====================================================================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
    /* Global Variables & Fonts */
    :root {
        --bg-main: #0f172a;
        --bg-sec: #111827;
        --bg-card: #1e293b;
        --border-color: #334155;
        --text-main: #f8fafc;
        --text-muted: #94a3b8;
        --cyan: #06b6d4;
        --purple: #8b5cf6;
        --buy: #10b981;
        --sell: #ef4444;
    }
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: var(--bg-main) !important;
        color: var(--text-main) !important;
    }

    /* Hide Streamlit Clutter */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background-color: transparent !important;} 
    /* Baris header {visibility: hidden;} dihapus agar panah sidebar tetap muncul */

    /* Premium Top Navbar */
    .ai-navbar {
        display: flex; justify-content: space-between; align-items: center;
        background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(12px);
        padding: 16px 24px; border-radius: 12px; border: 1px solid var(--border-color);
        margin-bottom: 24px; margin-top: -40px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    .nav-brand { font-family: 'Space Grotesk', sans-serif; font-size: 24px; font-weight: 700; background: linear-gradient(90deg, var(--cyan), var(--purple)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; display: flex; align-items: center; gap: 10px; }
    .live-badge { display: flex; align-items: center; gap: 6px; font-size: 13px; font-weight: 600; color: var(--buy); background: rgba(16, 185, 129, 0.1); padding: 4px 10px; border-radius: 20px; border: 1px solid rgba(16, 185, 129, 0.2); }
    .pulse-dot { height: 8px; width: 8px; background-color: var(--buy); border-radius: 50%; box-shadow: 0 0 8px var(--buy); animation: pulse 2s infinite; }
    @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(16,185,129, 0.7); } 70% { box-shadow: 0 0 0 6px rgba(16,185,129, 0); } 100% { box-shadow: 0 0 0 0 rgba(16,185,129, 0); } }

    /* Custom Metric Cards */
    .metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
    .ai-metric-card {
        background: linear-gradient(145deg, var(--bg-card), rgba(30, 41, 59, 0.4));
        border: 1px solid var(--border-color); border-radius: 12px; padding: 18px;
        transition: all 0.3s ease; position: relative; overflow: hidden;
    }
    .ai-metric-card:hover { transform: translateY(-3px); border-color: var(--cyan); box-shadow: 0 8px 25px rgba(6, 182, 212, 0.15); }
    .ai-metric-card::before { content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 2px; background: linear-gradient(90deg, transparent, var(--cyan), transparent); opacity: 0; transition: opacity 0.3s; }
    .ai-metric-card:hover::before { opacity: 1; }
    .metric-title { font-size: 13px; color: var(--text-muted); font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; display: flex; align-items: center; gap: 6px;}
    .metric-value { font-family: 'Space Grotesk', sans-serif; font-size: 28px; font-weight: 700; color: var(--text-main); }
    .metric-sub { font-size: 12px; color: var(--text-muted); margin-top: 4px; }

    /* Modern AI Insight Panel */
    .insight-panel { background: var(--bg-sec); border-left: 4px solid var(--purple); border-radius: 8px; padding: 16px; margin-bottom: 16px; }
    .insight-title { font-size: 14px; font-weight: 600; color: var(--purple); margin-bottom: 8px; display: flex; align-items: center; gap: 6px;}

    /* Sidebar Styling Override */
    [data-testid="stSidebar"] { background-color: var(--bg-sec) !important; border-right: 1px solid var(--border-color); }
    [data-testid="stSidebar"] hr { border-color: var(--border-color); }
    .stButton>button {
        width: 100%; background: linear-gradient(90deg, var(--cyan), var(--purple));
        color: white; border: none; border-radius: 8px; font-weight: 600; font-family: 'Space Grotesk';
        padding: 10px; transition: 0.3s; box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3);
    }
    .stButton>button:hover { box-shadow: 0 6px 20px rgba(6, 182, 212, 0.4); transform: scale(1.02); color: white;}
    
    /* Tabs Customization */
    .stTabs [data-baseweb="tab-list"] { background-color: transparent; gap: 24px; border-bottom: 1px solid var(--border-color); }
    .stTabs [data-baseweb="tab"] { color: var(--text-muted); font-weight: 500; padding: 10px 0; }
    .stTabs [aria-selected="true"] { color: var(--cyan) !important; border-bottom: 2px solid var(--cyan); }
</style>
""", unsafe_allow_html=True)

# ====================================================================
# 3. SIDEBAR (Control Panel Premium)
# ====================================================================
with st.sidebar:
    st.markdown("<h2 style='font-family: Space Grotesk; font-size: 20px; font-weight: 700; margin-bottom: 0;'>⚙️ Engine Config</h2>", unsafe_allow_html=True)
    st.caption("Konfigurasi parameter model & data historis.")
    
    with st.form(key="ai_config_form"):
        ticker = st.text_input("💎 Simbol Aset Kripto", value="BTC-USD")
        
        col_d1, col_d2 = st.columns(2)
        with col_d1: start_date = st.date_input("🗓️ Start", value=pd.to_datetime("2024-01-01"))
        with col_d2: end_date   = st.date_input("🗓️ End", value=pd.to_datetime("today"))
        
        st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:14px; font-weight:600; margin-bottom:10px; color:#8b5cf6;'>🧠 XGBoost Hyperparameters</div>", unsafe_allow_html=True)
        
        n_estimators  = st.slider("Trees (n_estimators)", 50, 500, 150, step=50, help="Jumlah pohon keputusan yang dibangun model.")
        max_depth     = st.slider("Max Depth", 2, 10, 5, help="Kedalaman maksimal setiap pohon. Semakin tinggi = risiko overfitting.")
        learning_rate = st.slider("Learning Rate", 0.01, 0.30, 0.05, step=0.01, help="Langkah pembaruan bobot model setiap iterasi.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("🚀 INITIALIZE AI ENGINE")

if start_date >= end_date:
    st.error("⚠️ Tanggal awal analisis harus lebih kecil dari tanggal akhir.")
    st.stop()

# ====================================================================
# 4. TOP NAVBAR (UI Injection)
# ====================================================================
st.markdown(f"""
<div class="ai-navbar">
    <div class="nav-brand">🤖 XGBoost Quantitative Intelligence</div>
    <div class="live-badge"><div class="pulse-dot"></div> Live Analysis: {ticker}</div>
</div>
""", unsafe_allow_html=True)

# ====================================================================
# 5. DATA FETCHING & FEATURE ENGINEERING (Core ML)
# ====================================================================
@st.cache_data(show_spinner=False)
def fetch_and_process_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end, interval="1d")
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
    
    # Feature Engineering (Indicators)
    df['EMA_20'] = ta.trend.ema_indicator(close=df['Close'], window=20, fillna=False)
    stoch = ta.momentum.StochRSIIndicator(close=df['Close'], window=14, smooth1=3, smooth2=3, fillna=False)
    df['StochRSI_K'] = stoch.stochrsi_k()
    df['StochRSI_D'] = stoch.stochrsi_d()
    df['RSI'] = ta.momentum.rsi(close=df['Close'], window=14, fillna=False)
    
    macd = ta.trend.MACD(close=df['Close'], fillna=False)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    bb = ta.volatility.BollingerBands(close=df['Close'], window=20, fillna=False)
    df['BB_Width'] = bb.bollinger_wband()
    df['ATR'] = ta.volatility.average_true_range(high=df['High'], low=df['Low'], close=df['Close'], window=14, fillna=False)
    
    # Target Labeling
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    df.dropna(inplace=True)
    return df

with st.spinner("Neural networks processing historical market data..."):
    df = fetch_and_process_data(ticker, start_date, end_date)

if df.empty:
    st.error("Aset tidak ditemukan. Periksa kembali ticker di sidebar.")
    st.stop()

# ====================================================================
# 6. MODEL TRAINING & EVALUATION
# ====================================================================
features = ['EMA_20', 'StochRSI_K', 'StochRSI_D', 'RSI', 'MACD', 'MACD_Signal', 'BB_Width', 'ATR', 'Close', 'Volume']
X = df[features]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
class_ratio = float((y_train == 0).sum() / (y_train == 1).sum()) if (y_train == 1).sum() > 0 else 1.0

@st.cache_resource(show_spinner=False)
def train_xgboost(n_est, depth, lr, spw, data_hash):
    model = XGBClassifier(n_estimators=n_est, max_depth=depth, learning_rate=lr, scale_pos_weight=spw, eval_metric='logloss', random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

data_hash = hash((ticker, str(start_date), str(end_date), len(X_train)))
with st.spinner("Optimizing XGBoost Hyperparameters..."):
    model = train_xgboost(n_estimators, max_depth, learning_rate, round(class_ratio, 4), data_hash)

# Inference
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

last_data = df[features].iloc[-1:]
pred_now = model.predict(last_data)[0]
prob = model.predict_proba(last_data)[0]
prob_up = prob[1] * 100
prob_down = prob[0] * 100

# ====================================================================
# 7. DASHBOARD UI: METRIC CARDS & AI INSIGHTS
# ====================================================================
st.markdown(f"""
<div class="metric-grid">
    <div class="ai-metric-card">
        <div class="metric-title">🎯 Model Accuracy</div>
        <div class="metric-value">{acc * 100:.2f}%</div>
        <div class="metric-sub">Prediksi benar / Total Prediksi</div>
    </div>
    <div class="ai-metric-card">
        <div class="metric-title">⚡ Precision Score</div>
        <div class="metric-value">{prec * 100:.2f}%</div>
        <div class="metric-sub">Kualitas sinyal Beli yang relevan</div>
    </div>
    <div class="ai-metric-card">
        <div class="metric-title">🔍 Recall Score</div>
        <div class="metric-value">{rec * 100:.2f}%</div>
        <div class="metric-sub">Kemampuan deteksi momentum Beli</div>
    </div>
    <div class="ai-metric-card">
        <div class="metric-title">⚖️ F1-Score (Balanced)</div>
        <div class="metric-value">{f1 * 100:.2f}%</div>
        <div class="metric-sub">Keseimbangan Precision & Recall</div>
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1.2, 1])

with col1:
    bg_color = "rgba(16, 185, 129, 0.1)" if pred_now == 1 else "rgba(239, 68, 68, 0.1)"
    border_color = "#10b981" if pred_now == 1 else "#ef4444"
    signal_text = "BUY (AKUMULASI)" if pred_now == 1 else "SELL (DISTRIBUSI)"
    signal_color = "#34d399" if pred_now == 1 else "#f87171"
    
    st.markdown(f"""
    <div style="background:{bg_color}; border: 1px solid {border_color}; border-radius: 12px; padding: 24px; position: relative; overflow: hidden; height: 100%;">
        <div style="font-size: 14px; font-weight: 600; color: #94a3b8; text-transform: uppercase; margin-bottom: 8px;">Real-Time AI Signal Output</div>
        <div style="font-family: 'Space Grotesk'; font-size: 32px; font-weight: 700; color: {signal_color}; margin-bottom: 12px;">{signal_text}</div>
        <div style="display: flex; gap: 20px;">
            <div><span style="color:#94a3b8; font-size:13px;">Upside Prob:</span> <span style="font-weight:600; color:#f8fafc;">{prob_up:.1f}%</span></div>
            <div><span style="color:#94a3b8; font-size:13px;">Downside Prob:</span> <span style="font-weight:600; color:#f8fafc;">{prob_down:.1f}%</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Generate simple AI Insights based on data
    rsi_val = last_data['RSI'].values[0]
    trend_val = "Bullish" if last_data['Close'].values[0] > last_data['EMA_20'].values[0] else "Bearish"
    momentum = "Overbought (Koreksi Risiko)" if rsi_val > 70 else "Oversold (Potensi Rebound)" if rsi_val < 30 else "Netral"
    
    st.markdown(f"""
    <div class="insight-panel" style="height: 100%;">
        <div class="insight-title">🧠 AI Market Insights</div>
        <div style="font-size: 13px; color: #cbd5e1; line-height: 1.6;">
            <strong>Macro Trend (EMA 20):</strong> Market saat ini berada pada fase <span style="color:var(--cyan);">{trend_val}</span>.<br>
            <strong>Momentum (RSI 14):</strong> Berada di level {rsi_val:.1f} — terindikasi <span style="color:var(--cyan);">{momentum}</span>.<br>
            <strong>Decision Base:</strong> XGBoost mendeteksi pola historis dengan probabilitas dominan <span style="color:var(--purple); font-weight:600;">{max(prob_up, prob_down):.1f}%</span> ke arah sinyal.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ====================================================================
# 8. TABS SECTION: MULTI-PANEL CHARTS & ANALYTICS
# ====================================================================
tab1, tab2, tab3 = st.tabs(["📊 Technical Multi-Chart", "🧩 Feature Intelligence", "📂 Raw Ledger"])

with tab1:
    # Build Bloomberg/TradingView style chart
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        vertical_spacing=0.03,
        subplot_titles=("Price Action & EMA", "Volume Profile", "MACD Oscillator", "Stochastic RSI")
    )

    # R1: Candlestick & EMA
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Market',
                                 increasing_line_color='#10b981', decreasing_line_color='#ef4444'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='#06b6d4', width=2), name='EMA 20'), row=1, col=1)

    # R2: Volume
    colors = ['#10b981' if row['Close'] >= row['Open'] else '#ef4444' for index, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volume', opacity=0.8), row=2, col=1)

    # R3: MACD
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], marker_color=['#10b981' if val >= 0 else '#ef4444' for val in df['MACD_Hist']], name='Histogram'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='#06b6d4', width=1.5), name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], line=dict(color='#8b5cf6', width=1.5), name='Signal'), row=3, col=1)

    # R4: RSI & StochRSI
    fig.add_trace(go.Scatter(x=df.index, y=df['StochRSI_K'], line=dict(color='#06b6d4', width=1.5), name='%K'), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['StochRSI_D'], line=dict(color='#8b5cf6', width=1.5), name='%D'), row=4, col=1)
    fig.add_hline(y=0.8, line_dash="dash", line_color="rgba(239, 68, 68, 0.5)", row=4, col=1)
    fig.add_hline(y=0.2, line_dash="dash", line_color="rgba(16, 185, 129, 0.5)", row=4, col=1)

    fig.update_layout(
        height=900,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#94a3b8'),
        xaxis_rangeslider_visible=False,
        showlegend=False,
        hovermode="x unified"
    )
    
    # Custom Gridlines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#1e293b', zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#1e293b', zeroline=False)
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("<div style='font-size: 16px; font-weight: 600; margin-bottom: 20px; color: #f8fafc;'>Analitik Kepentingan Fitur Prediktor</div>", unsafe_allow_html=True)
    
    imp_df = pd.DataFrame({'Fitur': features, 'Kepentingan': model.feature_importances_}).sort_values('Kepentingan', ascending=True)
    
    fig_imp = px.bar(imp_df, x='Kepentingan', y='Fitur', orientation='h', color='Kepentingan', color_continuous_scale=['#1e293b', '#8b5cf6', '#06b6d4'])
    fig_imp.update_layout(
        height=450, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#94a3b8'),
        coloraxis_showscale=False
    )
    fig_imp.update_xaxes(showgrid=True, gridcolor='#1e293b')
    fig_imp.update_yaxes(showgrid=False)
    
    st.plotly_chart(fig_imp, use_container_width=True)

with tab3:
    st.markdown("<div style='font-size: 14px; color: #94a3b8; margin-bottom: 10px;'>Data historis dan ekstraksi fitur siap komputasi (15 Bar Terakhir).</div>", unsafe_allow_html=True)
    display_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'EMA_20', 'StochRSI_K', 'RSI', 'MACD', 'ATR', 'Target']
    st.dataframe(df[display_cols].tail(15), use_container_width=True)
    
    csv = df[display_cols].to_csv(index=True).encode('utf-8')
    st.download_button("💾 Export Dataset (CSV)", data=csv, file_name=f"XGBoost_Data_{ticker}.csv", mime='text/csv')
