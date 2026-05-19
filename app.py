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
# 1. KONFIGURASI HALAMAN DAN ANTARMUKA (LAPISAN PRESENTASI / STREAMLIT)
#    Mengacu pada: Bab III Sub-bab 3.4.4 Perancangan Fisik
# ====================================================================
st.set_page_config(layout="wide", page_title="SPK Trading Kripto - XGBoost")

# ====================================================================
# PINDAHAN: Panel Kontrol (Sidebar) HARUS DI ATAS
# Dibungkus dengan st.form agar form_submit_button berfungsi
# ====================================================================
st.sidebar.header("Panel Kontrol (Input Parameter)")

with st.sidebar.form(key="form_parameter"):
    ticker     = st.text_input("Simbol Aset Kripto", value="BTC-USD")
    start_date = st.date_input("Tanggal Awal Analisis", value=pd.to_datetime("2024-01-01"))
    end_date   = st.date_input("Tanggal Akhir Analisis", value=pd.to_datetime("today"))

    st.markdown("---")
    st.subheader("Hyperparameter Model XGBoost")
    n_estimators  = st.slider("Jumlah Pohon (n_estimators)", 50, 500, 100, step=50)
    max_depth     = st.slider("Kedalaman Pohon (max_depth)", 2, 10, 4)
    learning_rate = st.slider("Learning Rate", 0.01, 0.30, 0.10, step=0.01)

    # Tombol submit form
    tombol_terapkan = st.form_submit_button("Terapkan")

# Validasi tanggal
if start_date >= end_date:
    st.error("Tanggal awal harus lebih kecil dari tanggal akhir.")
    st.stop()

# --- Custom UI: Google Font + CSS for modern/professional look ---
# MENGGUNAKAN TEMA GELAP ELEGAN (DARK/SLATE MODE)
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
    html, body, [class*="css"]  {font-family: 'Inter', sans-serif;}
    .topbar{display:flex;align-items:center;justify-content:space-between;padding:14px 18px;border-radius:8px;background:linear-gradient(90deg,#0f172a 0%, #1e293b 100%);color:#f8fafc;margin-bottom:18px; border: 1px solid #334155;}
    .logo{display:flex;align-items:center;gap:10px}
    .logo img{height:36px;border-radius:6px}
    .app-title{font-size:20px;font-weight:700;margin:0; color:#f8fafc;}
    .app-sub{color:#94a3b8;margin:0;font-size:13px}
    .card{background:#1e293b;border-radius:10px;padding:12px;box-shadow:0 4px 6px -1px rgba(0,0,0,0.1); border: 1px solid #334155;}
    .metric-card{background:#1e293b;border-radius:8px;padding:12px 14px; border: 1px solid #334155;}
    .kpi-row{display:flex;gap:12px}
    .muted{color:#94a3b8}
    </style>
    """,
    unsafe_allow_html=True
)

# Custom header 
st.markdown(
    """
    <div class='topbar'>
        <div class='logo'>
            <div style='width:44px;height:44px;border-radius:8px;background:linear-gradient(135deg,#06b6d4,#ff7a59);display:flex;align-items:center;justify-content:center;font-weight:700;color:white'>SP</div>
            <div>
                <div class='app-title'>Sistem Pendukung Keputusan — Trading Kripto</div>
                <div class='app-sub'>XGBoost • Indikator Teknikal • Visualisasi Interaktif</div>
            </div>
        </div>
        <div style='display:flex;align-items:center;gap:12px'>
            <div class='muted'>Rentang: {start_date} → {end_date}</div>
        </div>
    </div>
    """.format(start_date=start_date, end_date=end_date),
    unsafe_allow_html=True,
)

st.caption("Implementasi Algoritma eXtreme Gradient Boosting (XGBoost) - 5230411291")

# ====================================================================
# 2. LAPISAN INPUT / DATA LAYER (API YFINANCE PULLING)
#    Mengacu pada: Bab III Sub-bab 3.2 Data Penelitian & 3.3 Arsitektur Model
# ====================================================================
@st.cache_data
def ambil_data_historis(symbol, start, end):
    """
    Mengunduh data historis OHLCV melalui API Yahoo Finance.
    Mengacu pada Bab III Sub-bab 3.2.2 Cara Mendapatkan Data.
    """
    df = yf.download(symbol, start=start, end=end, interval="1d")
    return df

with st.spinner("Mengunduh data historis dari Yahoo Finance..."):
    data_mentah = ambil_data_historis(ticker, start_date, end_date)

if data_mentah.empty:
    st.error("Data pasar tidak ditemukan! Silakan periksa kembali simbol aset atau rentang waktu input.")
else:
    # Antisipasi struktur Multi-Index pada yfinance versi terbaru
    if isinstance(data_mentah.columns, pd.MultiIndex):
        data_mentah.columns = data_mentah.columns.droplevel(1)

    df = data_mentah.copy()

    # ====================================================================
    # 3. LAPISAN PEMROSESAN / PROCESSING LAYER
    #    Mengacu pada: Bab III Sub-bab 3.5 Prapemrosesan Data dan Ekstraksi Fitur
    # ====================================================================

    # --- Fitur Utama 1: EMA 20 ---
    df['EMA_20'] = ta.trend.ema_indicator(close=df['Close'], window=20, fillna=False)

    # --- Fitur Utama 2: Stochastic RSI ---
    stoch_rsi_indicator = ta.momentum.StochRSIIndicator(
        close=df['Close'], window=14, smooth1=3, smooth2=3, fillna=False
    )
    df['StochRSI_K'] = stoch_rsi_indicator.stochrsi_k()
    df['StochRSI_D'] = stoch_rsi_indicator.stochrsi_d()

    # --- Fitur Tambahan (Pengembangan dari ruang lingkup dasar) ---
    df['RSI'] = ta.momentum.rsi(close=df['Close'], window=14, fillna=False)
    
    macd_indicator = ta.trend.MACD(close=df['Close'], fillna=False)
    df['MACD']        = macd_indicator.macd()
    df['MACD_Signal'] = macd_indicator.macd_signal()

    bb_indicator  = ta.volatility.BollingerBands(close=df['Close'], window=20, fillna=False)
    df['BB_Width'] = bb_indicator.bollinger_wband()

    df['ATR'] = ta.volatility.average_true_range(
        high=df['High'], low=df['Low'], close=df['Close'], window=14, fillna=False
    )

    # --- Pelabelan Target Klasifikasi Biner ---
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

    # --- Pembersihan Data ---
    df.dropna(inplace=True)

    # ====================================================================
    # 4. LAPISAN PEMODELAN / MODELING LAYER
    #    Mengacu pada: Bab III Sub-bab 3.6 Skenario Pemodelan dan Evaluasi
    # ====================================================================

    fitur_prediktor = [
        'EMA_20', 'StochRSI_K', 'StochRSI_D',   # Fitur utama (sesuai laporan)
        'RSI', 'MACD', 'MACD_Signal',             # Fitur tambahan momentum
        'BB_Width', 'ATR',                        # Fitur tambahan volatilitas
        'Close', 'Volume'                         # Data OHLCV mentah
    ]
    X = df[fitur_prediktor]
    y = df['Target']

    # Pembagian Data: 80% Latih & 20% Uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Hitung rasio kelas untuk menangani class imbalance
    jumlah_negatif = (y_train == 0).sum()
    jumlah_positif = (y_train == 1).sum()
    rasio_kelas    = float(jumlah_negatif / jumlah_positif) if jumlah_positif > 0 else 1.0

    @st.cache_resource
    def latih_model(n_est, depth, lr, spw, data_hash):
        model = XGBClassifier(
            n_estimators=n_est,
            max_depth=depth,
            learning_rate=lr,
            scale_pos_weight=spw, 
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1              
        )
        model.fit(X_train, y_train)
        return model

    data_hash    = hash((ticker, str(start_date), str(end_date), len(X_train)))
    with st.spinner("Melatih model XGBoost (ini bisa memakan beberapa saat)..."):
        model_xgboost = latih_model(n_estimators, max_depth, learning_rate, round(rasio_kelas, 4), data_hash)

    # Evaluasi Model menggunakan Confusion Matrix
    y_prediksi   = model_xgboost.predict(X_test)
    skor_akurasi = accuracy_score(y_test, y_prediksi)
    skor_presisi = precision_score(y_test, y_prediksi, zero_division=0)
    skor_recall  = recall_score(y_test, y_prediksi, zero_division=0)
    skor_f1      = f1_score(y_test, y_prediksi, zero_division=0)

    # Inferensi: prediksi hari terakhir beserta probabilitasnya
    data_terakhir   = df[fitur_prediktor].iloc[-1:]
    prediksi_aktual = model_xgboost.predict(data_terakhir)[0]
    probabilitas    = model_xgboost.predict_proba(data_terakhir)[0]
    prob_naik       = probabilitas[1] * 100
    prob_turun      = probabilitas[0] * 100

    # ====================================================================
    # 5. LAPISAN OUTPUT / PRESENTATION LAYER
    #    Mengacu pada: Bab III Sub-bab 3.4.3 & Tabel 3.2 Use Case
    # ====================================================================
    st.markdown("---")

    # KPI row (cards) - Teks disesuaikan untuk mode gelap
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(
        f"<div class='metric-card'><div style='color:#94a3b8;font-size:12px'>Accuracy</div><div style='font-size:20px;font-weight:700;color:#f8fafc'>{skor_akurasi * 100:.2f}%</div></div>",
        unsafe_allow_html=True,
    )
    k2.markdown(
        f"<div class='metric-card'><div style='color:#94a3b8;font-size:12px'>Precision</div><div style='font-size:20px;font-weight:700;color:#f8fafc'>{skor_presisi * 100:.2f}%</div></div>",
        unsafe_allow_html=True,
    )
    k3.markdown(
        f"<div class='metric-card'><div style='color:#94a3b8;font-size:12px'>Recall</div><div style='font-size:20px;font-weight:700;color:#f8fafc'>{skor_recall * 100:.2f}%</div></div>",
        unsafe_allow_html=True,
    )
    k4.markdown(
        f"<div class='metric-card'><div style='color:#94a3b8;font-size:12px'>F1-Score</div><div style='font-size:20px;font-weight:700;color:#f8fafc'>{skor_f1 * 100:.2f}%</div></div>",
        unsafe_allow_html=True,
    )

    # Recommendation + details - Teks disesuaikan untuk mode gelap
    left_col, right_col = st.columns([1, 1])
    with left_col:
        st.subheader("Rekomendasi Sinyal Keputusan (Real-Time)")
        if prediksi_aktual == 1:
            st.markdown(
                f"<div class='card' style='border-left:6px solid #10b981;padding:16px'>\n                <strong style=\"font-size:18px;color:#34d399\">🟢 BUY (BELI)</strong>\n                <div style=\"color:#cbd5e1;margin-top:6px\">Probabilitas Naik: <strong style='color:#f8fafc'>{prob_naik:.1f}%</strong> — Turun: {prob_turun:.1f}%</div>\n                </div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='card' style='border-left:6px solid #ef4444;padding:16px'>\n                <strong style=\"font-size:18px;color:#f87171\">🔴 SELL (JUAL)</strong>\n                <div style=\"color:#cbd5e1;margin-top:6px\">Probabilitas Turun: <strong style='color:#f8fafc'>{prob_turun:.1f}%</strong> — Naik: {prob_naik:.1f}%</div>\n                </div>",
                unsafe_allow_html=True,
            )
            
    with right_col:
        st.subheader("Metrik Evaluasi — Detail")
        st.write("Skor dievaluasi menggunakan Confusion Matrix pada 20% data uji historis.")
        st.write(f"- Jumlah Data Latih: {len(X_train)} | Jumlah Data Uji: {len(X_test)}")
        st.write(f"- Rasio Kelas (neg/pos) pada latih: {rasio_kelas:.2f}")

    # ====================================================================
    # 6. USE CASE 2: VISUALISASI GRAFIK MARKET
    #    Mengacu pada: Bab III Tabel 3.2 Use Case 2 & Sub-bab 3.4.4 Perancangan Fisik
    # ====================================================================
    st.markdown("---")
    st.subheader("Visualisasi Grafik Market dan Momentum Indikator")

    grafik_subplots = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            f'Grafik Candlestick {ticker} & EMA 20',
            'Osilator Stochastic RSI (%K & %D)'
        ),
        row_heights=[0.65, 0.35]
    )

    # Panel Atas: Candlestick harga OHLCV
    grafik_subplots.add_trace(
        go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'], name='Harga Pasar'
        ), row=1, col=1
    )
    # Overlay: Garis EMA 20
    grafik_subplots.add_trace(
        go.Scatter(x=df.index, y=df['EMA_20'],
                   line=dict(color='orange', width=2), name='EMA 20'),
        row=1, col=1
    )

    # Panel Bawah: Stochastic RSI %K dan %D
    grafik_subplots.add_trace(
        go.Scatter(x=df.index, y=df['StochRSI_K'],
                   line=dict(color='cyan', width=1.5), name='Garis %K'),
        row=2, col=1
    )
    grafik_subplots.add_trace(
        go.Scatter(x=df.index, y=df['StochRSI_D'],
                   line=dict(color='red', width=1.5), name='Garis %D'),
        row=2, col=1
    )
    # Garis batas psikologis
    grafik_subplots.add_hline(y=0.8, line_dash="dash", row=2, col=1,
                               line_color="red",   annotation_text="Overbought (0.8)")
    grafik_subplots.add_hline(y=0.2, line_dash="dash", row=2, col=1,
                               line_color="green", annotation_text="Oversold (0.2)")

    # UBAH TEMA PLOTLY MENJADI DARK
    grafik_subplots.update_layout(
        xaxis_rangeslider_visible=False,
        height=750,
        margin=dict(l=25, r=25, t=40, b=25),
        template='plotly_dark',
        font=dict(family='Inter, Arial')
    )
    st.plotly_chart(grafik_subplots, use_container_width=True)

    # ====================================================================
    # 7. FEATURE IMPORTANCE 
    # ====================================================================
    st.markdown("---")
    st.subheader("Kontribusi Fitur Prediktor (Feature Importance)")

    importance_df = pd.DataFrame({
        'Fitur':     fitur_prediktor,
        'Importance': model_xgboost.feature_importances_
    }).sort_values('Importance', ascending=True)

    fig_imp = px.bar(
        importance_df,
        x='Importance', y='Fitur',
        orientation='h',
        title='Seberapa Besar Pengaruh Tiap Indikator terhadap Prediksi Model XGBoost',
        color='Importance',
        color_continuous_scale='Blues'
    )
    # UBAH TEMA PLOTLY MENJADI DARK
    fig_imp.update_layout(
        height=400,
        margin=dict(l=25, r=25, t=40, b=25),
        coloraxis_showscale=False,
        template='plotly_dark',
        font=dict(family='Inter, Arial')
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # ====================================================================
    # 8. TABEL DATASET 
    # ====================================================================
    with st.expander("Lihat Tabel Dataset Hasil Prapemrosesan & Ekstraksi Fitur"):
        display_df = df[['Open', 'High', 'Low', 'Close', 'Volume',
                          'EMA_20', 'StochRSI_K', 'StochRSI_D',
                          'RSI', 'MACD', 'BB_Width', 'ATR', 'Target']].tail(15)
        st.dataframe(display_df)
        csv = display_df.to_csv(index=True).encode('utf-8')
        st.download_button("Unduh CSV (15 bar terakhir)", data=csv, file_name=f"dataset_{ticker}.csv", mime='text/csv')
