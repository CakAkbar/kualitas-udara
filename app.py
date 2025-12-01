import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import folium
import json
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

# ==========================================
# 1. KONFIGURASI HALAMAN & GAYA (CSS)
# ==========================================
st.set_page_config(page_title="Dashboard Kualitas Udara Jatim",
                   layout="wide", page_icon="🌤️")

# CSS Custom (Modern Dark Mode with Glassmorphism)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        color: white;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar Modern Style */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.1);
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
        color: #38bdf8;
        font-weight: 700;
        font-size: 1.2rem;
        margin-bottom: 1.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Hero Section */
    .hero-container {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.15) 0%, rgba(99, 102, 241, 0.15) 100%);
        padding: 2.5rem;
        border-radius: 20px;
        border: 1px solid rgba(56, 189, 248, 0.3);
        margin-bottom: 2.5rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .hero-title {
        margin: 0;
        color: white;
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        position: relative;
        z-index: 1;
    }
    
    .hero-subtitle {
        margin: 0.5rem 0 0 0;
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    
    /* Metric Cards with Glassmorphism */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(51, 65, 85, 0.5) 100%);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 20px;
        padding: 2rem 1.5rem;
        backdrop-filter: blur(20px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #38bdf8);
        background-size: 200% 100%;
        animation: shimmer 3s linear infinite;
    }
    
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(56, 189, 248, 0.3);
        border-color: rgba(56, 189, 248, 0.4);
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        margin: 1rem 0 0.5rem 0;
        line-height: 1;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    
    .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    .metric-sublabel {
        color: #64748b;
        font-size: 0.75rem;
        font-weight: 500;
        margin-top: 0.5rem;
    }
    
    /* Info Box Modern */
    .stAlert {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.1) 0%, rgba(99, 102, 241, 0.1) 100%);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 15px;
        backdrop-filter: blur(10px);
        color: #e0f2fe;
    }
    
    /* Section Headers */
    h3 {
        color: #38bdf8;
        font-weight: 700;
        font-size: 1.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(56, 189, 248, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(56, 189, 248, 0.5);
    }
    
    /* Input Styles */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 10px;
        color: white;
        padding: 0.75rem;
        backdrop-filter: blur(10px);
    }
    
    .stSlider > div > div > div {
        background: rgba(56, 189, 248, 0.3);
    }
    
    /* Table Styling */
    [data-testid="stDataFrame"] {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(51, 65, 85, 0.4) 100%);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 15px;
        backdrop-filter: blur(20px);
        overflow: hidden;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(148, 163, 184, 0.3), transparent);
        margin: 3rem 0;
    }
    
    /* Color Indicators */
    .color-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
        box-shadow: 0 0 10px currentColor;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.9rem;
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 1px solid rgba(148, 163, 184, 0.1);
    }
    
    /* Success/Warning Messages */
    .stSuccess {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(22, 163, 74, 0.15) 100%);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 12px;
        color: #4ade80;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.15) 100%);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 12px;
        color: #f87171;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.15) 0%, rgba(245, 158, 11, 0.15) 100%);
        border: 1px solid rgba(251, 191, 36, 0.3);
        border-radius: 12px;
        color: #fbbf24;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. FUNGSI LOAD DATA & TRAINING MODEL
# ==========================================

@st.cache_resource
def load_and_train_models():
    try:
        df = pd.read_csv('airquality_jatim_modified_5 classes.csv')
    except FileNotFoundError:
        st.error("⚠️ File dataset 'airquality_jatim_modified_5 classes.csv' tidak ditemukan!")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Terjadi kesalahan saat membaca file: {e}")
        st.stop()

    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
    except Exception as e:
        st.error(f"⚠️ Gagal memproses kolom waktu: {e}")
        st.stop()

    cluster_features = ['pm25', 'pm10', 'co', 'no2', 'so2', 'aqius']

    if not all(col in df.columns for col in cluster_features):
        st.error("⚠️ Kolom dataset tidak lengkap untuk Clustering.")
        st.stop()

    scaler = MinMaxScaler()
    X_cluster = scaler.fit_transform(df[cluster_features])

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_cluster)

    cluster_means = df.groupby('cluster')['aqius'].mean()
    sehat_cluster_id = cluster_means.idxmin()
    
    df['status_wilayah'] = df['cluster'].apply(
        lambda x: 'Sehat' if x == sehat_cluster_id else 'Tidak Sehat')

    class_features = ['pm25', 'pm10', 'aqius', 'co', 'no2',
                      'so2', 'temperature', 'humidity', 'pressure']

    if not all(col in df.columns for col in class_features + ['air_quality_category']):
        st.error("⚠️ Kolom dataset tidak lengkap untuk Klasifikasi.")
        st.stop()

    X_class = df[class_features]
    y_class = df['air_quality_category']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_class)

    rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_model.fit(X_class, y_encoded)

    return df, kmeans, rf_model, scaler, le, class_features


@st.cache_data
def load_geojson():
    try:
        with open('filtered.json') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        return None


@st.cache_data
def calculate_batch_arima(df):
    forecast_results = {}
    cities = df['city'].unique()

    for city in cities:
        try:
            city_ts = df[df['city'] == city].set_index('timestamp').resample('h')[
                'aqius'].mean().interpolate()

            train_data = city_ts.tail(48)

            if len(train_data) > 10:
                model = ARIMA(train_data, order=(1, 1, 1))
                model_fit = model.fit()

                forecast = model_fit.forecast(steps=24)
                avg_forecast = forecast.mean()
                current_val = city_ts.iloc[-1]

                trend = "Naik 📈" if avg_forecast > current_val else "Turun 📉"

                forecast_results[city] = {
                    'arima_avg_24h': round(avg_forecast, 1),
                    'arima_trend': trend
                }
            else:
                forecast_results[city] = {
                    'arima_avg_24h': 0, 'arima_trend': "Data Kurang"}

        except Exception:
            forecast_results[city] = {
                'arima_avg_24h': 0, 'arima_trend': "Error"}

    return forecast_results


# --- EKSEKUSI LOAD DATA ---
df, kmeans_model, rf_model, scaler, le, feature_cols = load_and_train_models()
geojson_data = load_geojson()
arima_results = calculate_batch_arima(df)

latest_df = df.sort_values('timestamp').groupby('city').tail(1).copy()

latest_df['arima_forecast'] = latest_df['city'].map(
    lambda x: arima_results.get(x, {}).get('arima_avg_24h', 0))
latest_df['arima_trend'] = latest_df['city'].map(
    lambda x: arima_results.get(x, {}).get('arima_trend', '-'))

# ==========================================
# 3. LAYOUT DASHBOARD
# ==========================================

# --- HERO SECTION ---
st.markdown("""
    <div class='hero-container'>
        <h1 class='hero-title'>🌤️ Dashboard Kualitas Udara Jawa Timur</h1>
        <p class='hero-subtitle'>Real-time Monitoring · Machine Learning Clustering · ARIMA Forecasting</p>
    </div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.markdown("## 🔍 Filter & Prediksi")
selected_city = st.sidebar.selectbox("📍 Pilih Wilayah:", df['city'].unique())

st.sidebar.markdown("---")
st.sidebar.markdown("## 🔬 Simulasi Input Data")

if not latest_df.empty:
    default_aqi = int(latest_df[latest_df['city'] == selected_city]['aqius'].values[0])
    default_pm25 = float(latest_df[latest_df['city'] == selected_city]['pm25'].values[0])
else:
    default_aqi = 50
    default_pm25 = 15.0

input_aqi = st.sidebar.number_input("AQI US", value=default_aqi)
input_pm25 = st.sidebar.number_input("PM2.5 (μg/m³)", value=default_pm25)
input_temp = st.sidebar.slider("Suhu (°C)", 20, 40, 30)

if st.sidebar.button("🎯 Prediksi Kategori"):
    input_data = [input_pm25, input_pm25*1.5,
                  input_aqi, 400, 15, 8, input_temp, 60, 1010]
    try:
        pred_idx = rf_model.predict([input_data])[0]
        pred_label = le.inverse_transform([pred_idx])[0]
        st.sidebar.success(f"✅ Prediksi: **{pred_label}**")
    except Exception as e:
        st.sidebar.error(f"❌ Gagal prediksi: {e}")

# ==========================================
# 4. KONTEN UTAMA
# ==========================================

city_data = latest_df[latest_df['city'] == selected_city].iloc[0]

# --- KARTU INFORMASI (METRIC CARDS) ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    aqi_val = city_data['aqius']
    if aqi_val <= 50:
        aqi_color = "#22c55e"
        aqi_icon = "🟢"
    elif aqi_val <= 100:
        aqi_color = "#eab308"
        aqi_icon = "🟡"
    elif aqi_val <= 150:
        aqi_color = "#f97316"
        aqi_icon = "🟠"
    else:
        aqi_color = "#ef4444"
        aqi_icon = "🔴"

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">AQI Saat Ini {aqi_icon}</div>
        <div class="metric-value" style="color: {aqi_color};">
            {city_data['aqius']:.0f}
        </div>
        <div class="metric-sublabel">{selected_city}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    status_text = city_data['status_wilayah'].strip()
    color_cluster = "#4ade80" if status_text == "Sehat" else "#f87171"
    status_icon = "✅" if status_text == "Sehat" else "⚠️"
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Status Clustering {status_icon}</div>
        <div class="metric-value" style="color: {color_cluster};">
            {status_text}
        </div>
        <div class="metric-sublabel">K-Means Analysis</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Prediksi 24 Jam 🔮</div>
        <div class="metric-value" style="color: #60a5fa;">
            {city_data['arima_forecast']}
        </div>
        <div class="metric-sublabel">ARIMA Forecast</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    trend_color = "#10b981" if "Turun" in city_data['arima_trend'] else "#f59e0b"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Tren Kualitas Udara</div>
        <div class="metric-value" style="color: {trend_color}; font-size: 2rem;">
            {city_data['arima_trend']}
        </div>
        <div class="metric-sublabel">Perubahan Prediksi</div>
    </div>
    """, unsafe_allow_html=True)

# --- BAGIAN PETA ---
st.markdown("### 🗺️ Peta Interaktif Sebaran Kualitas Udara")
st.info("💡 **Legenda Warna:** 🟢 **0-50 (Baik)** | 🟡 **51-100 (Sedang)** | 🟠 **101-150 (Tidak Sehat bagi Sensitif)** | 🔴 **151-200 (Tidak Sehat)** | 🟣 **200+ (Berbahaya)**")

col_map, col_table = st.columns([2, 1])

with col_map:
    if geojson_data:
        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean()
        m = folium.Map(location=[center_lat, center_lon],
                       zoom_start=9, tiles='CartoDB dark_matter')

        for feature in geojson_data['features']:
            nama_kota_json = feature['properties']['NAMOBJ']
            match = latest_df[latest_df['city'].str.contains(nama_kota_json, case=False, na=False)]

            if not match.empty:
                row = match.iloc[0]
                current_aqi = int(row['aqius'])
                
                feature['properties']['aqi_display'] = current_aqi
                feature['properties']['status_display'] = str(row['status_wilayah']).strip()
                feature['properties']['forecast_display'] = row['arima_forecast']
                feature['properties']['trend_display'] = row['arima_trend']
                
                if current_aqi <= 50:
                    color_code = '#22c55e'
                elif current_aqi <= 100:
                    color_code = '#eab308'
                elif current_aqi <= 150:
                    color_code = '#f97316'
                elif current_aqi <= 200:
                    color_code = '#ef4444'
                elif current_aqi <= 300:
                    color_code = '#a855f7'
                else:
                    color_code = '#7f1d1d'

                feature['properties']['color_fill'] = color_code
            else:
                feature['properties']['aqi_display'] = 'N/A'
                feature['properties']['status_display'] = '-'
                feature['properties']['forecast_display'] = '-'
                feature['properties']['trend_display'] = '-'
                feature['properties']['color_fill'] = '#6b7280'

        folium.GeoJson(
            geojson_data,
            style_function=lambda x: {
                'fillColor': x['properties'].get('color_fill', '#6b7280'),
                'color': 'white',
                'weight': 1,
                'fillOpacity': 0.7
            },
            highlight_function=lambda x: {
                'weight': 3,
                'color': '#38bdf8',
                'fillOpacity': 0.9
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['NAMOBJ', 'aqi_display'],
                aliases=['Kota:', 'AQI:'],
                localize=True
            ),
            popup=folium.GeoJsonPopup(
                fields=['NAMOBJ', 'status_display', 'aqi_display',
                        'forecast_display', 'trend_display'],
                aliases=[
                    '🏙️ Wilayah',
                    '🏥 Cluster',
                    '💨 AQI',
                    '📈 Forecast',
                    '📊 Tren'
                ],
                labels=True,
                style="min-width: 200px;"
            )
        ).add_to(m)

        st_folium(m, height=500, width=None)
    else:
        st.error("❌ Gagal memuat file GeoJSON 'filtered.json'.")

with col_table:
    st.markdown("### 📋 rata rata 24 - 48 jam")
    
    def color_aqi(val):
        if val <= 50: color = '#22c55e'
        elif val <= 100: color = '#eab308'
        elif val <= 150: color = '#f97316'
        elif val <= 200: color = '#ef4444'
        else: color = '#a855f7'
        return f'color: {color}; font-weight: bold;'

    display_df = latest_df[['city', 'aqius', 'status_wilayah',
                            'arima_forecast']].sort_values('aqius', ascending=False)

    st.dataframe(
        display_df.style.map(color_aqi, subset=['aqius']),
        column_config={
            "city": "🏙️ Kota",
            "aqius": "💨 AQI",
            "status_wilayah": "🏥 Cluster",
            "arima_forecast": "🔮 Forecast"
        },
        hide_index=True,
        height=500,
        use_container_width=True
    )

# --- GRAFIK DETAIL FORECASTING ---
st.markdown("---")
st.markdown(f"### 📈 Analisis Detail Peramalan - {selected_city}")

try:
    city_ts_plot = df[df['city'] == selected_city].set_index(
        'timestamp').resample('h')['aqius'].mean().interpolate()

    if len(city_ts_plot) > 24:
        model_arima_plot = ARIMA(city_ts_plot, order=(1, 1, 1))
        model_fit_plot = model_arima_plot.fit()
        
        forecast_steps = 48
        forecast_values = model_fit_plot.forecast(steps=forecast_steps)

        fig_forecast = go.Figure()

        history_plot = city_ts_plot.tail(72)
        fig_forecast.add_trace(go.Scatter(
            x=history_plot.index, y=history_plot.values,
            mode='lines', name='Data Historis',
            line=dict(color='#60a5fa', width=3),
            fill='tozeroy',
            fillcolor='rgba(96, 165, 250, 0.1)'
        ))

        last_time = city_ts_plot.index[-1]
        forecast_dates = [last_time + timedelta(hours=x) for x in range(1, forecast_steps + 1)]
        
        fig_forecast.add_trace(go.Scatter(
            x=forecast_dates, y=forecast_values,
            mode='lines+markers', name='Prediksi ARIMA',
            line=dict(color='#10b981', width=3, dash='dot'),
            marker=dict(size=6, color='#10b981')
        ))

        fig_forecast.update_layout(
            title={
                'text': f"Prediksi Kualitas Udara 48 Jam ke Depan - {selected_city}",
                'font': {'size': 20, 'color': '#38bdf8', 'family': 'Inter'}
            },
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15, 23, 42, 0.5)',
            font=dict(color='white', family='Inter'),
            hovermode="x unified",
            xaxis_title="Waktu",
            yaxis_title="Indeks AQI (US)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(30, 41, 59, 0.8)',
                bordercolor='rgba(148, 163, 184, 0.3)',
                borderwidth=1
            ),
            xaxis=dict(
                gridcolor='rgba(148, 163, 184, 0.1)',
                showgrid=True
            ),
            yaxis=dict(
                gridcolor='rgba(148, 163, 184, 0.1)',
                showgrid=True
            )
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
    else:
        st.warning("⚠️ Data historis tidak cukup untuk menampilkan grafik prediksi.")
except Exception as e:
    st.error(f"❌ Gagal membuat grafik forecast: {e}")

# --- FOOTER ---
st.markdown("---")
st.markdown("""
    <div class='footer'>
        <p>© 2025 Dashboard Kualitas Udara Jawa Timur | Powered by Machine Learning & ARIMA</p>
        <p style='font-size: 0.8rem; margin-top: 0.5rem;'>Built with Streamlit · Plotly · Scikit-learn · Statsmodels</p>
    </div>
""", unsafe_allow_html=True)