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
                   layout="wide", page_icon="☁️")

# CSS Custom (Dark Mode & Gradient)
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom right, #111827, #1e3a8a, #111827);
        color: white;
    }
    .css-1d391kg { background-color: #1f2937; } 
    
    /* Card Style */
    .metric-card {
        background: rgba(31, 41, 55, 0.5);
        border: 1px solid rgba(75, 85, 99, 0.4);
        border-radius: 15px;
        padding: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: bold; }
    .metric-label { color: #9ca3af; font-size: 0.9rem; }
    
    /* Text Highlighting */
    .highlight-good { color: #4ade80; font-weight: bold; }
    .highlight-moderate { color: #facc15; font-weight: bold; }
    .highlight-unhealthy { color: #fb923c; font-weight: bold; }
    .highlight-danger { color: #f87171; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. FUNGSI LOAD DATA & TRAINING MODEL
# ==========================================

@st.cache_resource
def load_and_train_models():
    # --- A. LOAD DATA CSV ---
    try:
        # Pastikan nama file CSV sesuai dengan yang ada di folder Anda
        df = pd.read_csv('airquality_jatim_modified_5 classes.csv')
    except FileNotFoundError:
        st.error("⚠️ File dataset 'airquality_jatim_modified_5 classes.csv' tidak ditemukan!")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Terjadi kesalahan saat membaca file: {e}")
        st.stop()

    # Preprocessing Waktu
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
    except Exception as e:
        st.error(f"⚠️ Gagal memproses kolom waktu: {e}")
        st.stop()

    # --- B. CLUSTERING (K-MEANS) ---
    cluster_features = ['pm25', 'pm10', 'co', 'no2', 'so2', 'aqius']

    if not all(col in df.columns for col in cluster_features):
        st.error("⚠️ Kolom dataset tidak lengkap untuk Clustering.")
        st.stop()

    scaler = MinMaxScaler()
    X_cluster = scaler.fit_transform(df[cluster_features])

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_cluster)

    # --- LABELING CLUSTER DINAMIS (PENTING) ---
    # Kita cek rata-rata AQI di setiap cluster untuk menentukan mana yang "Sehat"
    cluster_means = df.groupby('cluster')['aqius'].mean()
    
    # Cluster dengan rata-rata AQI lebih rendah = SEHAT
    # Cluster dengan rata-rata AQI lebih tinggi = TIDAK SEHAT
    sehat_cluster_id = cluster_means.idxmin()
    
    df['status_wilayah'] = df['cluster'].apply(
        lambda x: 'Sehat' if x == sehat_cluster_id else 'Tidak Sehat')

    # --- C. KLASIFIKASI (RANDOM FOREST) ---
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
    """Load data batas wilayah dari file JSON user."""
    try:
        with open('filtered.json') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        return None


@st.cache_data
def calculate_batch_arima(df):
    """
    Menghitung forecast sederhana untuk SEMUA kota secara otomatis.
    """
    forecast_results = {}
    cities = df['city'].unique()

    for city in cities:
        try:
            # Resample data per jam
            city_ts = df[df['city'] == city].set_index('timestamp').resample('h')[
                'aqius'].mean().interpolate()

            # Gunakan data 48 jam terakhir saja agar proses loading cepat
            train_data = city_ts.tail(48)

            if len(train_data) > 10:
                # Simple ARIMA (1,1,1)
                model = ARIMA(train_data, order=(1, 1, 1))
                model_fit = model.fit()

                # Forecast 24 jam ke depan
                forecast = model_fit.forecast(steps=24)
                avg_forecast = forecast.mean()
                current_val = city_ts.iloc[-1]

                # Tentukan Tren
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

# Ambil data terbaru per kota
latest_df = df.sort_values('timestamp').groupby('city').tail(1).copy()

# Masukkan hasil ARIMA ke dalam dataframe agar mudah diakses peta
latest_df['arima_forecast'] = latest_df['city'].map(
    lambda x: arima_results.get(x, {}).get('arima_avg_24h', 0))
latest_df['arima_trend'] = latest_df['city'].map(
    lambda x: arima_results.get(x, {}).get('arima_trend', '-'))

# ==========================================
# 3. LAYOUT DASHBOARD
# ==========================================

# --- HERO SECTION ---
st.markdown("""
    <div style='background: rgba(245, 158, 11, 0.1); padding: 20px; border-radius: 10px; border: 1px solid rgba(245, 158, 11, 0.3); margin-bottom: 20px;'>
        <h1 style='margin:0; color: white;'>☁️ Dashboard Kualitas Udara Jawa Timur</h1>
        <p style='margin:0; color: #9ca3af;'>Monitoring, Clustering & Forecasting (ARIMA)</p>
    </div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.header("🔍 Filter & Prediksi")
selected_city = st.sidebar.selectbox("Pilih Wilayah:", df['city'].unique())

st.sidebar.markdown("---")
st.sidebar.subheader("📝 Simulasi Input Data")

# Input Simulasi
if not latest_df.empty:
    default_aqi = int(latest_df[latest_df['city'] == selected_city]['aqius'].values[0])
    default_pm25 = float(latest_df[latest_df['city'] == selected_city]['pm25'].values[0])
else:
    default_aqi = 50
    default_pm25 = 15.0

input_aqi = st.sidebar.number_input("AQI US", value=default_aqi)
input_pm25 = st.sidebar.number_input("PM2.5", value=default_pm25)
input_temp = st.sidebar.slider("Suhu (°C)", 20, 40, 30)

if st.sidebar.button("Prediksi Kategori (RF)"):
    # Contoh input sederhana, idealnya semua parameter diisi user
    input_data = [input_pm25, input_pm25*1.5,
                  input_aqi, 400, 15, 8, input_temp, 60, 1010]
    try:
        pred_idx = rf_model.predict([input_data])[0]
        pred_label = le.inverse_transform([pred_idx])[0]
        st.sidebar.success(f"Prediksi Kategori: **{pred_label}**")
    except Exception as e:
        st.sidebar.error(f"Gagal prediksi: {e}")

# ==========================================
# 4. KONTEN UTAMA
# ==========================================

city_data = latest_df[latest_df['city'] == selected_city].iloc[0]

# --- KARTU INFORMASI (METRIC CARDS) ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    # Tentukan class CSS berdasarkan nilai AQI
    aqi_val = city_data['aqius']
    if aqi_val <= 50:
        aqi_color = "#22c55e"
    elif aqi_val <= 100:
        aqi_color = "#eab308"
    elif aqi_val <= 150:
        aqi_color = "#f97316"
    else:
        aqi_color = "#ef4444"

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">AQI Saat Ini</div>
        <div class="metric-value" style="color: {aqi_color};">
            {city_data['aqius']:.0f}
        </div>
        <div class="metric-label">{selected_city}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    status_text = city_data['status_wilayah'].strip()
    # Warna khusus untuk status cluster
    color_cluster = "#4ade80" if status_text == "Sehat" else "#f87171"
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Status Clustering</div>
        <div class="metric-value" style="color: {color_cluster};">
            {status_text}
        </div>
        <div class="metric-label">K-Means</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Forecast (Avg 24h)</div>
        <div class="metric-value text-blue-400">
            {city_data['arima_forecast']}
        </div>
        <div class="metric-label">Prediksi ARIMA</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Tren Prediksi</div>
        <div class="metric-value text-yellow-400">
            {city_data['arima_trend']}
        </div>
        <div class="metric-label">Arah Perubahan</div>
    </div>
    """, unsafe_allow_html=True)

# --- BAGIAN PETA (GEOJSON + WARNA SESUAI AQI) ---
st.markdown("### 🗺️ Peta Sebaran Kualitas Udara")
st.info("💡 **Legenda Warna:** 🟢 **0-50 (Baik)** | 🟡 **51-100 (Sedang)** | 🟠 **101-150 (Tidak Sehat bagi Sensitif)** | 🔴 **151-200 (Tidak Sehat)** | 🟣 **200+ (Bahaya)**")

col_map, col_table = st.columns([2, 1])

with col_map:
    if geojson_data:
        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean()
        m = folium.Map(location=[center_lat, center_lon],
                       zoom_start=9, tiles='CartoDB dark_matter')

        # --- LOGIKA PENYUNTIKAN DATA & WARNA PETA ---
        for feature in geojson_data['features']:
            nama_kota_json = feature['properties']['NAMOBJ']

            # Matching Nama Kota (Case Insensitive & Partial Match)
            # Misal: JSON "Kab. Gresik" akan cocok dengan CSV "Gresik"
            match = latest_df[latest_df['city'].str.contains(nama_kota_json, case=False, na=False)]

            if not match.empty:
                row = match.iloc[0]
                current_aqi = int(row['aqius'])
                
                # Inject Data ke Property GeoJSON
                feature['properties']['aqi_display'] = current_aqi
                feature['properties']['status_display'] = str(row['status_wilayah']).strip()
                feature['properties']['forecast_display'] = row['arima_forecast']
                feature['properties']['trend_display'] = row['arima_trend']
                
                # --- [LOGIKA WARNA MUTLAK BERDASARKAN AQI] ---
                if current_aqi <= 50:
                    color_code = '#22c55e'  # Hijau (Baik)
                elif current_aqi <= 100:
                    color_code = '#eab308'  # Kuning (Sedang)
                elif current_aqi <= 150:
                    color_code = '#f97316'  # Jingga (Tidak Sehat bagi Sensitif)
                elif current_aqi <= 200:
                    color_code = '#ef4444'  # Merah (Tidak Sehat) - AQI 150+ masuk sini
                elif current_aqi <= 300:
                    color_code = '#a855f7'  # Ungu (Sangat Tidak Sehat)
                else:
                    color_code = '#7f1d1d'  # Maroon (Berbahaya)

                feature['properties']['color_fill'] = color_code
            else:
                # Default jika data tidak ditemukan
                feature['properties']['aqi_display'] = 'N/A'
                feature['properties']['status_display'] = '-'
                feature['properties']['forecast_display'] = '-'
                feature['properties']['trend_display'] = '-'
                feature['properties']['color_fill'] = '#6b7280' # Abu-abu

        # Tambahkan Layer GeoJson ke Peta
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
                'color': '#facc15',
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
        st.error("Gagal memuat file GeoJSON 'filtered.json'.")

with col_table:
    st.markdown("### 📋 Data Real-time")
    
    # Fungsi Styling Tabel Pandas
    def color_aqi(val):
        if val <= 50: color = '#22c55e'
        elif val <= 100: color = '#eab308'
        elif val <= 150: color = '#f97316'
        elif val <= 200: color = '#ef4444'
        else: color = '#a855f7'
        return f'color: {color}; font-weight: bold;'

    display_df = latest_df[['city', 'aqius', 'status_wilayah',
                            'arima_forecast']].sort_values('aqius', ascending=False)

    # Tampilkan Tabel dengan Highlight
    st.dataframe(
        display_df.style.map(color_aqi, subset=['aqius']),
        column_config={
            "city": "Kota",
            "aqius": "AQI",
            "status_wilayah": "Cluster",
            "arima_forecast": "Forecast"
        },
        hide_index=True,
        height=500,
        use_container_width=True
    )

# --- GRAFIK DETAIL FORECASTING (ARIMA) ---
st.markdown("---")
st.markdown(f"### 📈 Grafik Detail Peramalan: {selected_city}")

try:
    # Ambil data historis kota terpilih
    city_ts_plot = df[df['city'] == selected_city].set_index(
        'timestamp').resample('h')['aqius'].mean().interpolate()

    if len(city_ts_plot) > 24:
        # Train ulang model khusus untuk plotting grafik
        model_arima_plot = ARIMA(city_ts_plot, order=(1, 1, 1))
        model_fit_plot = model_arima_plot.fit()
        
        # Forecast 48 jam ke depan
        forecast_steps = 48
        forecast_values = model_fit_plot.forecast(steps=forecast_steps)

        # Visualisasi dengan Plotly
        fig_forecast = go.Figure()

        # 1. Data Historis (72 jam terakhir saja agar tidak padat)
        history_plot = city_ts_plot.tail(72)
        fig_forecast.add_trace(go.Scatter(
            x=history_plot.index, y=history_plot.values,
            mode='lines', name='Data Historis',
            line=dict(color='#3b82f6', width=3)
        ))

        # 2. Data Prediksi
        last_time = city_ts_plot.index[-1]
        forecast_dates = [last_time + timedelta(hours=x) for x in range(1, forecast_steps + 1)]
        
        fig_forecast.add_trace(go.Scatter(
            x=forecast_dates, y=forecast_values,
            mode='lines+markers', name='Prediksi (ARIMA)',
            line=dict(color='#10b981', width=2, dash='dot')
        ))

        fig_forecast.update_layout(
            title=f"Prediksi Kualitas Udara - {selected_city}",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'), hovermode="x unified",
            xaxis_title="Waktu", yaxis_title="Indeks AQI",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
    else:
        st.warning("Data historis tidak cukup untuk menampilkan grafik prediksi.")
except Exception as e:
    st.error(f"Gagal membuat grafik forecast: {e}")

# --- FOOTER ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>© 2025 Dashboard Kualitas Udara Jatim</div>",
            unsafe_allow_html=True)