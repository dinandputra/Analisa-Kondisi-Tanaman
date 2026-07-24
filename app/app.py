import streamlit as st
import pandas as pd
import plotly.express as px
import requests

# Konfigurasi dasar halaman
st.set_page_config(page_title="Dashboard IoT AI", page_icon="🌡️", layout="wide")

# ==========================================
# KONFIGURASI API GROQ
# ==========================================
# API Key milikmu sudah dimasukkan dengan benar menggunakan tanda kutip
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

def get_llm_comment(suhu, kelembaban, timestamp, prediksi):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Merangkai prompt sesuai format spesifikasi tugas
    prompt = f"""
    Analisis data sensor berikut:
    Suhu: {suhu}°C
    Kelembaban: {kelembaban}%
    Waktu: {timestamp}
    Prediksi 6 jam: {prediksi}
    
    Berikan komentar singkat (3-4 kalimat) tentang kondisi saat ini dan rekomendasi tindakan.
    """
    
    payload = {
        # Menggunakan Llama 3.1 untuk menghindari error 400 Bad Request
        "model": "llama-3.1-8b-instant", 
        "messages": [
            {"role": "system", "content": "Anda adalah analis lingkungan."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 300
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status() 
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        # Menangkap error spesifik dari Groq jika format masih ditolak
        return f"🚨 Error dari server Groq: {e.response.text}"
    except Exception as e:
        return f"🚨 Gagal terhubung ke API Groq: {e}"

# ==========================================
# MEMUAT DATA
# ==========================================
@st.cache_data
def load_data():
    df_historis = pd.read_csv('sensor_data.csv', parse_dates=['timestamp'], index_col='timestamp')
    df_prediksi = pd.read_csv('hasil_prediksi_lstm.csv', parse_dates=['Unnamed: 0'])
    df_prediksi.rename(columns={'Unnamed: 0': 'timestamp'}, inplace=True)
    df_prediksi.set_index('timestamp', inplace=True)
    return df_historis, df_prediksi

df_historis, df_prediksi = load_data()

# ==========================================
# NAVIGASI SIDEBAR
# ==========================================
st.sidebar.title("Navigasi Dashboard")
halaman = st.sidebar.radio("Pilih Halaman:", ["Monitoring Real-time", "Forecasting 6 Jam"])

# ==========================================
# HALAMAN 1: MONITORING
# ==========================================
if halaman == "Monitoring Real-time":
    st.title("📊 Monitoring Data Sensor IoT")
    st.markdown("Visualisasi data suhu dan kelembaban historis.")

    # Mengambil data terakhir yang masuk
    data_terakhir = df_historis.iloc[-1]
    
    # Kartu Metrik
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Waktu Terakhir Update", value=str(data_terakhir.name.strftime("%Y-%m-%d %H:%M")))
    with col2:
        st.metric(label="Suhu Terkini (°C)", value=f"{data_terakhir['suhu']} °C")
    with col3:
        st.metric(label="Kelembaban Terkini (%)", value=f"{data_terakhir['kelembaban']} %")

    st.divider()

    # Visualisasi Grafik Historis
    st.subheader("Grafik Historis (Suhu & Kelembaban)")
    fig = px.line(df_historis, x=df_historis.index, y=['suhu', 'kelembaban'], 
                  labels={'value': 'Nilai', 'variable': 'Parameter', 'timestamp': 'Waktu'},
                  title="Pergerakan Suhu dan Kelembaban per Jam")
    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------
    # SEGMEN INTEGRASI LLM GROQ
    # ----------------------------------------
    st.divider()
    st.subheader("🤖 Analisis Lingkungan oleh AI")
    st.markdown("Dapatkan *insight* dan rekomendasi instan berdasarkan kondisi suhu dan kelembaban terkini.")
    
    if st.button("Hasilkan Analisis AI", type="primary"):
        with st.spinner("AI sedang menganalisis data lingkungan..."):
            suhu_sekarang = data_terakhir['suhu']
            kel_sekarang = data_terakhir['kelembaban']
            waktu_sekarang = data_terakhir.name.strftime("%Y-%m-%d %H:%M")
            
            # Ekstrak rata-rata prediksi suhu untuk diberikan ke LLM
            rata_prediksi = df_prediksi['Suhu (°C)'].mean()
            tren_prediksi = f"Rata-rata suhu diprediksi {rata_prediksi:.2f}°C"
            
            # Panggil fungsi Groq
            hasil_analisis = get_llm_comment(suhu_sekarang, kel_sekarang, waktu_sekarang, tren_prediksi)
            
            st.success("Analisis Selesai!")
            st.info(hasil_analisis)

# ==========================================
# HALAMAN 2: FORECASTING
# ==========================================
elif halaman == "Forecasting 6 Jam":
    st.title("📈 Forecasting Suhu & Kelembaban")
    st.markdown("Hasil prediksi menggunakan model jaringan saraf tiruan (LSTM) untuk 6 jam ke depan.")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Grafik Prediksi 6 Jam ke Depan")
        fig_pred = px.line(df_prediksi, x=df_prediksi.index, y=['Suhu (°C)', 'Kelembaban (%)'],
                           markers=True, title="Prediksi Cuaca Dini Hari (21 Juli 2026)")
        st.plotly_chart(fig_pred, use_container_width=True)

    with col2:
        st.subheader("Tabel Hasil Prediksi")
        st.dataframe(df_prediksi.style.format("{:.2f}"))
        
        # Metrik Evaluasi statis dari proses training Colab
        st.subheader("Evaluasi Model")
        st.info("**Suhu:**\n* RMSE: 1.93\n* MAE: 1.93\n* MAPE: 8.88%")
        st.warning("**Kelembaban:**\n* RMSE: 12.51\n* MAE: 12.51\n* MAPE: 13.82%")