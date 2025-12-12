import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
import os

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Prediksi Penjualan Produk Nike",
    page_icon="👟",
    layout="wide"
)

# ===============================
# PATH AMAN (STREAMLIT CLOUD)
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "Nike Dataset.csv")
MODEL_LSTM_PATH = os.path.join(BASE_DIR, "model", "model_lstm.h5")
MODEL_RF_PATH = os.path.join(BASE_DIR, "model", "model_rf.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

# ===============================
# LOAD DATA (ANTI ERROR KOLUMN)
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    # rapikan nama kolom
    df.columns = [c.strip() for c in df.columns]

    # cari kolom tanggal otomatis
    date_col = None
    for c in df.columns:
        if "date" in c.lower():
            date_col = c
            break

    if date_col is None:
        st.error(f"Kolom tanggal tidak ditemukan. Kolom tersedia: {df.columns.tolist()}")
        st.stop()

    # mapping kolom
    rename_map = {date_col: "date"}

    for c in df.columns:
        if c.lower() == "state":
            rename_map[c] = "state"

        if "product" in c.lower():
            rename_map[c] = "product_name"

        cl = c.lower()
        if ("unit" in cl or "quantity" in cl) and ("price" not in cl):
            rename_map[c] = "quantity_sold"

    df = df.rename(columns=rename_map)

    # konversi tanggal
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])

    # tahun
    df["year"] = df["date"].dt.year

    # validasi kolom wajib
    required = ["date", "state", "product_name", "quantity_sold"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        st.error(f"Kolom wajib tidak lengkap: {missing}")
        st.write("Kolom tersedia:", df.columns.tolist())
        st.stop()

    return df

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_models():
    lstm_model = tf.keras.models.load_model(
        MODEL_LSTM_PATH,
        compile=False
    )
    rf_model = joblib.load(MODEL_RF_PATH)
    scaler = joblib.load(SCALER_PATH)

    return lstm_model, rf_model, scaler

# ===============================
# HEADER
# ===============================
st.markdown(
    """
    <h1 style='text-align:center;'>👟 Prediksi Penjualan Produk Nike</h1>
    <p style='text-align:center;'>
    Menggunakan <b>Random Forest</b> dan <b>LSTM</b>
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ===============================
# LOAD DATA & MODEL
# ===============================
df = load_data()
lstm_model, rf_model, scaler = load_models()

# ===============================
# SIDEBAR FILTER
# ===============================
st.sidebar.header("🔎 Filter Data")

state = st.sidebar.selectbox(
    "State / Wilayah",
    sorted(df["state"].unique())
)

product = st.sidebar.selectbox(
    "Produk",
    sorted(df["product_name"].unique())
)

year = st.sidebar.selectbox(
    "Tahun",
    sorted(df["year"].unique())
)

# ===============================
# FILTER DATA
# ===============================
filtered_df = df[
    (df["state"] == state) &
    (df["product_name"] == product) &
    (df["year"] == year)
].sort_values("date")

# ===============================
# DATA TERFILTER
# ===============================
st.subheader("📄 Data Penjualan Terfilter")

st.dataframe(
    filtered_df[["date", "state", "product_name", "quantity_sold"]],
    use_container_width=True
)

# ===============================
# GRAFIK TREN PENJUALAN
# ===============================
st.subheader("📈 Grafik Tren Penjualan")

if len(filtered_df) > 0:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        filtered_df["date"],
        filtered_df["quantity_sold"],
        marker="o"
    )
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Jumlah Terjual")
    ax.set_title("Tren Penjualan Produk")

    st.pyplot(fig)
else:
    st.warning("Data kosong untuk filter ini.")

# ===============================
# PREDIKSI
# ===============================
st.subheader("🔮 Prediksi Penjualan")

if len(filtered_df) >= 10:
    time_steps = 10

    series = filtered_df["quantity_sold"].values.reshape(-1, 1)
    series_scaled = scaler.transform(series)

    X_lstm = []
    for i in range(len(series_scaled) - time_steps):
        X_lstm.append(series_scaled[i:i + time_steps])

    X_lstm = np.array(X_lstm)

    lstm_pred_scaled = lstm_model.predict(X_lstm)
    lstm_pred = scaler.inverse_transform(lstm_pred_scaled)

    rf_x = np.arange(len(series)).reshape(-1, 1)
    rf_pred = rf_model.predict(rf_x)

    # ===============================
    # GRAFIK PREDIKSI
    # ===============================
    fig2, ax2 = plt.subplots(figsize=(10, 4))

    ax2.plot(
        filtered_df["date"],
        series,
        label="Data Aktual",
        marker="o"
    )

    ax2.plot(
        filtered_df["date"][time_steps:],
        lstm_pred,
        label="Prediksi LSTM",
        linestyle="--"
    )

    ax2.plot(
        filtered_df["date"],
        rf_pred,
        label="Prediksi Random Forest",
        linestyle=":"
    )

    ax2.set_xlabel("Tanggal")
    ax2.set_ylabel("Jumlah Terjual")
    ax2.set_title("Perbandingan Prediksi Penjualan")
    ax2.legend()

    st.pyplot(fig2)

    # ===============================
    # PENJELASAN
    # ===============================
    st.markdown(
        """
        ### 📝 Penjelasan Hasil Prediksi
        - **Data Aktual** menunjukkan penjualan asli.
        - **LSTM** mempelajari pola tren berdasarkan waktu.
        - **Random Forest** memprediksi berdasarkan pola umum data.

        📌 Model dianggap baik jika garis prediksi mengikuti arah tren data aktual.
        """
    )

else:
    st.warning("Data tidak cukup untuk prediksi (minimal 10 periode).")

# ===============================
# FOOTER
# ===============================
st.markdown(
    "<hr><p style='text-align:center;'>© 2025 | Prediksi Penjualan Produk Nike</p>",
    unsafe_allow_html=True
)
