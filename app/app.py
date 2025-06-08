# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from forecasting_utils import run_forecast

# =================== CẤU HÌNH ===================
st.set_page_config(page_title="Dự báo thời tiết 7 ngày", layout="wide")
st.title("📈 Dự báo Nhiệt độ 7 ngày tiếp theo")
st.markdown("Tải lên file dữ liệu thời tiết và chọn mô hình để dự báo:")

# =================== TẢI FILE CSV ===================
uploaded_file = st.file_uploader("📤 Tải lên file CSV", type=["csv"])

# =================== CHỌN MÔ HÌNH ===================
model_options = [
    "GRU", "LSTM", "Transformer", "TCN", "TiDE",
    "XGBoost", "Random Forest", "LightGBM"
]
selected_model = st.selectbox("🔽 Chọn mô hình dự báo", model_options)

# =================== XỬ LÝ DỰ BÁO ===================
if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)

    if st.button("🚀 Chạy dự báo"):
        with st.spinner("Đang xử lý và dự báo..."):
            try:
                forecast_df = run_forecast(selected_model, df_input)
                st.success("✅ Dự báo hoàn tất!")

                st.subheader("📊 Kết quả dự báo:")
                st.dataframe(forecast_df.style.format("{:.2f}"))

                for col in ['min_temp', 'mean_temp', 'max_temp']:
                    if col in forecast_df.columns:
                        st.subheader(f"📉 Biểu đồ: {col}")
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(forecast_df.index, forecast_df[col], marker='o', linestyle='--')
                        ax.set_ylabel("°C")
                        ax.set_xlabel("Ngày")
                        ax.grid(True, linestyle="--", alpha=0.6)
                        st.pyplot(fig)
            except Exception as e:
                st.error(f"❌ Lỗi: {e}")
