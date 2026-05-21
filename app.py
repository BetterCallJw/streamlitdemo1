import streamlit as st
import pandas as pd
import numpy as np

# Thiết lập cấu hình toàn cục cho trang
st.set_page_config(page_title="Mô phỏng Chỉ số Kinh tế", layout="wide")

st.title("Bảng điều khiển Mô phỏng Kinh tế - Xã hội")
st.write("Hệ thống trực quan hóa các chỉ số vĩ mô dựa trên dữ liệu mô phỏng.")

# Kỹ thuật Caching: Lưu trữ bộ nhớ đệm cho dữ liệu đầu vào
# Cơ chế này ngăn hệ thống tạo lại mảng ngẫu nhiên khi người dùng tương tác với bộ lọc
@st.cache_data
def generate_macroeconomic_data():
    np.random.seed(42)
    years = np.arange(2015, 2027)
    data = pd.DataFrame({
        "Nam": years,
        "Tang_truong_GDP_Phan_tram": np.random.normal(6.5, 1.2, len(years)),
        "Lam_phat_Phan_tram": np.random.normal(3.5, 0.8, len(years)),
        "Von_FDI_Ty_USD": np.random.normal(25.0, 4.5, len(years))
    })
    return data

df = generate_macroeconomic_data()

# Xây dựng khu vực điều khiển qua thanh bên
st.sidebar.header("Tham số Mô phỏng")

selected_metric = st.sidebar.selectbox(
    "Chỉ số phân tích trọng tâm:",
    ["Tang_truong_GDP_Phan_tram", "Lam_phat_Phan_tram", "Von_FDI_Ty_USD"]
)

start_year, end_year = st.sidebar.slider(
    "Giai đoạn phân tích:",
    min_value=int(df["Nam"].min()),
    max_value=int(df["Nam"].max()),
    value=(2018, 2026)
)

# Xử lý logic lọc dữ liệu
filtered_df = df[(df["Nam"] >= start_year) & (df["Nam"] <= end_year)]

# Trực quan hóa dữ liệu qua các khối hiển thị
st.subheader("Chỉ số Đo lường Hiệu suất Tổng quan")

# Phân chia bố cục thành 3 cột đều nhau
col1, col2, col3 = st.columns(3)

col1.metric(
    label="GDP Trung bình", 
    value=f"{filtered_df['Tang_truong_GDP_Phan_tram'].mean():.2f}%",
    delta=f"{filtered_df['Tang_truong_GDP_Phan_tram'].iloc[-1] - filtered_df['Tang_truong_GDP_Phan_tram'].iloc[-2]:.2f}% so với năm trước"
)
col2.metric(
    label="Lạm phát Trung bình", 
    value=f"{filtered_df['Lam_phat_Phan_tram'].mean():.2f}%",
    delta=f"{filtered_df['Lam_phat_Phan_tram'].iloc[-1] - filtered_df['Lam_phat_Phan_tram'].iloc[-2]:.2f}% so với năm trước",
    delta_color="inverse"
)
col3.metric(
    label="FDI Trung bình", 
    value=f"{filtered_df['Von_FDI_Ty_USD'].mean():.2f} Tỷ USD",
    delta=f"{filtered_df['Von_FDI_Ty_USD'].iloc[-1] - filtered_df['Von_FDI_Ty_USD'].iloc[-2]:.2f} Tỷ USD so với năm trước"
)

# Biểu diễn đồ thị chuỗi thời gian
st.subheader(f"Xu hướng biến động: {selected_metric}")
st.line_chart(filtered_df.set_index("Nam")[selected_metric])

# Trình bày dữ liệu thô
st.subheader("Cấu trúc Dữ liệu Chi tiết")
st.dataframe(filtered_df, use_container_width=True)
