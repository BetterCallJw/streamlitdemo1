import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import make_regression

st.set_page_config(page_title="Huấn luyện Mô hình Học máy", layout="wide")

st.title("Hệ thống Tinh chỉnh và Đánh giá Mô hình Random Forest")
st.write("Giao diện tương tác cho phép thay đổi siêu tham số, huấn luyện mô hình dự báo và phân tích mức độ đóng góp của từng biến số.")

@st.cache_data
def load_synthetic_economic_data():
    # Tạo tập dữ liệu giả lập với 5 biến số đầu vào ảnh hưởng đến 1 biến mục tiêu
    X, y = make_regression(
        n_samples=1000, 
        n_features=5, 
        n_informative=3, 
        noise=0.1, 
        random_state=42
    )
    
    feature_names = [
        "Vốn đầu tư công", 
        "Lãi suất ngân hàng", 
        "Chỉ số giá tiêu dùng", 
        "Tỷ giá hối đoái", 
        "Giá nguyên vật liệu"
    ]
    
    df_X = pd.DataFrame(X, columns=feature_names)
    df_y = pd.Series(y, name="Chỉ số Tăng trưởng")
    return df_X, df_y

X, y = load_synthetic_economic_data()

st.sidebar.header("Cấu hình Siêu tham số")

# Các thanh trượt để tinh chỉnh cấu trúc thuật toán Random Forest
n_estimators = st.sidebar.slider(
    "Số lượng cây quyết định (n_estimators):", 
    min_value=10, 
    max_value=200, 
    value=50, 
    step=10
)

max_depth = st.sidebar.slider(
    "Độ sâu tối đa của cây (max_depth):", 
    min_value=2, 
    max_value=20, 
    value=5, 
    step=1
)

test_size = st.sidebar.slider(
    "Tỷ lệ tập kiểm thử (test_size):", 
    min_value=0.1, 
    max_value=0.5, 
    value=0.2, 
    step=0.05
)

# Nút kích hoạt quá trình huấn luyện
start_training = st.button("Bắt đầu Huấn luyện Mô hình")

st.divider()

if start_training:
    with st.spinner("Đang phân tách dữ liệu và huấn luyện thuật toán..."):
        # Phân tách tập huấn luyện và tập kiểm thử
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Khởi tạo và huấn luyện mô hình
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        # Dự báo và tính toán sai số
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        st.subheader("1. Đánh giá Hiệu suất Mô hình")
        col1, col2 = st.columns(2)
        col1.metric("Hệ số xác định R2 Score", f"{r2:.4f}")
        col2.metric("Sai số toàn phương trung bình MSE", f"{mse:.2f}")
        
        # Trích xuất và trực quan hóa Feature Importance
        st.subheader("2. Phân tích Độ quan trọng của đặc trưng")
        
        # Chuyển đổi mảng mức độ quan trọng thành DataFrame và thiết lập tên cột chuẩn tiếng Việt
        importance_df = pd.DataFrame({
            "Đặc trưng": X.columns,
            "Độ quan trọng của đặc trưng": model.feature_importances_
        })
        
        # Sắp xếp lại dữ liệu theo thứ tự giảm dần để biểu đồ trực quan hơn
        importance_df = importance_df.sort_values(by="Độ quan trọng của đặc trưng", ascending=False)
        importance_df = importance_df.set_index("Đặc trưng")
        
        st.bar_chart(importance_df)
        
        st.write("Nhận xét: Biểu đồ trên thể hiện mức độ đóng góp của từng biến số đầu vào đối với kết quả dự báo của mô hình. Các biến có thanh biểu đồ cao nhất là những yếu tố quyết định chính đến sự biến động của Chỉ số Tăng trưởng.")
else:
    st.info("Nhấn nút 'Bắt đầu Huấn luyện Mô hình' để chạy luồng dữ liệu.")
