import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.set_page_config(page_title="Hệ thống Dự báo Học máy", layout="wide")

st.title("Huấn luyện và Dự báo Mô hình Random Forest")
st.write("Công cụ xây dựng mô hình dự báo trực tiếp từ tệp dữ liệu tùy chỉnh.")

uploaded_file = st.file_uploader("Tải lên tệp dữ liệu huấn luyện (định dạng CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = df.dropna()
    
    st.write("Bản xem trước dữ liệu:")
    st.dataframe(df.head(), use_container_width=True)
    
    st.sidebar.header("Cấu hình Huấn luyện")
    
    target_column = st.sidebar.selectbox("Chọn biến mục tiêu cần dự báo (Y):", df.columns)
    
    feature_columns = [col for col in df.columns if col != target_column]
    numeric_features = df[feature_columns].select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_features:
        st.error("Tệp dữ liệu không có cột dạng số hợp lệ để làm biến đầu vào.")
    else:
        X = df[numeric_features]
        y = df[target_column]
        
        n_estimators = st.sidebar.slider("Số lượng cây quyết định:", 10, 200, 50, 10)
        
        if st.sidebar.button("Thực thi Huấn luyện"):
            with st.spinner("Đang xử lý thuật toán..."):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                
                st.session_state.trained_model = model
                st.session_state.feature_cols = numeric_features
                
                st.success(f"Huấn luyện hoàn tất. Chỉ số độ chính xác R2 Score: {r2:.4f}")
        
        st.divider()
        
        st.subheader("Trích xuất Dự báo")
        
        if "trained_model" in st.session_state:
            st.write("Nhập các tham số đầu vào mới để mô hình tính toán kết quả.")
            
            input_data = {}
            cols = st.columns(3)
            
            for i, col_name in enumerate(st.session_state.feature_cols):
                with cols[i % 3]:
                    default_val = float(df[col_name].mean())
                    input_data[col_name] = st.number_input(col_name, value=default_val)
            
            if st.button("Khởi chạy Dự báo"):
                input_df = pd.DataFrame([input_data])
                prediction = st.session_state.trained_model.predict(input_df)
                
                st.info(f"Kết quả dự báo cho biến {target_column} là: {prediction[0]:.4f}")
        else:
            st.info("Hệ thống yêu cầu huấn luyện mô hình ở thanh cấu hình bên trái trước khi thực hiện dự báo.")
