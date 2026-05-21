import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.set_page_config(page_title="Hệ thống Dự báo Học máy", layout="wide")

st.title("Huấn luyện và Dự báo Mô hình Random Forest")
st.write("Công cụ xây dựng mô hình dự báo trực tiếp từ tệp dữ liệu tùy chỉnh.")

uploaded_file = st.file_uploader("Tải lên tệp dữ liệu huấn luyện định dạng CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Bẫy lỗi 1: Tệp không có dữ liệu
        if df.empty:
            st.error("Tệp dữ liệu trống. Hệ thống từ chối xử lý.")
            st.stop()
            
        original_length = len(df)
        df = df.dropna()
        
        # Bẫy lỗi 2: Thiếu mẫu quan sát sau khi làm sạch
        if len(df) < 15:
            st.error(f"Dữ liệu gốc có {original_length} dòng. Sau khi loại bỏ ô khuyết thiếu, chỉ còn {len(df)} dòng. Khối lượng này không đạt mức tối thiểu để phân tách tập kiểm thử.")
            st.stop()

        st.write("Bản xem trước dữ liệu:")
        st.dataframe(df.head(), use_container_width=True)
        
        st.sidebar.header("Cấu hình Huấn luyện")
        
        # Lọc sẵn các biến số để ngăn lỗi sai kiểu dữ liệu
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            st.error("Tập dữ liệu không chứa bất kỳ trường dữ liệu số nào để thực thi thuật toán hồi quy.")
            st.stop()
            
        # Bẫy lỗi 3: Ép buộc biến mục tiêu phải là biến số
        target_column = st.sidebar.selectbox("Chọn biến mục tiêu dự báo Y", numeric_cols)
        
        feature_columns = [col for col in numeric_cols if col != target_column]
        
        # Bẫy lỗi 4: Khuyết thiếu biến độc lập
        if not feature_columns:
            st.error("Sau khi thiết lập biến mục tiêu, hệ thống không tìm thấy biến đầu vào X nào ở định dạng số. Quy trình bị hủy.")
            st.stop()

        X = df[feature_columns]
        y = df[target_column]
        
        n_estimators = st.sidebar.slider("Số lượng cây quyết định", 10, 200, 50, 10)
        
        if st.sidebar.button("Thực thi Huấn luyện"):
            with st.spinner("Đang xử lý thuật toán"):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                
                st.session_state.trained_model = model
                st.session_state.feature_cols = feature_columns
                st.session_state.target_col = target_column
                
                st.success(f"Huấn luyện hoàn tất. Chỉ số độ chính xác R2 Score đạt {r2:.4f}")
        
        st.divider()
        st.subheader("Trích xuất Dự báo")
        
        if "trained_model" in st.session_state:
            # Kiểm tra đồng bộ dữ liệu phiên làm việc
            if st.session_state.target_col != target_column:
                st.warning("Biến mục tiêu đã bị thay đổi. Cần thực thi huấn luyện lại.")
            else:
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
                    st.info(f"Kết quả dự báo cho {target_column} là {prediction[0]:.4f}")
        else:
            st.info("Hệ thống yêu cầu hoàn tất huấn luyện ở thanh cấu hình bên trái trước khi cấp quyền truy cập chức năng dự báo.")

    except Exception as e:
        st.error(f"Tiến trình gián đoạn. Tập dữ liệu không thể phân tích cú pháp. Mã lỗi hệ thống: {str(e)}")
