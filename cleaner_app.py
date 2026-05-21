import streamlit as st
import pandas as pd
import numpy as np
import io

# Cấu hình giao diện ứng dụng Streamlit với chủ đề chuyên nghiệp
st.set_page_config(
    page_title="Công cụ Tiền xử lý Dữ liệu Tự động",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Nhúng CSS tùy chỉnh để tối ưu giao diện theo tiêu chuẩn premium
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Thiết kế tiêu đề Gradient */
    .title-gradient {
        background: linear-gradient(135deg, #00B4DB 0%, #0083B0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.1rem;
    }
    
    /* Card trang trí chuyên nghiệp */
    .status-card {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    /* Thiết lập màu sắc trực quan */
    .quality-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        text-align: center;
    }
    .badge-excellent { background-color: #D1FAE5; color: #065F46; }
    .badge-good { background-color: #FEF3C7; color: #92400E; }
    .badge-warning { background-color: #FEE2E2; color: #991B1B; }
</style>
""", unsafe_allow_html=True)

# Khởi tạo st.session_state để lưu trữ trạng thái xử lý dữ liệu
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
if "df_cleaned" not in st.session_state:
    st.session_state.df_cleaned = None
if "history_log" not in st.session_state:
    st.session_state.history_log = []

# Hàm hỗ trợ đặt lại toàn bộ trạng thái khi tải lên tệp mới
def reset_state():
    st.session_state.df_cleaned = None
    st.session_state.history_log = []

# Giao diện chính của Dashboard
st.markdown('<h1 class="title-gradient">Hệ thống Tiền xử lý Dữ liệu Tự động</h1>', unsafe_allow_html=True)
st.write("Ứng dụng hỗ trợ phân tích chất lượng, tối ưu hóa cấu trúc và tự động làm sạch bộ dữ liệu thô.")
st.divider()

# Sidebar: Khu vực tải lên và điều khiển chung
st.sidebar.header("Nguồn Dữ Liệu")
uploaded_file = st.sidebar.file_uploader(
    "Tải lên tập dữ liệu gốc (CSV):", 
    type=["csv"], 
    on_change=reset_state
)

if uploaded_file is not None:
    # Đọc dữ liệu ban đầu
    if st.session_state.df_raw is None or st.sidebar.button("Nạp lại dữ liệu gốc"):
        st.session_state.df_raw = pd.read_csv(uploaded_file)
        st.session_state.df_cleaned = st.session_state.df_raw.copy()
        st.session_state.history_log = ["Khởi tạo: Tải thành công tập dữ liệu gốc."]

    df = st.session_state.df_raw
    cleaned_df = st.session_state.df_cleaned

    # TÍNH TOÁN CÁC CHỈ SỐ CHẤT LƯỢNG DỮ LIỆU
    total_cells = df.size
    total_missing = df.isna().sum().sum()
    missing_ratio = total_missing / total_cells if total_cells > 0 else 0
    duplicate_rows = df.duplicated().sum()
    duplicate_ratio = duplicate_rows / df.shape[0] if df.shape[0] > 0 else 0
    
    # Tính điểm chất lượng tổng quát (Data Quality Score)
    quality_score = max(0, int(100 - (missing_ratio * 150) - (duplicate_ratio * 100)))
    
    # Phân loại trạng thái chất lượng
    if quality_score >= 90:
        quality_class = "badge-excellent"
        quality_text = "Hoàn hảo (Excellent)"
    elif quality_score >= 70:
        quality_class = "badge-good"
        quality_text = "Khá tốt (Good)"
    else:
        quality_class = "badge-warning"
        quality_text = "Cần làm sạch (Warning)"

    # TAB 1: TỔNG QUAN VÀ THỐNG KÊ CHI TIẾT
    tab_overview, tab_clean, tab_outlier = st.tabs([
        "Trạng Thái & Phân Tích Chất Lượng", 
        "Làm Sạch & Tiền Xử Lý", 
        "Phát Hiện Điểm Dị Biệt (Outliers)"
    ])

    with tab_overview:
        st.subheader("1. Tổng quan tình trạng dữ liệu")
        
        # Bố cục hiển thị các chỉ số đo lường (KPI Cards)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Tổng số dòng (Rows)", f"{df.shape[0]:,}")
        col2.metric("Số cột (Columns)", df.shape[1])
        col3.metric("Số ô khuyết thiếu (NaN)", f"{total_missing:,}", f"{missing_ratio*100:.2f}% khuyết")
        col4.metric("Dòng trùng lặp (Duplicates)", f"{duplicate_rows:,}", f"{duplicate_ratio*100:.2f}% trùng")

        # Hiển thị Điểm số Chất lượng
        st.markdown(f"""
        <div style="background-color: rgba(0, 180, 219, 0.05); padding: 15px; border-radius: 8px; border-left: 5px solid #0083B0; margin-bottom: 20px;">
            <span style="font-weight: 600; font-size: 1.1rem; color: #0083B0;">Điểm chất lượng dữ liệu gốc:</span>
            <span style="font-size: 1.3rem; font-weight: bold; margin-left: 10px;">{quality_score}/100</span>
            <span class="quality-badge {quality_class}" style="margin-left: 15px;">{quality_text}</span>
        </div>
        """, unsafe_allow_html=True)

        # Trực quan hóa tỷ lệ ô khuyết thiếu (Missing Rate) của từng cột
        st.subheader("Tỷ lệ dữ liệu hoàn chỉnh theo từng cột")
        missing_by_col = df.isna().mean() * 100
        completeness_by_col = 100 - missing_by_col
        
        # Biểu đồ cột ngang thể hiện độ đầy đủ dữ liệu
        st.bar_chart(completeness_by_col)

        # Hiển thị cấu trúc chi tiết từng cột (Schema Info)
        with st.expander("Xem cấu trúc cột chi tiết (Meta-data)"):
            schema_data = []
            for col in df.columns:
                schema_data.append({
                    "Tên Cột": col,
                    "Kiểu Dữ Liệu": str(df[col].dtype),
                    "Số Ô Trống": df[col].isna().sum(),
                    "Tỷ Lệ Trống": f"{(df[col].isna().mean()*100):.2f}%",
                    "Số Giá Trị Độc Nhất": df[col].nunique()
                })
            st.dataframe(pd.DataFrame(schema_data), width="stretch")

        st.subheader("Xem trước dữ liệu thô (10 dòng đầu tiên)")
        st.dataframe(df.head(10), width="stretch")

    # TAB 2: CÁC CÔNG CỤ LÀM SẠCH VÀ TIỀN XỬ LÝ TƯƠNG TÁC
    with tab_clean:
        st.subheader("2. Hộp công cụ làm sạch thông minh")
        st.write("Tùy biến cấu hình các bước tiền xử lý để làm sạch tập dữ liệu.")

        # CHỨC NĂNG 2.1: Xử lý dòng trùng lặp
        if duplicate_rows > 0:
            st.warning(f"Phát hiện {duplicate_rows} dòng dữ liệu bị trùng lặp hoàn toàn.")
            if st.button("Loại bỏ các dòng trùng lặp ngay lập tức", type="primary"):
                st.session_state.df_cleaned = cleaned_df.drop_duplicates()
                st.session_state.history_log.append("Xử lý trùng lặp: Đã xóa toàn bộ dòng trùng lặp.")
                st.success("Đã loại bỏ các dòng trùng lặp thành công!")
                st.rerun()
        else:
            st.info("Tuyệt vời! Không phát hiện dòng dữ liệu nào bị trùng lặp.")

        st.divider()

        # CHỨC NĂNG 2.2: Loại bỏ hoặc điền khuyết thiếu theo tùy chọn nâng cao
        col_ctrl, col_stats = st.columns([2, 1])
        
        with col_ctrl:
            st.markdown("#### Xử lý Giá trị Khuyết thiếu (Missing Values)")
            clean_method = st.radio(
                "Chọn chiến thuật xử lý:",
                [
                    "Không thay đổi (Giữ nguyên)",
                    "Loại bỏ hoàn toàn các dòng chứa ô khuyết thiếu (dropna)",
                    "Điền giá trị thay thế thông minh (Imputation)"
                ]
            )

            # Khởi tạo đối tượng xử lý
            temp_df = cleaned_df.copy()

            if clean_method == "Loại bỏ hoàn toàn các dòng chứa ô khuyết thiếu (dropna)":
                if st.button("Áp dụng loại bỏ dòng khuyết"):
                    st.session_state.df_cleaned = temp_df.dropna()
                    st.session_state.history_log.append("Xử lý ô trống: Đã xóa toàn bộ dòng chứa ô trống.")
                    st.success("Đã loại bỏ hoàn toàn các hàng có dữ liệu khuyết thiếu!")
                    st.rerun()

            elif clean_method == "Điền giá trị thay thế thông minh (Imputation)":
                st.write("Cấu hình điền tự động:")
                
                # Phân nhóm điền số và chữ nâng cao
                num_strategy = st.selectbox(
                    "Điền cột số bằng:",
                    ["Số 0 (Mặc định)", "Giá trị Trung bình (Mean)", "Giá trị Trung vị (Median)"]
                )
                cat_strategy = st.selectbox(
                    "Điền cột chữ bằng:",
                    ["Chuỗi mặc định ('Không xác định')", "Giá trị xuất hiện nhiều nhất (Mode)"]
                )

                if st.button("Áp dụng điền khuyết thông minh"):
                    num_cols = temp_df.select_dtypes(include=['number']).columns
                    cat_cols = temp_df.select_dtypes(exclude=['number']).columns

                    # Thực thi điền cột số
                    for col in num_cols:
                        if temp_df[col].isna().sum() > 0:
                            if num_strategy == "Số 0 (Mặc định)":
                                fill_val = 0
                            elif num_strategy == "Giá trị Trung bình (Mean)":
                                fill_val = temp_df[col].mean()
                            else:
                                fill_val = temp_df[col].median()
                            temp_df[col] = temp_df[col].fillna(fill_val)

                    # Thực thi điền cột chữ
                    for col in cat_cols:
                        if temp_df[col].isna().sum() > 0:
                            if cat_strategy == "Chuỗi mặc định ('Không xác định')":
                                fill_val = "Không xác định"
                            else:
                                fill_val = temp_df[col].mode().iloc[0] if not temp_df[col].mode().empty else "Không xác định"
                            temp_df[col] = temp_df[col].fillna(fill_val)

                    st.session_state.df_cleaned = temp_df
                    st.session_state.history_log.append(f"Xử lý ô trống: Điền cột số bằng {num_strategy}, cột chữ bằng {cat_strategy}.")
                    st.success("Đã hoàn tất điền giá trị khuyết thiếu tự động!")
                    st.rerun()

        with col_stats:
            st.markdown("#### Thống kê sau xử lý")
            diff_rows = df.shape[0] - cleaned_df.shape[0]
            st.metric("Dòng đã lọc bỏ", f"{diff_rows:,}")
            st.metric("Khuyết thiếu còn lại", cleaned_df.isna().sum().sum())
            st.metric("Kích thước hiện tại", f"{cleaned_df.shape[0]:,} dòng")

        st.divider()

        # CHỨC NĂNG 2.3: Bỏ bớt các cột không cần thiết
        st.markdown("#### Loại bỏ cột không cần thiết")
        columns_to_drop = st.multiselect(
            "Chọn các trường dữ liệu cần loại bỏ khỏi Dataset:", 
            options=cleaned_df.columns
        )
        if columns_to_drop:
            if st.button("Xác nhận xóa cột đã chọn", type="secondary"):
                st.session_state.df_cleaned = cleaned_df.drop(columns=columns_to_drop)
                st.session_state.history_log.append(f"Cấu trúc: Đã xóa cột {columns_to_drop}.")
                st.success(f"Đã loại bỏ {len(columns_to_drop)} cột khỏi tập dữ liệu!")
                st.rerun()

        st.divider()

        # Hiển thị bảng sau khi làm sạch
        st.subheader("Xem trước tập dữ liệu hiện tại sau khi xử lý (10 dòng đầu)")
        st.dataframe(cleaned_df.head(10), width="stretch")

    # TAB 3: PHÁT HIỆN VÀ XỬ LÝ ĐIỂM DỊ BIỆT (OUTLIERS)
    with tab_outlier:
        st.subheader("3. Phát hiện điểm dị biệt dựa trên phương pháp IQR (Interquartile Range)")
        st.write("Điểm dị biệt có thể gây nhiễu cho kết quả học máy. Sử dụng phân tích hộp để xác định điểm dị biệt.")

        # Lọc ra các cột số
        num_cols = cleaned_df.select_dtypes(include=['number']).columns.tolist()

        if len(num_cols) > 0:
            target_col = st.selectbox("Chọn cột số để phân tích phân phối & Outliers:", num_cols)
            
            # Tính toán IQR
            q1 = cleaned_df[target_col].quantile(0.25)
            q3 = cleaned_df[target_col].quantile(0.75)
            iqr = q3 - iqr if 'iqr' in locals() and False else (q3 - q1)  # Đảm bảo tính toán iqr chuẩn xác
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = cleaned_df[(cleaned_df[target_col] < lower_bound) | (cleaned_df[target_col] > upper_bound)]
            outlier_count = outliers.shape[0]
            outlier_ratio = outlier_count / cleaned_df.shape[0] if cleaned_df.shape[0] > 0 else 0

            col_o1, col_o2 = st.columns(2)
            with col_o1:
                st.markdown(f"""
                - **Khoảng tứ phân vị (IQR):** {iqr:.4f}
                - **Ngưỡng dưới:** {lower_bound:.4f}
                - **Ngưỡng trên:** {upper_bound:.4f}
                """)
            with col_o2:
                st.metric("Số lượng điểm dị biệt phát hiện", f"{outlier_count:,}", f"{outlier_ratio*100:.2f}% của dữ liệu", delta_color="inverse")

            if outlier_count > 0:
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("Loại bỏ điểm dị biệt này", key="drop_outliers"):
                        st.session_state.df_cleaned = cleaned_df[(cleaned_df[target_col] >= lower_bound) & (cleaned_df[target_col] <= upper_bound)]
                        st.session_state.history_log.append(f"Xử lý dị biệt: Đã loại bỏ dị biệt ở cột '{target_col}' bằng phương pháp IQR.")
                        st.success(f"Đã xóa thành công {outlier_count} hàng chứa điểm dị biệt trên cột '{target_col}'!")
                        st.rerun()
                with col_btn2:
                    with st.expander("Xem các giá trị dị biệt"):
                        st.dataframe(outliers[[target_col]], width="stretch")
            else:
                st.success(f"Không phát hiện điểm dị biệt nào trên cột '{target_col}' dựa theo công thức IQR.")
        else:
            st.info("Không tìm thấy thuộc tính dạng số nào trong bộ dữ liệu hiện tại để chạy bộ lọc IQR.")

    # KHU VỰC LIÊN TỰC: LỊCH SỬ THAO TÁC & XUẤT FILE TẢI VỀ
    st.divider()
    
    col_hist, col_export = st.columns([1, 1])

    with col_hist:
        st.subheader("Nhật ký lịch sử làm sạch")
        for log in st.session_state.history_log:
            st.markdown(f"- `{log}`")
        
        if len(st.session_state.history_log) > 1:
            if st.button("Reset: Khôi phục dữ liệu ban đầu"):
                reset_state()
                st.success("Đã hoàn khôi dữ liệu gốc thành công.")
                st.rerun()

    with col_export:
        st.subheader("Trích xuất kết quả làm sạch")
        st.write("Nhận tệp dữ liệu hoàn chỉnh sau khi đi qua hệ thống xử lý tự động.")
        
        # Tạo bộ đệm dữ liệu CSV
        csv_buffer = cleaned_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Tải xuống tệp CSV đã tiền xử lý",
            data=csv_buffer,
            file_name="Du_lieu_Tien_xu_ly.csv",
            mime="text/csv",
            type="primary"
        )

else:
    # Giao diện gợi ý khi chưa tải tệp lên
    st.info("Vui lòng tải lên tệp CSV từ thanh điều hướng bên trái để bắt đầu tiền xử lý.")
    
    # Tạo dữ liệu giả lập mẫu giúp người dùng thử nhanh
    st.write("---")
    st.subheader("Bạn chưa có tệp dữ liệu? Thử nhanh với dữ liệu mẫu bên dưới:")
    
    sample_data = pd.DataFrame({
        "Ho_Ten": ["Nguyễn Văn A", "Trần Thị B", "Nguyễn Văn A", "Lê Văn C", "Phạm Thị D", "Hoàng Văn E", "Vũ Thị F"],
        "Tuoi": [25, 30, 25, np.nan, 45, 120, 28], # 120 là outlier, NaN là khuyết
        "Thu_Nhap_Trieu": [15.5, 20.0, 15.5, 35.2, np.nan, 18.0, 22.5],
        "Thanh_Pho": ["Hà Nội", "TP.HCM", "Hà Nội", "Đà Nẵng", "Cần Thơ", "Hải Phòng", np.nan],
        "Vai_Tro": ["Kỹ sư", "Quản lý", "Kỹ sư", "Nhân viên", "Giám đốc", "Học sinh", "Nhân viên"]
    })
    
    st.write("Cấu trúc tập dữ liệu mẫu (Có dòng trùng lặp, khuyết thiếu, dị biệt):")
    st.dataframe(sample_data, width="stretch")
    
    csv_sample = sample_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Tải tập dữ liệu mẫu (.CSV) về máy",
        data=csv_sample,
        file_name="Du_lieu_Mau_Streamlit.csv",
        mime="text/csv"
    )
