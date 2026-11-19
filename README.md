# Nhận dạng cảm xúc khuôn mặt

Dự án notebook Jupyter nhận dạng bảy trạng thái cảm xúc khuôn mặt và hiển thị kết quả bằng emoji.

## Nội dung

- `nhan_dang_cam_xuc_khuon_mat.ipynb`: notebook huấn luyện, đánh giá và demo.
- `emotion_model.keras`: mô hình đã huấn luyện.
- `test_images/`: ảnh dùng để kiểm thử.
- `test_labels.csv`: nhãn tham chiếu của tập test.
- `ket_qua_test.csv`: kết quả dự đoán trên tập test.

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate
pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn ipywidgets pillow
```

## Chạy

Mở notebook bằng Jupyter hoặc VS Code rồi chạy lần lượt các cell:

```bash
jupyter notebook nhan_dang_cam_xuc_khuon_mat.ipynb
```
