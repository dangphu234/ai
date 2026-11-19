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
pip install -r requirements.txt
```

Phiên bản đã kiểm thử:

- Python 3.14.6
- TensorFlow 2.22.0-rc1
- OpenCV 4.11.0
- NumPy 2.5.3
- pandas 3.0.5
- scikit-learn 1.9.1

Đặt `fer2013.csv` cạnh notebook trước khi chạy phần huấn luyện. File dataset không được lưu trong repository.

## Chạy

Mở notebook bằng Jupyter hoặc VS Code rồi chạy lần lượt các cell:

```bash
jupyter notebook nhan_dang_cam_xuc_khuon_mat.ipynb
```
