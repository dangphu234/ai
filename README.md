# Nhận dạng cảm xúc khuôn mặt

Dự án notebook Jupyter nhận dạng bảy trạng thái cảm xúc khuôn mặt và hiển thị kết quả bằng emoji.

## Nội dung

- `nhan_dang_cam_xuc_khuon_mat.ipynb`: notebook huấn luyện, đánh giá và demo.
- `emotion_model.keras`: mô hình đã huấn luyện.
- `test_images/`: ảnh dùng cho bộ demo/báo cáo, độc lập với các split của FER2013.
- `test_labels.csv`: nhãn tham chiếu của bộ demo/báo cáo.
- `ket_qua_test.csv`: kết quả dự đoán trên bộ demo/báo cáo, không phải kết quả PrivateTest của FER2013.

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Phiên bản yêu cầu trong `requirements.txt`:

- Python 3.14.6
- TensorFlow 2.22.0-rc1
- OpenCV 4.11.0
- NumPy 2.5.3
- pandas 3.0.5
- scikit-learn 1.9.1

Đặt `fer2013.csv` cạnh notebook trước khi chạy phần huấn luyện. Dataset được lưu trong repository qua Git LFS. Cài Git LFS và clone bằng:

```bash
git lfs install
git clone https://github.com/dangphu234/ai.git
```

Lưu ý: nếu tải repository bằng nút **Download ZIP**, GitHub có thể chỉ cung cấp file trỏ LFS thay vì đầy đủ dữ liệu `fer2013.csv`; trong trường hợp đó, hãy clone bằng Git hoặc chạy `git lfs pull` trong thư mục repository để tải file đầy đủ.

## Chạy

Mở notebook bằng Jupyter hoặc VS Code rồi chạy lần lượt các cell:

```bash
jupyter notebook nhan_dang_cam_xuc_khuon_mat.ipynb
```
