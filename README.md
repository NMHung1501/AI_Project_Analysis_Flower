# 🌸 ANN Iris Classifier - Phân Loại Hoa Iris Bằng Mạng Nơ-ron Nhân Tạo

Đây là ứng dụng web sử dụng Flask để phân loại hoa Iris dựa trên mạng nơ-ron nhân tạo (ANN) được viết **100% bằng NumPy thuần** mà không phụ thuộc vào các thư viện học máy lớn như TensorFlow hay Scikit-learn (phần ANN).

## 🌟 Tính Năng Nổi Bật

1. **Mô hình ANN thuần NumPy**: Toàn bộ thuật toán lan truyền xuôi (Forward Propagation) và lan truyền ngược (Backpropagation) được tự code tay.
2. **Giao diện Web Trực Quan**: 
   - Hỗ trợ tải lên (upload) dataset (CSV/Excel).
   - Tùy chỉnh tham số huấn luyện (Epochs, Batch Size, Learning Rate).
   - Biểu đồ theo dõi Accuracy và Loss trực tiếp theo thời gian thực (vẽ bằng HTML5 Canvas).
3. **Dự Đoán Nhanh Chóng**:
   - Nhập số liệu kích thước hoa (Chiều dài/rộng của đài hoa và cánh hoa) và dự đoán ngay.
   - **Tự động hiển thị ảnh minh họa**: Khi có kết quả dự đoán, hệ thống sẽ tự động tìm và hiển thị ảnh tương ứng với loài hoa đó (Setosa, Versicolor, hoặc Virginica) để bạn dễ hình dung. (Bản cập nhật V2)

## 📂 Cấu Trúc Dự Án

- `app.py` / `server.py`: Chứa mã nguồn Flask server cung cấp các API để huấn luyện (train) và dự đoán (predict).
- `numpy_ann.py`: File chứa toàn bộ thuật toán về Mạng nơ-ron nhân tạo (ANN), chuẩn hóa dữ liệu (CustomStandardScaler), và các hàm chia dữ liệu được code tay bằng NumPy.
- `iris_ann_classifier.py`: Script CLI để chạy và test mô hình học máy mà không cần giao diện web.
- `static/`: Thư mục chứa giao diện tĩnh.
  - `css/style.css`: Các file CSS tạo kiểu cho web (Glassmorphism, animations).
  - `js/app.js`: Xử lý tương tác của người dùng, gọi API và vẽ biểu đồ.
- `templates/index.html`: File HTML chính của ứng dụng web.

## 🚀 Hướng Dẫn Sử Dụng

### 1. Cài đặt môi trường
Bạn cần cài đặt các thư viện cần thiết. Nếu có sẵn `requirements.txt`, hãy chạy:
```bash
pip install -r requirements.txt
```
*(Các thư viện chính bao gồm: `flask`, `numpy`, `werkzeug`)*

### 2. Khởi chạy ứng dụng
Chạy một trong các lệnh sau tại thư mục gốc của dự án:
```bash
python app.py
# Hoặc: python server.py
```
Sau đó, mở trình duyệt web và truy cập vào địa chỉ: **http://127.0.0.1:5000**

### 3. Quy trình thao tác trên Web
1. **Bước 1 - Chọn Dataset**: Có thể chọn file CSV/Excel của bạn (cột cuối cùng là nhãn) hoặc click nút **Dùng dataset Iris mặc định** để tải dữ liệu có sẵn.
2. **Bước 2 - Huấn Luyện**: Tuỳ chỉnh các tham số Epochs, Batch Size, Learning Rate và nhấn **Bắt đầu huấn luyện**. Đợi mô hình chạy và xem biểu đồ đánh giá.
3. **Bước 3 - Dự Đoán**: Ở khu vực dự đoán, nhập kích thước cánh hoa/đài hoa (Sepal Length/Width, Petal Length/Width) rồi nhấn **Dự đoán**. Kết quả cùng với **ảnh hoa ví dụ** sẽ hiện ra ngay bên dưới!

## 🔄 Phiên Bản Cập Nhật
- **Version 2**: Đã loại bỏ phần Upload ảnh thừa thãi ở bước dự đoán. Tự động hiển thị ảnh mẫu của loài hoa ngay khi hệ thống dự đoán xong từ số liệu đo đạc, giúp cải thiện trải nghiệm người dùng tối đa.
