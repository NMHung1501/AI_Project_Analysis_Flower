# =============================================================================
# PHÂN LOẠI HOA IRIS BẰNG MẠNG NƠRON NHÂN TẠO (ANN)
# Sử dụng 100% Numpy thuần (Tương thích mọi máy, không cần sklearn/TensorFlow)
# Kiến trúc: Input(4) → Dense(8, ReLU) → Dense(6, ReLU) → Output(3, Softmax)
# =============================================================================

import numpy as np

# Import mạng Nơ-ron tự xây dựng từ file numpy_ann
from numpy_ann import NumpyANN, CustomStandardScaler, train_test_split_custom, get_iris_dataset

# Đặt seed để đảm bảo kết quả có thể tái tạo (reproducibility)
np.random.seed(42)

print("=" * 60)
print("   PHÂN LOẠI HOA IRIS BẰNG MẠNG NƠRON NHÂN TẠO (ANN)")
print("   Backend: 100% Numpy Thuần (Không phụ thuộc TF/Sklearn)")
print("=" * 60)


# =============================================================================
# BƯỚC 1: CHUẨN BỊ DỮ LIỆU (DATA PREPARATION)
# =============================================================================

# --- 1.1: Tải bộ dữ liệu Iris ---
X, y, flower_names, feature_names = get_iris_dataset()

print(f"\n[1] Thông tin bộ dữ liệu Iris:")
print(f"    - Số lượng mẫu    : {X.shape[0]}")
print(f"    - Số lượng đặc trưng: {X.shape[1]}")
print(f"    - Các đặc trưng   : {feature_names}")
print(f"    - Các loài hoa    : {flower_names}")

# --- 1.2: Chuẩn hoá đặc trưng (Feature Scaling) ---
# StandardScaler đưa các đặc trưng về phân phối chuẩn (mean=0, std=1)
scaler = CustomStandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 1.3: Chia dữ liệu thành tập huấn luyện (80%) và tập kiểm tra (20%) ---
X_train, X_test, y_train, y_test = train_test_split_custom(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"\n[3] Chia dữ liệu Train/Test:")
print(f"    - Tập huấn luyện (Train): {X_train.shape[0]} mẫu ({X_train.shape[0]/150*100:.0f}%)")
print(f"    - Tập kiểm tra  (Test) : {X_test.shape[0]} mẫu ({X_test.shape[0]/150*100:.0f}%)")


# =============================================================================
# BƯỚC 2: XÂY DỰNG KIẾN TRÚC MÔ HÌNH (MODEL ARCHITECTURE)
# =============================================================================

# Khởi tạo mô hình NumpyANN với kiến trúc yêu cầu:
# Đầu vào (4) -> Lớp ẩn 1 (8) -> Lớp ẩn 2 (6) -> Đầu ra (3)
model = NumpyANN(layer_sizes=(4, 8, 6, 3), learning_rate=0.01, epochs=50)

print(f"\n[4] Kiến trúc mô hình ANN:")
print(f"    Input Layer  : 4 đặc trưng")
print(f"    Hidden Layer 1: 8 nơron · ReLU")
print(f"    Hidden Layer 2: 6 nơron · ReLU")
print(f"    Output Layer : 3 nơron · Softmax")
print(f"    Optimizer    : Gradient Descent (lr=0.01)")


# =============================================================================
# BƯỚC 3: BIÊN DỊCH VÀ HUẤN LUYỆN MÔ HÌNH (COMPILATION & TRAINING)
# =============================================================================

print(f"\n[5] Bắt đầu huấn luyện ({model.epochs} epochs)...")
print("-" * 60)

# Tiến hành huấn luyện
history = model.fit(X_train, y_train)

print("-" * 60)
print("[5] Huấn luyện hoàn tất!")
print(f"    Loss cuối cùng (training): {history['loss'][-1]:.6f}")


# =============================================================================
# BƯỚC 4: ĐÁNH GIÁ MÔ HÌNH (EVALUATION)
# =============================================================================

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)

# Tính độ chính xác
accuracy = np.mean(y_pred == y_test)
test_loss = -np.mean(np.log(y_pred_prob[np.arange(len(y_test)), y_test] + 1e-8))

print(f"\n[6] Kết quả đánh giá trên tập kiểm tra ({X_test.shape[0]} mẫu):")
print(f"    ✦ Loss                           : {test_loss:.4f}")
print(f"    ✦ Accuracy (Độ chính xác)        : {accuracy * 100:.2f}%")


# =============================================================================
# BƯỚC 5: DỰ ĐOÁN MẪU MỚI (PREDICTION)
# =============================================================================

# Mẫu: [Sepal Length=5.1, Sepal Width=3.5, Petal Length=1.4, Petal Width=0.2]
new_sample_raw = np.array([[5.1, 3.5, 1.4, 0.2]])

# Chuẩn hoá mẫu mới
new_sample_scaled = scaler.transform(new_sample_raw)

print(f"\n[7] Dự đoán cho mẫu hoa mới:")
print(f"    - Sepal Length: {new_sample_raw[0][0]} cm")
print(f"    - Sepal Width : {new_sample_raw[0][1]} cm")
print(f"    - Petal Length: {new_sample_raw[0][2]} cm")
print(f"    - Petal Width : {new_sample_raw[0][3]} cm")

predicted_probabilities = model.predict_proba(new_sample_scaled)

print(f"\n    Xác suất dự đoán cho từng loài hoa:")
for i, (name, prob) in enumerate(zip(flower_names, predicted_probabilities[0])):
    bar = "█" * int(prob * 30)
    print(f"    [{i}] {name:<12}: {prob:.6f} ({prob*100:.2f}%)  |{bar}|")

predicted_class_index = int(np.argmax(predicted_probabilities, axis=1)[0])
predicted_flower_name = flower_names[predicted_class_index]

print(f"\n    ══════════════════════════════════════════")
print(f"    ✦ Kết luận dự đoán: Loài hoa được dự đoán là")
print(f"      >>> '{predicted_flower_name.upper()}' <<<")
print(f"      (Xác suất: {predicted_probabilities[0][predicted_class_index]*100:.2f}%)")
print(f"    ══════════════════════════════════════════")

print("\n" + "=" * 60)
print("   KẾT THÚC CHƯƠNG TRÌNH")
print("=" * 60)
