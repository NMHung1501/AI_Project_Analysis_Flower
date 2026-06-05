# =============================================================================
# PHÂN LOẠI HOA IRIS BẰNG MẠNG NƠRON NHÂN TẠO (ANN)
# Sử dụng Scikit-learn MLPClassifier (tương thích Python 3.14)
# Kiến trúc: Input(4) → Dense(8, ReLU) → Dense(6, ReLU) → Output(3, Softmax)
# =============================================================================

# --- Bước 0: Import các thư viện cần thiết ---
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss, accuracy_score

# Đặt seed để đảm bảo kết quả có thể tái tạo (reproducibility)
np.random.seed(42)

print("=" * 60)
print("   PHÂN LOẠI HOA IRIS BẰNG MẠNG NƠRON NHÂN TẠO (ANN)")
print("   Backend: sklearn.neural_network.MLPClassifier")
print("=" * 60)


# =============================================================================
# BƯỚC 1: CHUẨN BỊ DỮ LIỆU (DATA PREPARATION)
# =============================================================================

# --- 1.1: Tải bộ dữ liệu Iris ---
# load_iris() trả về một đối tượng chứa:
#   .data   -> Ma trận đặc trưng (150 mẫu x 4 đặc trưng)
#   .target -> Mảng nhãn (0, 1, 2 tương ứng 3 loài hoa)
iris = load_iris()
X = iris.data          # Ma trận đặc trưng: shape (150, 4)
y = iris.target        # Nhãn gốc dạng số nguyên: shape (150,)

flower_names = iris.target_names  # ['setosa', 'versicolor', 'virginica']

print(f"\n[1] Thông tin bộ dữ liệu Iris:")
print(f"    - Số lượng mẫu    : {X.shape[0]}")
print(f"    - Số lượng đặc trưng: {X.shape[1]}")
print(f"    - Các đặc trưng   : {iris.feature_names}")
print(f"    - Các loài hoa    : {flower_names}")

# --- 1.2: Áp dụng One-Hot Encoding cho nhãn (y) ---
# Mạng nơron với hàm softmax ở đầu ra yêu cầu nhãn dạng vector nhị phân.
# Ví dụ: class 0 (setosa)     -> [1, 0, 0]
#         class 1 (versicolor) -> [0, 1, 0]
#         class 2 (virginica)  -> [0, 0, 1]
encoder = OneHotEncoder(sparse_output=False)

# reshape(-1, 1) chuyển mảng 1D (150,) thành 2D (150, 1) vì OneHotEncoder yêu cầu đầu vào 2D
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

print(f"\n[2] Kết quả One-Hot Encoding (5 mẫu đầu tiên):")
print(f"    Nhãn gốc : {y[:5]}")
print(f"    Sau encoding:\n{y_encoded[:5]}")

# --- 1.3: Chuẩn hoá đặc trưng (Feature Scaling) ---
# StandardScaler đưa các đặc trưng về phân phối chuẩn (mean=0, std=1)
# Giúp mạng nơron hội tụ nhanh hơn và ổn định hơn
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 1.4: Chia dữ liệu thành tập huấn luyện (80%) và tập kiểm tra (20%) ---
# test_size=0.2  -> 20% dữ liệu dùng để kiểm tra (30 mẫu)
# random_state=42 -> Đặt seed để kết quả chia ổn định khi chạy lại
# stratify=y      -> Đảm bảo tỉ lệ các lớp đồng đều trong cả 2 tập
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n[3] Chia dữ liệu Train/Test:")
print(f"    - Tập huấn luyện (Train): {X_train.shape[0]} mẫu ({X_train.shape[0]/150*100:.0f}%)")
print(f"    - Tập kiểm tra  (Test) : {X_test.shape[0]} mẫu ({X_test.shape[0]/150*100:.0f}%)")


# =============================================================================
# BƯỚC 2: XÂY DỰNG KIẾN TRÚC MÔ HÌNH (MODEL ARCHITECTURE)
# =============================================================================

# MLPClassifier = Multi-Layer Perceptron Classifier (Mạng Nơron Nhiều Lớp)
# Đây là cách triển khai ANN trong scikit-learn, tương đương Keras Sequential
#
# Kiến trúc tương đương bài tập yêu cầu:
#   Lớp ẩn 1: 8 nơron, ReLU  ←→ Dense(8, activation='relu', input_shape=(4,))
#   Lớp ẩn 2: 6 nơron, ReLU  ←→ Dense(6, activation='relu')
#   Lớp đầu ra: 3 nơron, Softmax ←→ Dense(3, activation='softmax')

model = MLPClassifier(
    # hidden_layer_sizes=(8, 6): 2 lớp ẩn với 8 và 6 nơron tương ứng
    hidden_layer_sizes=(8, 6),

    # activation='relu': Hàm kích hoạt ReLU cho các lớp ẩn
    #                     f(x) = max(0, x)
    activation='relu',

    # solver='adam': Thuật toán tối ưu Adam (tương đương Adam optimizer trong Keras)
    solver='adam',

    # learning_rate_init=0.01: Learning rate ban đầu = 0.01
    learning_rate_init=0.01,

    # batch_size=10: Kích thước mini-batch (tương đương batch_size=10 trong Keras)
    batch_size=10,

    # max_iter=50: Số lần duyệt tối đa (tương đương epochs=50)
    max_iter=50,

    # random_state=42: Seed để kết quả tái tạo được
    random_state=42,

    # verbose=True: Hiển thị tiến trình huấn luyện
    verbose=True,

    # Hàm mất mát: log_loss (tương đương categorical_crossentropy)
    # Lưu ý: MLPClassifier luôn dùng log_loss, không cần khai báo
)

print(f"\n[4] Kiến trúc mô hình ANN:")
print(f"    Input Layer  : 4 đặc trưng")
print(f"    Hidden Layer 1: 8 nơron · ReLU")
print(f"    Hidden Layer 2: 6 nơron · ReLU")
print(f"    Output Layer : 3 nơron · Softmax (ẩn trong MLPClassifier)")
print(f"    Optimizer    : Adam (lr=0.01)")
print(f"    Loss         : Log Loss (= Categorical Cross-Entropy)")


# =============================================================================
# BƯỚC 3: BIÊN DỊCH VÀ HUẤN LUYỆN MÔ HÌNH (COMPILATION & TRAINING)
# =============================================================================

print(f"\n[5] Bắt đầu huấn luyện ({model.max_iter} epochs)...")
print("-" * 60)

# model.fit() thực hiện toàn bộ quá trình huấn luyện
# Lưu ý: verbose=True sẽ in loss sau mỗi epoch nếu hội tụ
model.fit(X_train, y_train)

print("-" * 60)
print("[5] Huấn luyện hoàn tất!")
print(f"    Số epoch thực tế đã chạy: {model.n_iter_}")
print(f"    Loss cuối cùng (training): {model.loss_:.6f}")


# =============================================================================
# BƯỚC 4: ĐÁNH GIÁ MÔ HÌNH (EVALUATION)
# =============================================================================

# Dự đoán trên tập kiểm tra
y_pred      = model.predict(X_test)          # Nhãn dự đoán (0, 1, 2)
y_pred_prob = model.predict_proba(X_test)    # Xác suất dự đoán

# Tính độ chính xác (Accuracy)
accuracy = accuracy_score(y_test, y_pred)

# Tính loss (Log Loss = Categorical Cross-Entropy)
test_loss = log_loss(y_test, y_pred_prob)

print(f"\n[6] Kết quả đánh giá trên tập kiểm tra ({X_test.shape[0]} mẫu):")
print(f"    ✦ Loss (Log Loss / Cross-Entropy): {test_loss:.4f}")
print(f"    ✦ Accuracy (Độ chính xác)        : {accuracy * 100:.2f}%")


# =============================================================================
# BƯỚC 5: DỰ ĐOÁN MẪU MỚI (PREDICTION)
# =============================================================================

# --- 5.1: Chuẩn bị mẫu dữ liệu mới để dự đoán ---
# Mẫu: [Sepal Length=5.1, Sepal Width=3.5, Petal Length=1.4, Petal Width=0.2]
new_sample_raw = np.array([[5.1, 3.5, 1.4, 0.2]])  # shape (1, 4)

# QUAN TRỌNG: Phải chuẩn hoá mẫu mới bằng CÙNG scaler đã dùng lúc train
new_sample_scaled = scaler.transform(new_sample_raw)

print(f"\n[7] Dự đoán cho mẫu hoa mới:")
print(f"    - Sepal Length (Chiều dài đài hoa): {new_sample_raw[0][0]} cm")
print(f"    - Sepal Width  (Chiều rộng đài hoa): {new_sample_raw[0][1]} cm")
print(f"    - Petal Length (Chiều dài cánh hoa): {new_sample_raw[0][2]} cm")
print(f"    - Petal Width  (Chiều rộng cánh hoa): {new_sample_raw[0][3]} cm")

# --- 5.2: Thực hiện dự đoán ---
# predict_proba() trả về ma trận xác suất shape (1, 3)
predicted_probabilities = model.predict_proba(new_sample_scaled)

print(f"\n    Xác suất dự đoán cho từng loài hoa:")
for i, (name, prob) in enumerate(zip(flower_names, predicted_probabilities[0])):
    bar = "█" * int(prob * 30)
    print(f"    [{i}] {name:<12}: {prob:.6f} ({prob*100:.2f}%)  |{bar}|")

# --- 5.3: Xác định lớp được dự đoán ---
predicted_class_index = int(np.argmax(predicted_probabilities, axis=1)[0])
predicted_flower_name = flower_names[predicted_class_index]

print(f"\n    ══════════════════════════════════════════")
print(f"    ✦ Kết luận dự đoán: Loài hoa được dự đoán là")
print(f"      >>> '{predicted_flower_name.upper()}' <<<")
print(f"      (Chỉ số lớp: {predicted_class_index} | Xác suất: {predicted_probabilities[0][predicted_class_index]*100:.2f}%)")
print(f"    ══════════════════════════════════════════")

print("\n" + "=" * 60)
print("   KẾT THÚC CHƯƠNG TRÌNH")
print("=" * 60)
