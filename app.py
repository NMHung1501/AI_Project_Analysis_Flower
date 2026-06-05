"""
Flask Web Application - ANN Iris Flower Classifier
Backend sử dụng sklearn.neural_network.MLPClassifier (tương đương ANN)
Tương thích Python 3.14 — không cần TensorFlow
"""

import os
import base64
import numpy as np
import pandas as pd

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# ─── Cấu hình Flask ────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY']          = 'iris-ann-secret-2024'
app.config['UPLOAD_FOLDER']       = 'uploads'
app.config['MAX_CONTENT_LENGTH']  = 16 * 1024 * 1024  # 16 MB

ALLOWED_DATASET = {'csv', 'xlsx', 'xls'}
ALLOWED_IMAGE   = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Tạo thư mục upload / models nếu chưa tồn tại
for d in ['uploads/datasets', 'uploads/images', 'models']:
    os.makedirs(d, exist_ok=True)

# ─── Trạng thái global của mô hình ─────────────────────────
state = {
    'model':        None,
    'trained':      False,
    'accuracy':     None,
    'loss':         None,
    'history':      None,
    'class_names':  ['setosa', 'versicolor', 'virginica'],
    'feature_names':['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
    'dataset_info': None,
    'dataset_path': None,
    'scaler':       None,
    'encoder':      None,
}


# ════════════════════════════════════════════════════════════
# UTILITY
# ════════════════════════════════════════════════════════════

def allowed_file(filename, allowed_set):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_set


def build_ann_model():
    """
    Xây dựng ANN bằng sklearn MLPClassifier.
    Kiến trúc: Input(4) → Hidden(8, relu) → Hidden(6, relu) → Output(3, softmax)
    Tương đương Dense layers trong Keras.
    """
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier(
        hidden_layer_sizes=(8, 6),   # 2 lớp ẩn: 8 và 6 nơron
        activation='relu',
        solver='adam',
        learning_rate_init=0.01,
        max_iter=1,                  # sẽ gọi partial_fit theo epoch
        warm_start=True,
        random_state=42,
    )


# ════════════════════════════════════════════════════════════
# ROUTES — PAGES
# ════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


# ════════════════════════════════════════════════════════════
# API — DATASET UPLOAD
# ════════════════════════════════════════════════════════════

@app.route('/api/upload-dataset', methods=['POST'])
def upload_dataset():
    """Upload file CSV/Excel và trả về preview 5 dòng đầu"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Không tìm thấy file trong request'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Chưa chọn file'})

    if not allowed_file(file.filename, ALLOWED_DATASET):
        return jsonify({'success': False, 'error': 'Chỉ hỗ trợ CSV và Excel (.xlsx, .xls)'})

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads/datasets', filename)
        file.save(filepath)

        df = pd.read_csv(filepath) if filename.endswith('.csv') else pd.read_excel(filepath)

        info = {
            'filename':     filename,
            'rows':         int(df.shape[0]),
            'columns':      int(df.shape[1]),
            'column_names': df.columns.tolist(),
            'preview':      df.head(5).fillna('').to_dict(orient='records'),
            'filepath':     filepath,
        }
        state['dataset_info'] = info
        state['dataset_path'] = filepath

        return jsonify({'success': True, 'data': info})

    except Exception as e:
        return jsonify({'success': False, 'error': f'Lỗi đọc file: {str(e)}'})


# ════════════════════════════════════════════════════════════
# API — TRAIN
# ════════════════════════════════════════════════════════════

@app.route('/api/train', methods=['POST'])
def train_model():
    """
    Huấn luyện mô hình ANN (MLPClassifier).
    Mô phỏng lịch sử loss/accuracy theo từng epoch.
    """
    try:
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import log_loss

        body       = request.get_json()
        use_default = body.get('use_default', False)
        epochs      = int(body.get('epochs', 50))
        batch_size  = int(body.get('batch_size', 10))  # không dùng trực tiếp nhưng ghi nhận
        lr          = float(body.get('lr', 0.01))

        # ── Nạp dữ liệu ──────────────────────────────────────
        if use_default or state['dataset_path'] is None:
            from sklearn.datasets import load_iris
            iris = load_iris()
            X       = iris.data.astype(np.float32)
            y_raw   = iris.target
            class_names   = list(iris.target_names)
            feature_names = list(iris.feature_names)
        else:
            fpath = state['dataset_path']
            df    = pd.read_csv(fpath) if fpath.endswith('.csv') else pd.read_excel(fpath)
            X     = df.iloc[:, :-1].values.astype(np.float32)
            y_col = df.iloc[:, -1]
            feature_names = df.columns[:-1].tolist()
            le    = LabelEncoder()
            y_raw = le.fit_transform(y_col)
            class_names = list(le.classes_.astype(str))

        num_classes = len(np.unique(y_raw))

        # ── Chuẩn hoá & chia dữ liệu ─────────────────────────
        scaler    = StandardScaler()
        X_scaled  = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_raw, test_size=0.2, random_state=42, stratify=y_raw
        )

        # ── Xây dựng model & huấn luyện từng epoch ───────────
        model = MLPClassifier(
            hidden_layer_sizes=(8, 6),
            activation='relu',
            solver='adam',
            learning_rate_init=lr,
            max_iter=1,
            warm_start=True,
            random_state=42,
        )

        history = {'loss': [], 'accuracy': []}
        classes = np.arange(num_classes)

        for epoch in range(epochs):
            model.fit(X_train, y_train)              # warm_start: tiếp tục từ trước
            y_prob  = model.predict_proba(X_train)
            ep_loss = round(log_loss(y_train, y_prob), 4)
            ep_acc  = round(float(np.mean(model.predict(X_train) == y_train)), 4)
            history['loss'].append(ep_loss)
            history['accuracy'].append(ep_acc)

        # ── Đánh giá trên test set ────────────────────────────
        y_prob_test = model.predict_proba(X_test)
        test_loss   = round(float(log_loss(y_test, y_prob_test)), 4)
        test_acc    = round(float(np.mean(model.predict(X_test) == y_test)) * 100, 2)

        # ── Lưu state ─────────────────────────────────────────
        state.update({
            'model':        model,
            'trained':      True,
            'accuracy':     test_acc,
            'loss':         test_loss,
            'history':      history,
            'class_names':  class_names,
            'feature_names': feature_names,
            'scaler':       scaler,
        })

        return jsonify({
            'success':    True,
            'accuracy':   test_acc,
            'loss':       test_loss,
            'history':    history,
            'epochs':     epochs,
            'train_size': int(X_train.shape[0]),
            'test_size':  int(X_test.shape[0]),
            'class_names': class_names,
        })

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()})


# ════════════════════════════════════════════════════════════
# API — PREDICT
# ════════════════════════════════════════════════════════════

@app.route('/api/predict-features', methods=['POST'])
def predict_features():
    """Dự đoán loài hoa từ 4 đặc trưng số"""
    if not state['trained'] or state['model'] is None:
        return jsonify({'success': False, 'error': 'Mô hình chưa huấn luyện. Vui lòng huấn luyện trước!'})

    try:
        body     = request.get_json()
        features = body.get('features', [])
        if not features:
            return jsonify({'success': False, 'error': 'Không nhận được đặc trưng đầu vào'})

        X_input = np.array([features], dtype=np.float32)

        # Chuẩn hoá giống lúc train
        if state['scaler']:
            X_input = state['scaler'].transform(X_input)

        probs         = state['model'].predict_proba(X_input)[0]
        class_names   = state['class_names']
        predicted_idx = int(np.argmax(probs))

        results = [
            {'class': class_names[i], 'probability': round(float(probs[i]) * 100, 2)}
            for i in range(len(class_names))
        ]

        return jsonify({
            'success':         True,
            'results':         results,
            'predicted_class': class_names[predicted_idx],
            'confidence':      round(float(probs[predicted_idx]) * 100, 2),
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ════════════════════════════════════════════════════════════
# API — IMAGE UPLOAD
# ════════════════════════════════════════════════════════════

@app.route('/api/upload-image', methods=['POST'])
def upload_image():
    """Upload ảnh hoa mẫu — trả về base64 để hiển thị trên UI"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Không tìm thấy file ảnh'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Chưa chọn ảnh'})

    if not allowed_file(file.filename, ALLOWED_IMAGE):
        return jsonify({'success': False, 'error': 'Định dạng không hợp lệ (PNG, JPG, GIF, WEBP…)'})

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads/images', filename)
        file.save(filepath)

        with open(filepath, 'rb') as f:
            img_data = f.read()

        ext       = filename.rsplit('.', 1)[1].lower()
        mime_type = {'jpg': 'jpeg', 'jpeg': 'jpeg', 'png': 'png',
                     'gif': 'gif', 'bmp': 'bmp', 'webp': 'webp'}.get(ext, 'jpeg')

        return jsonify({
            'success':   True,
            'filename':  filename,
            'image_b64': f'data:image/{mime_type};base64,{base64.b64encode(img_data).decode()}',
            'filepath':  filepath,
            'size_kb':   round(len(img_data) / 1024, 1),
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ════════════════════════════════════════════════════════════
# API — STATUS
# ════════════════════════════════════════════════════════════

@app.route('/api/status', methods=['GET'])
def get_status():
    """Trả về trạng thái hiện tại của mô hình"""
    return jsonify({
        'trained':        state['trained'],
        'accuracy':       state['accuracy'],
        'loss':           state['loss'],
        'class_names':    state['class_names'],
        'feature_names':  state['feature_names'],
        'dataset_loaded': state['dataset_info'] is not None,
    })


# ════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "═" * 52)
    print("  🌸  ANN Iris Flower Classifier — Web App")
    print("  📡  Truy cập: http://127.0.0.1:5000")
    print("  🔧  Backend: sklearn MLPClassifier (Python 3.14)")
    print("═" * 52 + "\n")
    app.run(debug=True, port=5000, use_reloader=False)
