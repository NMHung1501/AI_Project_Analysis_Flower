"""
Flask Web Application - ANN Iris Flower Classifier
Backend sử dụng NumpyANN 100% Thuần Python (Bypass lỗi cài đặt sklearn/TF)
Tương thích tuyệt đối Python 3.14
"""

import os
import base64
# pyrefly: ignore [missing-import]
import numpy as np
import csv

# pyrefly: ignore [missing-import]
from flask import Flask, render_template, request, jsonify
# pyrefly: ignore [missing-import]
from werkzeug.utils import secure_filename
from numpy_ann import NumpyANN, CustomStandardScaler, get_iris_dataset, train_test_split_custom

# ─── Cấu hình Flask ────────────────────────────────────────
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY']          = 'iris-ann-secret-2024'
app.config['UPLOAD_FOLDER']       = 'uploads'
app.config['MAX_CONTENT_LENGTH']  = 16 * 1024 * 1024  # 16 MB

ALLOWED_DATASET = {'csv', 'xlsx', 'xls'}
ALLOWED_IMAGE   = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Tạo thư mục
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
}

def allowed_file(filename, allowed_set):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_set

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload-dataset', methods=['POST'])
def upload_dataset():
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

        # Read CSV without pandas
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = []
            for row in reader:
                if row: rows.append(row)

        preview_rows = rows[:5]
        preview_dicts = []
        for row in preview_rows:
            preview_dicts.append({header[i]: row[i] for i in range(len(header))})

        info = {
            'filename':     filename,
            'rows':         len(rows),
            'columns':      len(header),
            'column_names': header,
            'preview':      preview_dicts,
            'filepath':     filepath,
        }
        state['dataset_info'] = info
        state['dataset_path'] = filepath

        return jsonify({'success': True, 'data': info})
    except Exception as e:
        return jsonify({'success': False, 'error': f'Lỗi đọc file: {str(e)}'})

@app.route('/api/train', methods=['POST'])
def train_model():
    try:
        body       = request.get_json()
        use_default = body.get('use_default', False)
        epochs      = int(body.get('epochs', 50))
        lr          = float(body.get('lr', 0.01))

        if use_default or state['dataset_path'] is None:
            X, y_raw, class_names, feature_names = get_iris_dataset()
        else:
            fpath = state['dataset_path']
            with open(fpath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                rows = [r for r in reader if r]
                
            X = np.array([r[:-1] for r in rows], dtype=np.float32)
            y_col = np.array([r[-1] for r in rows])
            feature_names = header[:-1]
            
            unique_classes = np.unique(y_col)
            class_map = {name: i for i, name in enumerate(unique_classes)}
            y_raw = np.array([class_map[val] for val in y_col])
            class_names = list(unique_classes.astype(str))

        # Chuẩn hoá
        scaler    = CustomStandardScaler()
        X_scaled  = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split_custom(
            X_scaled, y_raw, test_size=0.2, random_state=42
        )

        model = NumpyANN(layer_sizes=(4, 8, 6, 3), learning_rate=lr, epochs=epochs)
        
        history = model.fit(X_train, y_train)

        y_prob_test = model.predict_proba(X_test)
        test_acc = float(np.mean(np.argmax(y_prob_test, axis=1) == y_test)) * 100
        test_loss = -float(np.mean(np.log(y_prob_test[np.arange(len(y_test)), y_test] + 1e-8)))

        state.update({
            'model':        model,
            'trained':      True,
            'accuracy':     round(test_acc, 2),
            'loss':         round(test_loss, 4),
            'history':      history,
            'class_names':  class_names,
            'feature_names': feature_names,
            'scaler':       scaler,
        })

        return jsonify({
            'success':    True,
            'accuracy':   state['accuracy'],
            'loss':       state['loss'],
            'history':    history,
            'epochs':     epochs,
            'train_size': int(X_train.shape[0]),
            'test_size':  int(X_test.shape[0]),
            'class_names': class_names,
        })
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()})

@app.route('/api/predict-features', methods=['POST'])
def predict_features():
    if not state['trained'] or state['model'] is None:
        return jsonify({'success': False, 'error': 'Mô hình chưa huấn luyện!'})

    try:
        body     = request.get_json()
        features = body.get('features', [])
        if not features:
            return jsonify({'success': False, 'error': 'Không nhận đặc trưng'})

        X_input = np.array([features], dtype=np.float32)

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

@app.route('/api/upload-image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Không tìm thấy file ảnh'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Chưa chọn ảnh'})
    if not allowed_file(file.filename, ALLOWED_IMAGE):
        return jsonify({'success': False, 'error': 'Định dạng không hợp lệ'})

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

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        'trained':        state['trained'],
        'accuracy':       state['accuracy'],
        'loss':           state['loss'],
        'class_names':    state['class_names'],
        'feature_names':  state['feature_names'],
        'dataset_loaded': state['dataset_info'] is not None,
    })

if __name__ == '__main__':
    print("\n" + "═" * 52)
    print("  🌸  ANN Iris Flower Classifier — Web App")
    print("  📡  Truy cập: http://127.0.0.1:5000")
    print("  🔧  Backend: Numpy ANN 100% Thuần (Không cần Sklearn/TF)")
    print("═" * 52 + "\n")
    app.run(debug=True, port=5000, use_reloader=False)
