import numpy as np

class CustomStandardScaler:
    def fit_transform(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        return (X - self.mean_) / (self.scale_ + 1e-8)
        
    def transform(self, X):
        return (X - self.mean_) / (self.scale_ + 1e-8)

def train_test_split_custom(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_len = int(len(X) * test_size)
    test_idx, train_idx = indices[:test_len], indices[test_len:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

class NumpyANN:
    def __init__(self, layer_sizes=(4, 8, 6, 3), learning_rate=0.01, epochs=50):
        self.layer_sizes = layer_sizes
        self.lr = learning_rate
        self.epochs = epochs
        self.params = self.init_params()

    def init_params(self):
        np.random.seed(42)
        params = {}
        for i in range(1, len(self.layer_sizes)):
            params[f'W{i}'] = np.random.randn(self.layer_sizes[i-1], self.layer_sizes[i]) * 0.1
            params[f'b{i}'] = np.zeros((1, self.layer_sizes[i]))
        return params

    def relu(self, Z):
        return np.maximum(0, Z)
        
    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        
    def relu_deriv(self, Z):
        return (Z > 0).astype(float)
        
    def forward(self, X):
        cache = {'A0': X}
        # Layer 1
        Z1 = X.dot(self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)
        cache['Z1'], cache['A1'] = Z1, A1
        # Layer 2
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        A2 = self.relu(Z2)
        cache['Z2'], cache['A2'] = Z2, A2
        # Layer 3
        Z3 = A2.dot(self.params['W3']) + self.params['b3']
        A3 = self.softmax(Z3)
        cache['Z3'], cache['A3'] = Z3, A3
        return A3, cache
        
    def backward(self, Y, cache):
        m = Y.shape[0]
        dZ3 = cache['A3'] - Y
        dW3 = cache['A2'].T.dot(dZ3) / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m
        
        dA2 = dZ3.dot(self.params['W3'].T)
        dZ2 = dA2 * self.relu_deriv(cache['Z2'])
        dW2 = cache['A1'].T.dot(dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        dA1 = dZ2.dot(self.params['W2'].T)
        dZ1 = dA1 * self.relu_deriv(cache['Z1'])
        dW1 = cache['A0'].T.dot(dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}
        
    def fit(self, X, y):
        history = {'loss': [], 'accuracy': []}
        num_classes = self.layer_sizes[-1]
        Y_onehot = np.zeros((y.size, num_classes))
        Y_onehot[np.arange(y.size), y] = 1
        
        for i in range(self.epochs):
            A3, cache = self.forward(X)
            loss = -np.mean(np.sum(Y_onehot * np.log(A3 + 1e-8), axis=1))
            acc = np.mean(np.argmax(A3, axis=1) == y)
            history['loss'].append(round(float(loss), 4))
            history['accuracy'].append(round(float(acc), 4))
            
            grads = self.backward(Y_onehot, cache)
            for key in self.params:
                self.params[key] -= self.lr * grads[key]
                
        return history
        
    def predict_proba(self, X):
        A3, _ = self.forward(X)
        return A3
        
    def predict(self, X):
        A3 = self.predict_proba(X)
        return np.argmax(A3, axis=1)

def get_iris_dataset():
    # Hardcoded Iris dataset to avoid requiring scikit-learn or network downloads
    # 150 samples: 50 Setosa (0), 50 Versicolor (1), 50 Virginica (2)
    # [sepal length, sepal width, petal length, petal width]
    X = np.array([
        [5.1,3.5,1.4,0.2],[4.9,3.0,1.4,0.2],[4.7,3.2,1.3,0.2],[4.6,3.1,1.5,0.2],[5.0,3.6,1.4,0.2],[5.4,3.9,1.7,0.4],[4.6,3.4,1.4,0.3],[5.0,3.4,1.5,0.2],[4.4,2.9,1.4,0.2],[4.9,3.1,1.5,0.1],
        [5.4,3.7,1.5,0.2],[4.8,3.4,1.6,0.2],[4.8,3.0,1.4,0.1],[4.3,3.0,1.1,0.1],[5.8,4.0,1.2,0.2],[5.7,4.4,1.5,0.4],[5.4,3.9,1.3,0.4],[5.1,3.5,1.4,0.3],[5.7,3.8,1.7,0.3],[5.1,3.8,1.5,0.3],
        [5.4,3.4,1.7,0.2],[5.1,3.7,1.5,0.4],[4.6,3.6,1.0,0.2],[5.1,3.3,1.7,0.5],[4.8,3.4,1.9,0.2],[5.0,3.0,1.6,0.2],[5.0,3.4,1.6,0.4],[5.2,3.5,1.5,0.2],[5.2,3.4,1.4,0.2],[4.7,3.2,1.6,0.2],
        [4.8,3.1,1.6,0.2],[5.4,3.4,1.5,0.4],[5.2,4.1,1.5,0.1],[5.5,4.2,1.4,0.2],[4.9,3.1,1.5,0.2],[5.0,3.2,1.2,0.2],[5.5,3.5,1.3,0.2],[4.9,3.6,1.4,0.1],[4.4,3.0,1.3,0.2],[5.1,3.4,1.5,0.2],
        [5.0,3.5,1.3,0.3],[4.5,2.3,1.3,0.3],[4.4,3.2,1.3,0.2],[5.0,3.5,1.6,0.6],[5.1,3.8,1.9,0.4],[4.8,3.0,1.4,0.3],[5.1,3.8,1.6,0.2],[4.6,3.2,1.4,0.2],[5.3,3.7,1.5,0.2],[5.0,3.3,1.4,0.2],
        [7.0,3.2,4.7,1.4],[6.4,3.2,4.5,1.5],[6.9,3.1,4.9,1.5],[5.5,2.3,4.0,1.3],[6.5,2.8,4.6,1.5],[5.7,2.8,4.5,1.3],[6.3,3.3,4.7,1.6],[4.9,2.4,3.3,1.0],[6.6,2.9,4.6,1.3],[5.2,2.7,3.9,1.4],
        [5.0,2.0,3.5,1.0],[5.9,3.0,4.2,1.5],[6.0,2.2,4.0,1.0],[6.1,2.9,4.7,1.4],[5.6,2.9,3.6,1.3],[6.7,3.1,4.4,1.4],[5.6,3.0,4.5,1.5],[5.8,2.7,4.1,1.0],[6.2,2.2,4.5,1.5],[5.6,2.5,3.9,1.1],
        [5.9,3.2,4.8,1.8],[6.1,2.8,4.0,1.3],[6.3,2.5,4.9,1.5],[6.1,2.8,4.7,1.2],[6.4,2.9,4.3,1.3],[6.6,3.0,4.4,1.4],[6.8,2.8,4.8,1.4],[6.7,3.0,5.0,1.7],[6.0,2.9,4.5,1.5],[5.7,2.6,3.5,1.0],
        [5.5,2.4,3.8,1.1],[5.5,2.4,3.7,1.0],[5.8,2.7,3.9,1.2],[6.0,2.7,5.1,1.6],[5.4,3.0,4.5,1.5],[6.0,3.4,4.5,1.6],[6.7,3.1,4.7,1.5],[6.3,2.3,4.4,1.3],[5.6,3.0,4.1,1.3],[5.5,2.5,4.0,1.3],
        [5.5,2.6,4.4,1.2],[6.1,3.0,4.6,1.4],[5.8,2.6,4.0,1.2],[5.0,2.3,3.3,1.0],[5.6,2.7,4.2,1.3],[5.7,3.0,4.2,1.2],[5.7,2.9,4.2,1.3],[6.2,2.9,4.3,1.3],[5.1,2.5,3.0,1.1],[5.7,2.8,4.1,1.3],
        [6.3,3.3,6.0,2.5],[5.8,2.7,5.1,1.9],[7.1,3.0,5.9,2.1],[6.3,2.9,5.6,1.8],[6.5,3.0,5.8,2.2],[7.6,3.0,6.6,2.1],[4.9,2.5,4.5,1.7],[7.3,2.9,6.3,1.8],[6.7,2.5,5.8,1.8],[7.2,3.6,6.1,2.5],
        [6.5,3.2,5.1,2.0],[6.4,2.7,5.3,1.9],[6.8,3.0,5.5,2.1],[5.7,2.5,5.0,2.0],[5.8,2.8,5.1,2.4],[6.4,3.2,5.3,2.3],[6.5,3.0,5.5,1.8],[7.7,3.8,6.7,2.2],[7.7,2.6,6.9,2.3],[6.0,2.2,5.0,1.5],
        [6.9,3.2,5.7,2.3],[5.6,2.8,4.9,2.0],[7.7,2.8,6.7,2.0],[6.3,2.7,4.9,1.8],[6.7,3.3,5.7,2.1],[7.2,3.2,6.0,1.8],[6.2,2.8,4.8,1.8],[6.1,3.0,4.9,1.8],[6.4,2.8,5.6,2.1],[7.2,3.0,5.8,1.6],
        [7.4,2.8,6.1,1.9],[7.9,3.8,6.4,2.0],[6.4,2.8,5.6,2.2],[6.3,2.8,5.1,1.5],[6.1,2.6,5.6,1.4],[7.7,3.0,6.1,2.3],[6.3,3.4,5.6,2.4],[6.4,3.1,5.5,1.8],[6.0,3.0,4.8,1.8],[6.9,3.1,5.4,2.1],
        [6.7,3.1,5.6,2.4],[6.9,3.1,5.1,2.3],[5.8,2.7,5.1,1.9],[6.8,3.2,5.9,2.3],[6.7,3.3,5.7,2.5],[6.7,3.0,5.2,2.3],[6.3,2.5,5.0,1.9],[6.5,3.0,5.2,2.0],[6.2,3.4,5.4,2.3],[5.9,3.0,5.1,1.8]
    ])
    y = np.array([0]*50 + [1]*50 + [2]*50)
    return X, y, ['setosa', 'versicolor', 'virginica'], ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
