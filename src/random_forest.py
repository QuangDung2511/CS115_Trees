import numpy as np
from src.decision_tree import DecisionTreeRegressor   # dùng lại cây bạn đã có

class RandomForestRegressor:
    def __init__(self, n_estimators=10, max_depth=3, min_samples_split=2, max_features=None):
        self.n_estimators = n_estimators              # số cây trong rừng
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features              # số lượng feature được chọn ngẫu nhiên
        self.trees = []                               # danh sách các cây

    def bootstrap_sample(self, X, y):
        """Lấy mẫu bootstrap: chọn ngẫu nhiên có lặp lại"""
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def fit(self, X, y):
        """Huấn luyện toàn bộ forest"""
        self.trees = []

        for _ in range(self.n_estimators):
            # Lấy bootstrap sample
            X_sample, y_sample = self.bootstrap_sample(X, y)

            # Tạo 1 cây mới
            tree = DecisionTreeRegressor(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth
            )

            # Nếu chọn max_features
            if self.max_features is not None:
                feature_idxs = np.random.choice(X.shape[1], self.max_features, replace=False)
            else:
                feature_idxs = np.arange(X.shape[1])

            # Lưu lại feature được chọn để dùng khi predict
            tree.feature_idxs = feature_idxs

            # Fit cây theo subset đặc trưng
            tree.fit(X_sample[:, feature_idxs], y_sample)

            self.trees.append(tree)

    def predict(self, X):
        """Dự đoán theo trung bình của dự đoán từ tất cả cây"""
        tree_preds = np.array([
            tree.predict(X[:, tree.feature_idxs]) for tree in self.trees
        ])

        return np.mean(tree_preds, axis=0)
