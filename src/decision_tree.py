import numpy as np

import numpy as np

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, var_red=None, value=None):
        # Các thuộc tính cho nút quyết định (Decision Node)
        self.feature_index = feature_index  # Chỉ số của đặc trưng dùng để chia (cột nào?)
        self.threshold = threshold          # Giá trị ngưỡng để chia (lớn hơn hay nhỏ hơn?)
        self.left = left                    # Nút con bên trái
        self.right = right                  # Nút con bên phải
        self.var_red = var_red              # Mức độ giảm phương sai (thông tin thêm để debug)
        
        # Thuộc tính cho nút lá (Leaf Node)
        self.value = value                  # Giá trị dự đoán (mean của y) nếu đây là nút lá

class DecisionTreeRegressor:
    def __init__(self, min_samples_split=2, max_depth=2):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def fit(self, X, y):
        """Hàm huấn luyện mô hình"""
        self.root = self.build_tree(X, y)

    def predict(self, X):
        """Hàm dự đoán cho nhiều mẫu"""
        return [self.make_prediction(x, self.root) for x in X]
    
    def calculate_variance_reduction(self, parent, l_child, r_child):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        reduction = np.var(parent) - (weight_l * np.var(l_child) + weight_r * np.var(r_child))
        return reduction
    
    def get_best_split(self, dataset, num_samples, num_features):
            ''' Tìm đặc trưng và ngưỡng tốt nhất để chia dữ liệu '''
            best_split = {}
            max_var_red = -float("inf")

            # Duyệt qua từng cột đặc trưng
            for feature_index in range(num_features):
                feature_values = dataset[:, feature_index]
                possible_thresholds = np.unique(feature_values)

                # Duyệt qua từng ngưỡng giá trị của đặc trưng đó
                for threshold in possible_thresholds:
                    # Tách dữ liệu thành 2 phần: trái (<= threshold) và phải (> threshold)
                    dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
                    dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])

                    # Chỉ xét nếu cả 2 bên đều có dữ liệu
                    if len(dataset_left) > 0 and len(dataset_right) > 0:
                        y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                        
                        # Tính mức độ giảm phương sai
                        curr_var_red = self.calculate_variance_reduction(y, left_y, right_y)

                        # Nếu tìm thấy cách chia tốt hơn thì lưu lại
                        if curr_var_red > max_var_red:
                            best_split["feature_index"] = feature_index
                            best_split["threshold"] = threshold
                            best_split["dataset_left"] = dataset_left
                            best_split["dataset_right"] = dataset_right
                            best_split["var_red"] = curr_var_red
                            max_var_red = curr_var_red
                            
            return best_split
    
    def build_tree(self, X, y, curr_depth=0):
        num_samples, num_features = np.shape(X)
        # Gộp X và y lại để dễ xử lý trong hàm split
        dataset = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        
        # Điều kiện dừng (Stopping criteria)
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            # Tìm cách chia tốt nhất
            best_split = self.get_best_split(dataset, num_samples, num_features)
            
            # Nếu tìm được cách chia có lợi (giảm được phương sai)
            if best_split.get("var_red", 0) > 0:
                # Xây dựng cây con bên trái
                left_subtree = self.build_tree(best_split["dataset_left"][:, :-1], 
                                               best_split["dataset_left"][:, -1], 
                                               curr_depth + 1)
                # Xây dựng cây con bên phải
                right_subtree = self.build_tree(best_split["dataset_right"][:, :-1], 
                                                best_split["dataset_right"][:, -1], 
                                                curr_depth + 1)
                
                # Trả về Nút quyết định
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["var_red"])
        
        # Nếu đạt điều kiện dừng, trả về Nút lá (Leaf Node)
        leaf_value = np.mean(y)
        return Node(value=leaf_value)
    
    def make_prediction(self, x, tree):
        # Nếu là nút lá, trả về giá trị dự đoán
        if tree.value is not None:
            return tree.value
        
        # Nếu không, tiếp tục đi xuống cây
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)