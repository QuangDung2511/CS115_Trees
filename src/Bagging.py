import numpy as np
from decision_tree import DecisionTreeRegressor

class BaggingRegressor:
    def __init__(self, base_tree_cls, n_estimators=10, min_samples_split=2, max_depth=100):
        # base_tree_cls: Class cây quyết định gốc (Decision_Tree)
        # n_estimators: Số lượng cây con muốn tạo 
        self.base_tree_cls = base_tree_cls 
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.trees = [] # List chứa các cây đã huấn luyện

    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]

        # Tạo và huấn luyện từng cây một
        for _ in range(self.n_estimators):
            # --- PHẦN 1: BOOTSTRAP SAMPLING ---
            # Chọn ngẫu nhiên các chỉ số dòng dữ liệu 
            #(có hoàn lại - replace=True)
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            #Sở dĩ có 2 chữ n_samples đứng cạnh nhau là vì hàm choice
            #nó yêu cầu 2 tham số theo thứ tự:
            # (Lấy trong khoảng nào, Lấy bao nhiêu cái).

            X_sample, y_sample = X[idxs], y[idxs]

            # --- PHẦN 2: HUẤN LUYỆN CÂY CON ---
            # Khởi tạo một cây mới với tham số đã định
            tree = self.base_tree_cls(min_samples_split=self.min_samples_split, 
                                      max_depth=self.max_depth)
            # Huấn luyện cây đó trên tập dữ liệu mẫu vừa random
            tree.fit(X_sample, y_sample)
            
            # Lưu cây vào danh sách
            self.trees.append(tree)

    def predict(self, X):
        # --- PHẦN 3: AGGREGATING (GỘP KẾT QUẢ) ---
        
        # 1. Lấy dự đoán từ tất cả các cây
        # Kết quả là mảng 2D: (số_lượng_cây, số_mẫu)
        # Ví dụ: Có 5 cây, dự đoán cho 3 mẫu -> Matrix (5, 3)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        
        # 2. Tính TRUNG BÌNH CỘNG (Mean) 
        # axis=0 nghĩa là tính dọc theo cột (dọc theo các cây)

        y_pred = np.mean(tree_preds, axis=0)
            
        return y_pred


