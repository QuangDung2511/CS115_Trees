# 🔥 Calories Burned Predictor: Tree-Based Models From Scratch

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a comprehensive implementation of popular tree-based ensemble machine learning algorithms **completely from scratch**. The project applies these models to predict the calories burned during physical activity based on physiological and exercise data.

---

## 1. The Problem

Accurately estimating energy expenditure (calories burned) is fundamental to fitness tracking and health monitoring. While modern wearables provide estimates, understanding the underlying mechanics of the predictive models is often overlooked. 

### The Dataset
The models are trained on a dataset of **15,000 samples** containing:
- **Physiological Features**: `Gender`, `Age`, `Height`, `Weight`.
- **Workout Features**: `Duration` (min), `Heart Rate` (bpm), `Body Temperature` (°C).
- **Target Variable**: `Calories` (kcal).

### Objective
The core objective is to build a robust regression pipeline that demonstrates the inner workings of:
- **Decision Tree Regressor**
- **Random Forest Regressor**
- **Bagging Regressor**
- **XGBoost Regressor**

By implementing these from scratch (using only `NumPy` and `Pandas` for core logic), we gain a deep understanding of variance reduction, ensemble diversity, and gradient boosting optimization.

---

## 2. The Method / Math

### Decision Tree (Variance Reduction)
Each node in our custom Decision Tree is split by maximizing **Variance Reduction**. For a parent node $P$ split into children $L$ and $R$:

$$
\Delta Var = Var(P) - \left( \frac{n_L}{n_P} Var(L) + \frac{n_R}{n_P} Var(R) \right)
$$

where $Var(y) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \bar{y})^2$. The tree builds recursively until a maximum depth or minimum samples split is reached.

### Random Forest & Bagging
**Bagging** (Bootstrap Aggregating) reduces variance by training multiple trees on different bootstrap samples of the data. 
**Random Forest** enhances this by adding **Feature Randomness**; at each split, only a random subset of $m$ features is considered, where typically $m = \sqrt{p}$ or $\log_2(p)$.

### XGBoost (Extreme Gradient Boosting)
Our XGBoost implementation follows the mathematical framework described by Chen & Guestrin:

1.  **Objective Function**: We minimize a regularized objective:

    $$
    \mathcal{L}^{(t)} = \sum_{i=1}^n [g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)] + \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2
    $$

    where $g_i$ and $h_i$ are the first and second-order gradients of the loss function.

2.  **Optimal Leaf Weight**:

    $$
    w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}
    $$

3.  **Split Scoring (Gain)**:

    $$
    Gain = \frac{1}{2} \left[ \frac{(\sum_{i \in I_L} g_i)^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{(\sum_{i \in I_R} g_i)^2}{\sum_{i \in I_R} h_i + \lambda} - \frac{(\sum_{i \in I} g_i)^2}{\sum_{i \in I} h_i + \lambda} \right] - \gamma
    $$

4.  **Weighted Quantile Sketch**: To handle large datasets efficiently, we implemented an approximate split finding algorithm that proposes candidate points based on the distribution of feature weights (Hessians).

---

## 3. The Result / Demo

### Interactive Dashboard
The project includes a **Streamlit** web application that allows users to interact with the models in real-time.

**Key Features**:
- **Real-time Prediction**: Adjust your workout parameters via sliders and see instant results.
- **Side-by-Side Comparison**: View predictions from all 4 scratch-built models simultaneously.
- **Visual Insights**: A dynamic bar chart compares the model outputs, highlighting how different architectures (e.g., Boosting vs Bagging) value the input features.

> [!TIP]
> Use the **"Show Debug Mode"** in the sidebar if you wish to see the underlying data processing steps or troubleshoot prediction errors.

### Benchmark Results
The models were evaluated on a test set of 3,000 samples. The results below showcase the performance of each custom implementation:

| Model | RMSE  | MAE  | R<sup>2</sup> Score | Runtime (s) | 
| :--- | :---: | :---: | :---: | :---: |
| **🏆 XGBoost** | **2.20** | **1.55** | **0.9988** | 0.77s |
| **Bagging** | 2.77 | 1.80 | 0.9981 | 4.02s |
| **Decision Tree** | 5.56 | 3.60 | 0.9923 | 0.01s |
| **Random Forest** | 5.84 | 4.13 | 0.9915 | 1.92s |



---

## 4. How to Run

### Prerequisites
- Python 3.8 or higher
- Git

### Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/QuangDung2511/CS115_Trees.git
    cd CS115_Trees
    ```

2.  **Set up Virtual Environment**:
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application
Launch the demo using Streamlit:
```bash
streamlit run app.py
```
The application will open automatically in your default browser at `http://localhost:8501`.

### Exploring the Math
To see the step-by-step math and benchmarking against `scikit-learn` and `xgboost` libraries, navigate to the `notebooks/` directory and run:
```bash
jupyter notebook
```

---

## Project Structure
```text
CS115_Tree_Project/
├── app.py                  # Interactive Streamlit UI
├── src/                    # Core Model Logic
│   ├── decision_tree.py    # Recursive Partitioning
│   ├── random_forest.py    # Bootstrap & Feature Subspacing
│   ├── bagging.py          # Parallel Ensembles
│   └── xgboost.py          # Gradient Boosting with Taylor Expansion
├── notebooks/              # EDA & Benchmarking
└── data/                   # Exercise Dataset
```
