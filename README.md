# Calories Burned Predictor (Tree-Based Models From Scratch)

Welcome to the **Calories Burned Predictor**! This project focuses on implementing popular tree-based ensemble machine learning algorithms **completely from scratch in Python** and applying them to predict the number of calories a person burns during a workout.

The project demonstrates the inner workings of Decision Trees, Random Forests, Bagging, and XGBoost, bypassing standard machine learning libraries (like `scikit-learn` or `xgboost`) for the core model logic. It also includes an interactive **Streamlit** web application to let users test the models in real-time.

---

## Features

- **Models built from scratch**:
  - **Decision Tree Regressor** (`src/decision_tree.py`)
  - **Random Forest Regressor** (`src/random_forest.py`)
  - **Bagging Regressor** (`src/bagging.py`)
  - **XGBoost Regressor** (`src/xgboost.py`)
- **Data Exploration (EDA) & Evaluation**: Comprehensive Jupyter Notebooks that explore the dataset, preprocess it, and benchmark the custom scratch models against their industry-standard library counterparts.
- **Interactive UI**: A sleek Streamlit application to predict calories burned using inputs such as Gender, Age, Height, Weight, Workout Duration, Heart Rate, and Body Temperature.
- **Model Comparisons**: Visual comparison of predictions made by the 4 models directly in the Streamlit app.

---

## Project Structure

```text
CS115_Tree_Project/
├── app.py                  # Main Streamlit web application script
├── data/                   # Folder containing datasets (train, val, test, calories)
├── notebooks/              # Jupyter notebooks for EDA and model comparisons
│   ├── models/             # Saved `.pkl` trained model artifacts
│   ├── EDA.ipynb           # Exploratory Data Analysis
│   ├── compare_*.ipynb     # Notebooks comparing custom models vs scikit-learn/xgboost
│   └── final_comparision.ipynb # Summary comparisons
├── src/                    # Source code for the from-scratch algorithms
│   ├── decision_tree.py    # Custom Decision Tree implementation
│   ├── random_forest.py    # Custom Random Forest implementation
│   ├── bagging.py          # Custom Bagging implementation
│   └── xgboost.py          # Custom XGBoost implementation
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

---

## Installation & Setup

1. **Clone the repository** (if applicable) or download the project files.
2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

*(Dependencies include: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `streamlit`, and `streamlit-lottie`)*

---

## Usage

### 1. Web Application (Streamlit)
To interact with the models via the browser, run the Streamlit app from your terminal:
```bash
streamlit run app.py
```
This will open a new tab in your web browser where you can input workout details and see predictions from all 4 custom models side-by-side.

### 2. Jupyter Notebooks
To explore how the algorithms were built, trained, and evaluated:
- Launch Jupyter Notebook or Jupyter Lab:
  ```bash
  jupyter notebook
  ```
- Navigate to the `notebooks/` directory and open the `.ipynb` files sequentially to see the EDA (`EDA.ipynb`) and individual model comparisons (`compare_xgboost.ipynb`, `compare_randomforest.ipynb`, etc.).

---

## Core Methodology

The project models the problem as an **Regression** task.
- **Input Features**: `Gender`, `Age`, `Height`, `Weight`, `Duration`, `Heart_Rate`, `Body_Temp`
- **Target Variable**: `Calories`
- **Core logic**: Models construct nodes splitting features using Variance Reduction as the main criteria for regression (calculating how much a split reduces target variance). XGBoost builds upon this by sequentially learning from the residuals (errors) of previous trees.

---

## Notes

- The `src/` directory is critical for both the Notebooks and the Streamlit app. All trained models (saved in `notebooks/models/`) require the specific custom model classes present in `src/` to be loaded properly.
- If the Streamlit application encounters errors during prediction, you can enable the **"Show Debug Mode"** checkbox in the sidebar for complete traceback information.
