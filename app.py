import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import traceback  # NEW: For detailed error tracking

# --- 1. SETUP & IMPORTS ---
st.set_page_config(page_title="Calories Burned Predictor", layout="wide")

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# IMPORT CUSTOM CLASSES
try:
    from src.xgboost import XGBoostRegressor
    from src.decision_tree import DecisionTreeRegressor
    from src.random_forest import RandomForestRegressor
    from src.bagging import BaggingRegressor
except ImportError as e:
    st.error(f"Error importing custom classes: {e}")
    st.stop()

# --- 2. LOAD MODELS ---
@st.cache_resource
def load_models():
    models = {}
    # Use absolute path relative to this script
    current_dir = os.path.dirname(__file__)
    model_dir = os.path.join(current_dir, "notebooks", "models")
    
    # Define your model files here. Keys must match what you want to see in the UI.
    files = {
        "XGBoost": "scratch_xgboost.pkl",
        "Random Forest": "scratch_randomforest.pkl",
        "Bagging": "scratch_bagging.pkl",
        "Decision Tree": "scratch_decision_tree.pkl"
    }
    
    if not os.path.exists(model_dir):
        st.error(f"⚠️ Error: The folder `{model_dir}` does not exist.")
        return {}

    for name, filename in files.items():
        path = os.path.join(model_dir, filename)
        if os.path.exists(path):
            try:
                models[name] = joblib.load(path)
            except Exception as e:
                st.warning(f"Could not load {name}: {e}")
        else:
            st.warning(f"File not found: {filename} (Expected in notebooks/models)")
            
    return models

models = load_models()

# --- 3. UI LAYOUT ---
st.title("🔥 Calories Burned Prediction")
st.markdown("""
This app compares **4 Custom Machine Learning Models** built from scratch.
Enter your workout details on the left to see how many calories each model predicts you burned.
""")

# --- 4. SIDEBAR INPUTS ---
st.sidebar.header("Your Workout Details")

# NEW: Debug Mode Toggle
debug_mode = st.sidebar.checkbox("Show Debug Mode (View Errors)", value=False)

def user_input_features():
    gender_select = st.sidebar.radio("Gender", ["Male", "Female"])
    age = st.sidebar.slider("Age (years)", 10, 90, 25)
    height = st.sidebar.slider("Height (cm)", 120, 220, 175)
    weight = st.sidebar.slider("Weight (kg)", 30, 150, 70)
    duration = st.sidebar.slider("Duration (min)", 1, 180, 30)
    heart_rate = st.sidebar.slider("Heart Rate (bpm)", 60, 200, 140)
    body_temp = st.sidebar.slider("Body Temp (°C)", 36.0, 42.0, 40.0)
    
    gender_enc = 0 if gender_select == "Male" else 1
    
    data = {
        'Gender': gender_enc,
        'Age': age,
        'Height': height,
        'Weight': weight,
        'Duration': duration,
        'Heart_Rate': heart_rate,
        'Body_Temp': body_temp
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

st.subheader("Your Input Parameters")
st.write(input_df)

# --- 5. PREDICTION LOGIC ---
if st.button("Predict Calories"):
    if not models:
        st.error("No models loaded. Please check your .pkl files.")
    else:
        results = []
        
        # Create a placeholder for debug messages
        debug_container = st.empty()
        
        for name, model in models.items():
            try:
                # Handle different input expectations (DataFrame vs Numpy)
                if "XGBoost" in name:
                    pred = model.predict(input_df)[0]
                else:
                    pred = model.predict(input_df.values)[0]
                
                results.append({"Model": name, "Calories": pred})
                
            except Exception as e:
                # If Debug Mode is ON, show exactly why it crashed
                if debug_mode:
                    with debug_container.container():
                        st.error(f"❌ Error in **{name}**:")
                        st.code(traceback.format_exc()) # Shows the full error log
                else:
                    st.warning(f"⚠️ {name} failed to predict. (Enable Debug Mode to see why)")

        # --- 6. VISUALIZATION ---
        if results:
            results_df = pd.DataFrame(results)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Predictions")
                
                # --- SAFETY FIX ---
                # Check for "XGBoost" (matching the key in files dictionary)
                if "XGBoost" in results_df['Model'].values:
                    best_pred = results_df.loc[results_df['Model'] == "XGBoost", "Calories"].values[0]
                    label_text = "Estimated Calories (XGBoost)"
                else:
                    # Fallback to whatever worked first
                    best_pred = results_df.iloc[0]['Calories']
                    best_model_name = results_df.iloc[0]['Model']
                    label_text = f"Estimated Calories ({best_model_name})"
                
                st.metric(label=label_text, value=f"{best_pred:.1f} kcal")
                st.table(results_df.set_index("Model"))

            with col2:
                st.subheader("Model Comparison")
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(data=results_df, x="Model", y="Calories", palette="viridis", ax=ax)
                ax.set_ylabel("Calories Burned")
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.1f')
                st.pyplot(fig)