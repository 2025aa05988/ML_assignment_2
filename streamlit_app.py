
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(
    page_title="Breast Cancer Classification",
    layout="wide",
    page_icon="🔬"
)
st.title("🔬 Breast Cancer Classification Dashboard")

# ==========================================
# 2. Load Models (Hardcoded Filenames)
# ==========================================
# We explicitly list the files we generated in the training phase
MODEL_FILES = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "KNN": "knn_model.pkl",
    "Naive Bayes": "naive_bayes_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "XGBoost": "xgboost_model.pkl"
}

@st.cache_resource
def load_resources():
    resources = {}

    # 1. Load Scaler and Feature Names (Critical)
    try:
        resources["scaler"] = joblib.load("scaler.pkl")
        resources["feature_names"] = joblib.load("feature_names.pkl")
    except FileNotFoundError:
        st.error("❌ Critical Error: 'scaler.pkl' or 'feature_names.pkl' not found.")
        st.info("Please ensure these files are uploaded to the same directory as app.py.")
        return None

    # 2. Load Models
    loaded_models = {}
    for name, filename in MODEL_FILES.items():
        try:
            loaded_models[name] = joblib.load(filename)
        except FileNotFoundError:
            st.warning(f"⚠️ Warning: Model file '{filename}' not found. Skipping {name}.")

    if not loaded_models:
        st.error("❌ No model files found! Please upload the .pkl files.")
        return None

    resources["models"] = loaded_models
    return resources

# Execute Loader
resources = load_resources()

if resources is None:
    st.stop()

models = resources["models"]
scaler = resources["scaler"]
feature_names = resources["feature_names"]

# ==========================================
# 3. Sidebar Configuration
# ==========================================
st.sidebar.header("Configuration")

# Dropdown populated by whatever models successfully loaded
selected_model_name = st.sidebar.selectbox("Select ML Model", list(models.keys()))
model = models[selected_model_name]

# Upload Test Data
st.sidebar.subheader("Upload Test Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV (must contain target column)", type=["csv"])

# ==========================================
# 4. Main Execution
# ==========================================
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # --- Data Preparation ---
        st.write("### Data Preview")
        st.dataframe(df.head())

        # Logic to separate Features (X) and Target (y)
        if 'target' in df.columns:
            X_test = df.drop('target', axis=1)
            y_test = df['target']
        else:
            # Fallback: assume last column is target
            X_test = df.iloc[:, :-1]
            y_test = df.iloc[:, -1]

        # --- Feature Validation ---
        if list(X_test.columns) != feature_names:
            try:
                # Try to select only the required features
                X_test = X_test[feature_names]
            except KeyError:
                st.error("❌ Feature Mismatch! Uploaded CSV does not match training features.")
                st.stop()

        # --- Scaling ---
        X_test_scaled = scaler.transform(X_test)

        # --- Prediction ---
        y_pred = model.predict(X_test_scaled)

        # Get Probabilities for AUC
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_proba = y_pred

        # --- Display Results ---
        st.subheader(f"Results: {selected_model_name}")

        # Metrics
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        m2.metric("AUC", f"{roc_auc_score(y_test, y_proba):.4f}")
        m3.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
        m4.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
        m5.metric("F1 Score", f"{f1_score(y_test, y_pred):.4f}")
        m6.metric("MCC", f"{matthews_corrcoef(y_test, y_pred):.4f}")

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a CSV file to generate predictions.")
