import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import PowerTransformer  # For type checking if needed

# --- Load model + preprocessing artifacts ---
try:
    with open("classifier2.pkl", "rb") as model_file:
        artifacts = pickle.load(model_file)
    model = artifacts["model"]
    yeo = artifacts["yeo"]
    columns = artifacts["columns"]
    mapping = artifacts["mapping"]
except FileNotFoundError:
    st.error("Error: 'classifier2.pkl' not found. Run the notebook to generate it.")
    st.stop()
except KeyError as e:
    st.error(f"Error: Missing key in artifacts: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error loading pickle: {e}")
    st.stop()

# --- Load dataset to extract categorical options ---
try:
    df = pd.read_csv("global_house_purchase_dataset.csv")
    countries = sorted(df["country"].dropna().unique())
    cities = sorted(df["city"].dropna().unique())
    property_types = sorted(df["property_type"].dropna().unique())
except FileNotFoundError:
    st.error("Error: 'global_house_purchase_dataset.csv' not found.")
    st.stop()

# --- Streamlit UI setup ---
st.set_page_config(page_title="🏠 House Purchase Predictor", page_icon="🏠")
st.title("🏠 House Purchase Decision Predictor (XGBoost)")
st.write("Enter details to predict whether the buyer will **Buy or Not Buy** the house.")

# --- Furnishing ---
st.subheader("🏡 Furnishing Status")
furnishing_status_selected = st.selectbox("Select Furnishing Status", list(mapping.keys()), index=0)
input_data = {"furnishing_status": mapping[furnishing_status_selected]}

# --- Location & property type ---
st.subheader("🌍 Property Location & Type")
selected_country = st.selectbox("Country", countries, index=0)
selected_city = st.selectbox("City", cities, index=0)
selected_type = st.selectbox("Property Type", property_types, index=0)

# --- Numerical features ---
st.subheader("📊 Numerical Features")

# Default values that tend to make a person BUY (adjusted based on dataset patterns)
def smart_default(col):
    name = col.lower()
    if "salary" in name: return 150000.0  # High salary
    if "price" in name: return 200000.0   # Reasonable price
    if "loan_amount" in name: return 100000.0
    if "ratio" in name: return 0.2        # Low ratio
    if "year" in name or "constructed" in name: return 2019.0
    if "owners" in name: return 1.0
    if "room" in name: return 4.0
    if "bath" in name: return 3.0
    if "garage" in name or "garden" in name: return 1.0
    if "crime" in name or "legal" in name: return 0.0  # Low issues
    if "tenure" in name: return 15.0
    if "expenses" in name: return 500.0   # Low expenses
    if "down" in name: return 50000.0     # High down payment
    if "score" in name or "rating" in name: return 8.0  # High scores
    return 0.0

num_features = [
    col for col in columns
    if not any(prefix in col for prefix in ["country_", "city_", "property_type_"]) and col != "furnishing_status"
]

for col in num_features:
    default_val = smart_default(col)
    min_val = 0.0 if "ratio" not in col.lower() else 0.0  # Prevent negatives
    input_data[col] = st.number_input(f"{col}:", min_value=min_val, value=default_val, step=1.0)

# --- Prepare DataFrame ---
df_input = pd.DataFrame([input_data])

# --- Transformations ---
if "emi_to_income_ratio" in df_input.columns and yeo:
    try:
        df_input["emi_to_income_ratio"] = yeo.transform(df_input[["emi_to_income_ratio"]])
    except Exception as e:
        st.warning(f"Warning: Yeo-Johnson transform failed: {e}. Skipping...")

# --- One-Hot Encoding ---
for col in columns:
    if col not in df_input.columns:
        df_input[col] = 0
    if col.startswith("country_"):
        df_input[col] = 1 if col == f"country_{selected_country}" else 0
    elif col.startswith("city_"):
        df_input[col] = 1 if col == f"city_{selected_city}" else 0
    elif col.startswith("property_type_"):
        df_input[col] = 1 if col == f"property_type_{selected_type}" else 0

# --- Align columns safely ---
df_input = df_input.reindex(columns=columns, fill_value=0)

# Optional: Debug input DataFrame
# st.write("Debug: Input DataFrame", df_input)

# --- Prediction ---
if st.button("🔮 Predict Decision", type="primary"):
    try:
        pred = model.predict(df_input)[0]
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df_input)[0]
            confidence = float(np.max(proba))
            st.progress(confidence)
            st.info(f"Prediction Confidence: {confidence*100:.1f}%")
        else:
            confidence = None

        buy_label = 1  # Assuming 1 = Buy, 0 = Not Buy (common in binary classification)
        label = "✅ BUY" if pred == buy_label else "❌ NOT BUY"

        st.subheader(f"### 🧠 Model Prediction: {label}")
        if confidence:
            st.caption(f"Confidence: {confidence*100:.1f}%")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.caption("Default values are tuned for a likely **BUY** case. Adjust to test NOT BUY scenarios.")