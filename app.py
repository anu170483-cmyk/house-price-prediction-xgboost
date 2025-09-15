import streamlit as st
import pandas as pd
import joblib

# ------------------------------
# Load trained model & feature names
# ------------------------------
model = joblib.load("house_price_model.pkl")
feature_names = joblib.load("feature_names.pkl")  # 44 features used in training

# ------------------------------
# App title
# ------------------------------
st.title("🏡 House Price Prediction App")

# ------------------------------
# Sidebar: User input
# ------------------------------
st.sidebar.header("House Features Input")

overallqual = st.sidebar.slider("Overall Quality (1-10)", 1, 10, 5)
grlivarea = st.sidebar.number_input("Above Ground Living Area (sq ft)", 500, 5000, 1500)
garagecars = st.sidebar.slider("Garage Cars", 0, 4, 2)
totalbsmt = st.sidebar.number_input("Total Basement Area (sq ft)", 0, 3000, 800)
yearbuilt = st.sidebar.number_input("Year Built", 1900, 2025, 2000)

# ------------------------------
# Create input DataFrame
# ------------------------------
raw_input = pd.DataFrame({
    "OverallQual": [overallqual],
    "GrLivArea": [grlivarea],
    "GarageCars": [garagecars],
    "TotalBsmtSF": [totalbsmt],
    "YearBuilt": [yearbuilt]
})

# Expand to full 44-feature DataFrame (missing columns filled with 0)
input_df = pd.DataFrame(columns=feature_names)
for col in raw_input.columns:
    if col in input_df.columns:
        input_df[col] = raw_input[col]
input_df = input_df.fillna(0)

# ------------------------------
# Predict and display
# ------------------------------
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    
    # Show price as metric
    st.metric(label="Estimated House Price", value=f"${prediction:,.0f}")
    
    # Show bar chart of input features
    feature_values = [overallqual, grlivarea, garagecars, totalbsmt, yearbuilt]
    feature_names_list = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "YearBuilt"]
    st.bar_chart(pd.DataFrame({"Value": feature_values}, index=feature_names_list))
    
    # Optional info note
    st.info("Note: Prices are estimated based on the trained model and may vary in real scenarios.")
