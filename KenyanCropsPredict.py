# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib

# #App configuration

# st.set_page_config(page_title="Kenyan Food Crop Prices Dashboard", layout="wide")
# st.title("📊 Kenyan Food Crop Market Analysis & Prediction (2013–2015)")


import streamlit as st
import pandas as pd
import numpy as np
import joblib

def cyclical_features(df):
    df = df.copy()
    if 'Year' in df.columns:
        df['Year'] = pd.to_datetime(df['Year'], format='%Y', errors='coerce')
        df['month'] = df['Year'].dt.month
        df['day_of_week'] = df['Year'].dt.dayofweek
        df['day_of_month'] = df['Year'].dt.day
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['dom_sin'] = np.sin(2 * np.pi * (df['day_of_month'] - 1) / 31)
        df['dom_cos'] = np.cos(2 * np.pi * (df['day_of_month'] - 1) / 31)
        df = df.drop(columns=['Year', 'month', 'day_of_week', 'day_of_month'], errors='ignore')
    return df

# Load trained model and preprocessor
model = joblib.load('/Users/harold/DataAnalyctisandScience/AnalysisPredictionKenyanCrops/best_model.pkl')
preprocessor = joblib.load('/Users/harold/DataAnalyctisandScience/AnalysisPredictionKenyanCrops/preprocessor_pipeline.pkl')

st.title("Kenyan Crop Price Prediction")
st.markdown("Enter the details below to predict the price (in KES and FCFA).")

commodity = st.selectbox("Crop Type (Commodity)", [
    "Cabbages", "Tomatoes", "Carrots", "Ripe Bananas", "Kales",
    "Cooking Bananas", "Mangoes Local", "Avocado", "Beans Rosecoco", "Wheat"])
unit = st.selectbox("Packaging Unit", [
    "Bag", "Lg Box", "Ext Bag", "Med Bunch", "Sm Basket"])
package_weight = st.slider("Package Weight (kg)", min_value=1, max_value=150, value=90)
month = st.slider("Month", 1, 12, 1)
year = st.selectbox("Year", [2016])
volume = st.number_input("Total Volume in Kilograms (kg)", min_value=1, max_value=2000, value=100)

# if st.button("Predict Price"):
#     # Match training columns (add all required columns)
#     input_data = pd.DataFrame({
#         "Commodity_Type": [commodity],
#         "Type_of_Commodity": [commodity],  # Use same value or adjust as needed
#         "produce_variety": [0],            # Use a default or let user select
#         "Unit": [unit],
#         "Package_Type": ["Bag"],           # Use default or let user select
#         "Month": [month],
#         "Day": [1],                        # Use default or let user select
#         "Year": [year],
#         "Volume_in_Kgs": [volume],
#         "package_weight(Kg)": [package_weight]
#     })
if st.button("Predict Price"):
    # List of all columns expected by the pipeline
    expected_cols = [
        "Commodity_Type",
        "Type_of_Commodity",
        "produce_variety",
        "Unit",
        "Package_Type",
        "Month",
        "Day",
        "Year",
        "Volume_in_Kgs",
        "package_weight(Kg)"
    ]

    # Build input data dictionary
    input_dict = {
        "Commodity_Type": commodity,
        "Type_of_Commodity": commodity,
        "produce_variety": 0,
        "Unit": unit,
        "Package_Type": "Bag",
        "Month": month,
        "Day": 1,
        "Year": year,
        "Volume_in_Kgs": volume,
        "package_weight(Kg)": package_weight
    }
     # Create DataFrame with all expected columns
    input_data = pd.DataFrame([{col: input_dict[col] for col in expected_cols}])

    try:
        transformed_input = preprocessor.transform(input_data)
        predicted_price = model.predict(transformed_input)[0]
        price_fcfa = predicted_price * 4.5

        st.success(f"💰 Predicted Price (in KES): {predicted_price:,.2f} KES")
        st.info(f"💱 Equivalent Price (in FCFA): {price_fcfa:,.2f} FCFA")
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")