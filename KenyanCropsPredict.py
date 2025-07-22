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
import joblib

# Load the trained model and preprocessor
model = joblib.load('/Users/harold/DataAnalyctisandScience/AnalysisPredictionKenyanCrops/best_model2.pkl')
preprocessor = joblib.load('/Users/harold/DataAnalyctisandScience/AnalysisPredictionKenyanCrops/best_model2.pkl')  # Ensure you have saved the original ColumnTransformer

# App title
st.title("Kenyan Crop Price Prediction")
st.markdown("Enter the details below to predict the price (in KES and FCFA).")

# User input form
commodity = st.selectbox("Crop Type (Commodity)", [
    "Cabbages", "Tomatoes", "Carrots", "Ripe Bananas", "Kales",
    "Cooking Bananas", "Mangoes Local", "Avocado", "Beans Rosecoco", "Wheat"])
unit = st.selectbox("Packaging Unit", [
    "Bag", "Lg Box", "Ext Bag", "Med Bunch", "Sm Basket"])
package_weight = st.slider("Package Weight (kg)", min_value=1, max_value=150, value=90)
month = st.slider("Month", 1, 12, 1)
year = st.selectbox("Year", [2016])
volume = st.number_input("Total Volume in Kilograms (kg)", min_value=1, max_value=2000, value=100)

# Make prediction
if st.button("Predict Price"):
    # Create a DataFrame with user inputs
    input_data = pd.DataFrame({
        "Commodity_Type": [commodity],
        "Unit": [unit],
        "Month": [month],
        "Year": [year],
        "Volume_in_Kgs": [volume],
        "package_weight(Kg)": [package_weight]
    })

    try:
        # Apply preprocessing (same as used during training)
        transformed_input = preprocessor.transform(input_data)

        # Predict
        predicted_price = model.predict(transformed_input)[0]
        price_fcfa = predicted_price * 4.5  # Rough KES to FCFA conversion

        # Display results
        st.success(f"💰 Predicted Price (in KES): {predicted_price:,.2f} KES")
        st.info(f"💱 Equivalent Price (in FCFA): {price_fcfa:,.2f} FCFA")
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
