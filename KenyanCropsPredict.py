import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_extras.metric_cards import style_metric_cards


# Function used inside the saved pipeline
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

# Load model and preprocessor
model = joblib.load('/Users/harold/DataAnalyctisandScience/AnalysisPredictionKenyanCrops/best_model.pkl')
preprocessor = joblib.load('/Users/harold/DataAnalyctisandScience/AnalysisPredictionKenyanCrops/preprocessor_pipeline.pkl')

# Mapping dictionaries
produce_variety_mapping = {
    'Cereals': {
        'Dry Maize': 15, 'Green Maize': 19, 'Finger Millet': 16, 'Sorghum': 35, 'Wheat': 39
    },
    'Horticulture': {
        'Cabbages': 5, 'Cooking Bananas': 11, 'Ripe Bananas': 34, 'Carrots': 7, 'Tomatoes': 38,
        'Onions Dry': 28, 'Spring Onions': 36, 'Chillies': 10, 'Cucumber': 13, 'Capsicums': 6,
        'Brinjals': 4, 'Cauliflower': 9, 'Lettuce': 23, 'Passion Fruits': 30, 'Oranges': 29,
        'Lemons': 22, 'Mangoes Local': 25, 'Mangoes Ngowe': 26, 'Limes': 24, 'Pineapples': 32,
        'Pawpaw': 31, 'Avocado': 0, 'Kales': 21
    },
    'Legumes': {
        'Beans Canadian': 1, 'Beans Rosecoco': 3, 'Beans Mwitemania': 2, 'Mwezi Moja': 27,
        'Dolichos (Njahi)': 14, 'Green Gram': 18, 'Cowpeas': 12, 'Fresh Peas': 17, 'Groundnuts': 20
    },
    'Roots & Tubers': {
        'Red Irish Potatoes': 33, 'White Irish Potatoes': 40, 'Cassava Fresh': 8, 'Sweet Potatoes': 37
    }
}

produce_variety_map = {
    'Cereals': 0,
    'Horticulture': 1,
    'Legumes': 2,
    'Roots & Tubers': 3
}

unit_map = {
    'Ext Bag': 2, 'Med Bunch': 4, 'Lg Box': 3, 'Net': 7, 'Bag': 0,
    'Sm Basket': 5, 'Dozen': 1, 'Crate': 6
}

# --- App Layout ---
st.set_page_config(page_title="Kenyan Crop Price Predictor", layout="wide")

st.markdown("""
    <style>
            [data-testid="metric-container"] {
            background-color: #f0f2f6;
            border: 1px solid #d3d3d3;
            padding: 15px;
            border-radius: 10px;
            color: #1a1a1a;
        }
        [data-testid="metric-container"] > div {
            font-size: 18px;
        }
        .main {
            background-color: black;
        }
        h1, h4, .stMarkdown, .stSelectbox, .stSlider, .stNumberInput, .stButton {
            font-family: 'Segoe UI', sans-serif;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: black;
            border-radius: 8px;
        }
       
    
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>🌾 Kenyan Crop Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: red;'>Estimate market price in KES & FCFA</h4>", unsafe_allow_html=True)
st.write("---")

# Sidebar Inputs
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/415/415733.png", width=100)
st.sidebar.header("🔧 Prediction Parameters")

with st.sidebar.form("input_form"):
    category = st.selectbox("Produce Variety Category", list(produce_variety_mapping.keys()))
    commodity = st.selectbox("Commodity Type", list(produce_variety_mapping[category].keys()))
    commodity_id = produce_variety_mapping[category][commodity]
    produce_variety_id = produce_variety_map[category]

    unit = st.selectbox("Unit Type", list(unit_map.keys()))
    unit_id = unit_map[unit]

    month = st.slider("Month", 1, 12, 6)
    year = st.selectbox("Year", [2016, 2017, 2018, 2019])
    volume = st.number_input("Volume in Kilograms (kg)", min_value=1, max_value=1000, value=100)

    submitted = st.form_submit_button("🔮 Predict")

if submitted:
    input_data = pd.DataFrame({
        "Commodity_Type": [commodity_id],
        "Type_of_Commodity": [commodity_id],
        "produce_variety": [produce_variety_id],
        "Unit": [unit],
        "Package_Type": [unit_id],
        "Month": [month],
        "Day": [1],
        "Year": [year],
        "Volume_in_Kgs": [volume],
        "package_weight(Kg)": [90]
    })

    try:
        transformed_input = preprocessor.transform(input_data)
        predicted_price = model.predict(transformed_input)[0]
        price_fcfa = predicted_price * 4.5

        st.success("✅ Prediction Successful!")
        st.markdown("### 📊 Results")
        # col1, col2 = st.columns(2)
        # col1.metric("🇰🇪 Price (KES)", f"{predicted_price:,.2f} KES")
        # col2.metric("🇨🇮 Equivalent (FCFA)", f"{price_fcfa:,.2f} FCFA")
        # style_metric_cards()

        col1, col2 = st.columns(2)

        col1.metric(
        label="🇰🇪 Prix prédit (KES)",
        value=f"{predicted_price:,.2f} KES",
        delta="Estimation"
        )

        col2.metric(
        label="🇨🇮 Équivalent (FCFA)",
        value=f"{price_fcfa:,.2f} FCFA",
        delta="~x4.5"
        )
    except Exception as e:
     st.error(f"Prediction error: {str(e)}")

