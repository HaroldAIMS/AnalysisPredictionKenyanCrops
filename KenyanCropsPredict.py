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

# Charger le modèle
model = joblib.load('best_model2.pkl')

# Titre de l'application
st.title("Prédiction des prix des cultures au Kenya")

st.markdown("Entrez les paramètres ci-dessous pour prédire le prix (en KES).")

# Formulaire utilisateur
commodity = st.selectbox("Type de culture (Commodity)", ["Cabbages", "Tomatoes", "Carrots", "Ripe Bananas", "Kales"])
unit = st.selectbox("Unité", ["Bag", "Lg Box", "Ext Bag"])
month = st.slider("Mois", 1, 12, 1)
year = st.selectbox("Année", [2016, 2017])
volume = st.number_input("Volume en kilogrammes (kg)", min_value=1, max_value=1000, value=100)

# Prédire le prix
if st.button("Prédire le prix"):
    # Création d'un DataFrame avec les entrées utilisateur
    input_data = pd.DataFrame({
        "Commodity_Type": [commodity],
        "Unit": [unit],
        "Month": [month],
        "Year": [year],
        "Volume_in_Kgs": [volume]
    })

    # Prédiction
    predicted_price = model.predict(input_data)[0]

    # Conversion en FCFA
    price_fcfa = predicted_price * 4.5

    # Affichage des résultats
    st.success(f"Prix prédit (en KES) : {predicted_price:,.2f} KES")
    st.info(f"Prix équivalent (en FCFA) : {price_fcfa:,.2f} FCFA")


