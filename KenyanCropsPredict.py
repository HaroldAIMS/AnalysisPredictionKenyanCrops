import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# App configuration
st.set_page_config(page_title="Kenyan Food Crop Prices Dashboard", layout="wide")
st.title("📊 Kenyan Food Crop Market Analysis & Prediction (2013–2015)")