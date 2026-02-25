import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="ML Project App", layout="wide")

st.title("🚀 Machine Learning Project App")
st.markdown("""
Welcome to the ML project dashboard. Use the sidebar to navigate through different stages of the model.
""")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Model Prediction"])

if page == "Home":
    st.header("Overview")
    st.write("This application serves as the interface for our Machine Learning model.")
    
elif page == "Data Exploration":
    st.header("📊 Data Exploration")
    st.write("Upload or view the processed data.")
    
elif page == "Model Prediction":
    st.header("🔮 Model Prediction")
    st.write("Input parameters to get a prediction from the trained model.")
