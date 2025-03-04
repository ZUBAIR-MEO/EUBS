import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
 
# Load the trained model safely
@st.cache_resource
def load_model():
    try:
        return joblib.load('house_price_model.pkl')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
 
# Load model
model = load_model()
 
# Streamlit App
st.title("🏡 House Price Prediction App")
st.write("Enter house details to predict the sale price.")
 
# Sidebar for user input
st.sidebar.header("🔹 Input House Features")
 
# Define user input fields
lot_area = st.sidebar.number_input("Lot Area", min_value=500, max_value=100000, step=100)
overall_quality = st.sidebar.selectbox("Overall Quality", options=list(range(1, 11)))
 
# Ensure model is loaded
if model:
    # Prepare features (Ensure feature names match training data)
    features = np.array([lot_area, overall_quality]).reshape(1, -1)
 
    # Predict when the user presses the button
    if st.sidebar.button("🔍 Predict Price"):
        try:
            prediction = model.predict(features)[0]
            st.sidebar.success(f"🏠 **Estimated Sale Price:** ${prediction:,.2f}")
        except Exception as e:
            st.sidebar.error(f"Prediction Error: {e}")
 
# Display dataset information
st.subheader("📊 Dataset Overview")
try:
    train = pd.read_csv('train.csv')  # Ensure this file is available
    st.write(train.head())
 
    # Plot SalePrice distribution
    import matplotlib.pyplot as plt
    import seaborn as sns
 
    st.subheader("📉 Sale Price Distribution")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(train['SalePrice'], kde=True, bins=30, ax=ax)
    ax.set_title("Distribution of Sale Price")
    ax.set_xlabel("Sale Price")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
except Exception as e:
    st.warning(f"Could not load dataset: {e}")
