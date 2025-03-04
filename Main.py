import streamlit as st
import joblib
import numpy as np
import pandas as pd
 
# Load the trained model
model = joblib.load('house_price_model.pkl')
 
# Define function for prediction
def predict_house_price(features):
    return model.predict([features])[0]
 
# Streamlit App
st.title("🏡 House Price Prediction App")
st.write("Enter house details below to predict the sale price.")
 
# Sidebar for user input
st.sidebar.header("Input House Features")
 
# Define user input fields
lot_area = st.sidebar.number_input("Lot Area", min_value=500, max_value=100000, step=100)
overall_quality = st.sidebar.selectbox("Overall Quality", options=list(range(1, 11)))
 
# Feature list (Modify based on your dataset)
features = [lot_area, overall_quality]
 
# Predict when the user presses the button
if st.sidebar.button("Predict Price"):
    prediction = predict_house_price(features)
    st.sidebar.write(f"🏠 **Estimated Sale Price:** ${prediction:,.2f}")
 
# Display dataset information
st.subheader("Dataset Overview")
train = pd.read_csv('train.csv')  # Load dataset (Ensure it's available in the same directory)
st.write(train.head())
 
# Plot SalePrice distribution
st.subheader("Sale Price Distribution")
import matplotlib.pyplot as plt
import seaborn as sns
 
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(train['SalePrice'], kde=True, bins=30, ax=ax)
ax.set_title("Distribution of Sale Price")
ax.set_xlabel("Sale Price")
ax.set_ylabel("Frequency")
st.pyplot(fig)
