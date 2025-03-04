import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load('house_price_model.pkl')

# Define function for prediction
def predict_house_price(features):
    return model.predict([features])[0]

# Streamlit app
st.title('House Price Prediction')
st.write('Enter house details for price prediction')

# Input fields for prediction
lot_area = st.number_input('Lot Area', min_value=0)
overall_quality = st.selectbox('Overall Quality', options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Get features from user
features = [lot_area, overall_quality]

# Predict when the user presses the button
if st.button('Predict'):
    prediction = predict_house_price(features)
    st.write(f'Predicted House Price: ${prediction:,.2f}')
