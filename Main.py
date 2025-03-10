import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    return df

df = load_data()

# Load pre-trained model
@st.cache_resource
def load_model():
    model = joblib.load("house_price_AR_model.pkl")
    return model

model = load_model()

# Sidebar options
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Data Visualization", "Price Prediction"])

# Overview Page
if page == "Overview":
    st.title("House Prices Data Analysis")
    st.write("### Dataset Overview")
    st.write(df.head())
    
    st.write("### Missing Values")
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
    st.write(missing_values)
    
# Data Visualization Page
elif page == "Data Visualization":
    st.title("Data Visualization")
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    feature = st.selectbox("Select a feature", numeric_features, index=numeric_features.index("SalePrice"))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[feature].dropna(), bins=30, kde=True, ax=ax)
    st.pyplot(fig)
    
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, ax=ax)
    st.pyplot(fig)
    
# Price Prediction Page
elif page == "Price Prediction":
    st.title("Predict House Prices")
    
    selected_features = ["GrLivArea", "OverallQual", "GarageCars", "TotalBsmtSF"]
    X = df[selected_features].fillna(0)
    y = df["SalePrice"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    y_pred = model.predict(X_test)
    
    st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):,.2f}")
    st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):,.2f}")
    
    st.write("### Predict Your Own House Price")
    input_data = {}
    for feature in selected_features:
        input_data[feature] = st.number_input(f"{feature}", value=float(X[feature].median()))
    
    input_df = pd.DataFrame([input_data])
    predicted_price = model.predict(input_df)[0]
    st.write(f"Predicted House Price: ${predicted_price:,.2f}")
