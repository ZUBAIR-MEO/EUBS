import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle
from io import StringIO

# Function to train the model
def train_model(data, target_variable):
    # Separate features and target variable
    features = data.drop(columns=[target_variable])
    target = data[target_variable]
    
    # Standardize the features using StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.3, random_state=42)

    # Train a RandomForest model for regression (House Price prediction)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    
    # Save the trained model to disk
    with open('house_price_model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    return model, mae

# Streamlit User Interface
st.title("House Price Prediction Model")

# File uploader for dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])

if uploaded_file is not None:
    # Load the uploaded CSV file into a DataFrame
    dataframe = pd.read_csv(uploaded_file)

    # Show the uploaded data
    st.write("Preview of the dataset:", dataframe.head())

    # Let the user select the target variable (column to predict)
    target_variable = st.selectbox("Select the target variable (column to predict)", dataframe.columns)
    
    # Train the model
    if st.button("Train Model"):
        # Train the model using the selected target variable
        model, mae = train_model(dataframe, target_variable)

        # Display the results
        st.write(f"Model trained successfully!")
        st.write(f"Mean Absolute Error (MAE) of the model: {mae:.2f}")

        # Show the feature importance from the trained model
        feature_importance = pd.DataFrame({
            'Feature': dataframe.drop(columns=[target_variable]).columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        st.write("Feature Importance:", feature_importance)

        # Option to download the trained model
        st.write("Download the trained model:")
        with open('house_price_model.pkl', 'rb') as f:
            st.download_button(
                label="Download Model",
                data=f,
                file_name="house_price_model.pkl",
                mime="application/octet-stream"
            )

# Prediction on new data
st.subheader("Predict on New Data")

# Option to upload new data for prediction
new_data = st.file_uploader("Upload new data for prediction (CSV file)", type=["csv"])

if new_data is not None:
    # Load the new data to make predictions
    new_dataframe = pd.read_csv(new_data)
    
    # Show the new data preview
    st.write("Preview of new data:", new_dataframe.head())

    # Make predictions if the model is available
    try:
        with open('house_price_model.pkl', 'rb') as f:
            model = pickle.load(f)
            
        # Select features based on the training data
        features = new_dataframe[model.feature_importances_].columns
        features_scaled = StandardScaler().fit_transform(new_dataframe[features])

        predictions = model.predict(features_scaled)
        
        # Display predictions
        st.write("Predictions on new data:", predictions)
    except FileNotFoundError:
        st.error("Model is not trained yet. Please train the model first.")
