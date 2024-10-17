import streamlit as st
import pickle
import numpy as np

# Load the trained RandomForest model
model = pickle.load(open('wine_quality_model.pkl', 'rb'))

st.title("Wine Quality Prediction App 🍷")
st.write("Enter the characteristics of the wine below:")

# Input fields for wine features
fixed_acidity = st.number_input('Fixed Acidity', min_value=0.0, max_value=20.0, step=0.1)
volatile_acidity = st.number_input('Volatile Acidity', min_value=0.0, max_value=2.0, step=0.01)
citric_acid = st.number_input('Citric Acid', min_value=0.0, max_value=1.0, step=0.01)
residual_sugar = st.number_input('Residual Sugar', min_value=0.0, max_value=20.0, step=0.1)
chlorides = st.number_input('Chlorides', min_value=0.0, max_value=1.0, step=0.001)
free_sulfur_dioxide = st.number_input('Free Sulfur Dioxide', min_value=0, max_value=100, step=1)
total_sulfur_dioxide = st.number_input('Total Sulfur Dioxide', min_value=0, max_value=300, step=1)
density = st.number_input('Density', min_value=0.990, max_value=1.010, step=0.0001)
pH = st.number_input('pH', min_value=2.0, max_value=4.0, step=0.01)
sulphates = st.number_input('Sulphates', min_value=0.0, max_value=2.0, step=0.01)
alcohol = st.number_input('Alcohol', min_value=0.0, max_value=15.0, step=0.1)

# Predict button
if st.button('Predict Quality'):
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                            chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                            density, pH, sulphates, alcohol]])
    
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success('Good quality wine! 🍾')
    else:
        st.error('Bad quality wine! 😞')
