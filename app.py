import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd
import numpy as np

# 1. Load Assets with Caching
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('Best_model.keras') 
    
    with open('state_emcoder.pkl', 'rb') as label:
        state_encoder = pickle.load(label)
        
    with open('Preprocess.pkl', 'rb') as preproo:
        preprocessor = pickle.load(preproo)
        
    return model, state_encoder, preprocessor

model, state_encoder, preprocessor = load_assets()

# 2. Header Section
st.title('🌊 Statewise Flood Risk Detection')
st.markdown('Enter the geographic and meteorological details below to predict flood risk.')
st.divider()

# 3. Input Columns
col1, col2, col3 = st.columns(3)

with col1:
    st.header('Geography')
    state_name = st.selectbox('State Name', state_encoder.classes_)
    elevation = st.number_input('Elevation (m)', value=100.0)
    
    # Updated exactly from your dataset
    land_cover = st.selectbox('Land Cover', ['Water Body', 'Forest', 'Agricultural', 'Desert', 'Urban']) 
    soil_type = st.selectbox('Soil Type', ['Clay', 'Peat', 'Loam', 'Sandy', 'Silt']) 

with col2:
    st.header('Weather & Water')
    rainfall = st.number_input('Rainfall (mm)', min_value=0.0, value=150.0)
    temperature = st.number_input('Temperature (°C)', value=25.0)
    humidity = st.slider('Humidity (%)', min_value=0, max_value=100, value=60)
    river_discharge = st.number_input('River Discharge (m³/s)', min_value=0.0, value=50.0)
    water_level = st.number_input('Water Level (m)', min_value=0.0, value=2000.0)

with col3:
    st.header('Human & History')
    population_density = st.number_input('Population Density (per sq km)', min_value=0.0, value=5000.0)
    
    # Updated to 0 or 1 based on your dataset snippet
    infrastructure = st.selectbox('Infrastructure', [0, 1], format_func=lambda x: "0 (Poor)" if x == 0 else "1 (Good)") 
    historical_floods = st.selectbox('Historical Floods', [0, 1], format_func=lambda x: "0 (No)" if x == 0 else "1 (Yes)")

st.divider()

# 4. Prediction Button & Output
if st.button('Predict Flood Risk', use_container_width=True, type="primary"):
    
    # Gather all inputs into a DataFrame that matches your training data exactly
    input_df = pd.DataFrame({
        'State Name': [state_name],
        'Rainfall (mm)': [rainfall],
        'Temperature (°C)': [temperature],
        'Humidity (%)': [humidity],
        'River Discharge (m³/s)': [river_discharge],
        'Water Level (m)': [water_level],
        'Elevation (m)': [elevation],
        'Land Cover': [land_cover],
        'Soil Type': [soil_type],
        'Population Density': [population_density],
        'Infrastructure': [infrastructure],
        'Historical Floods': [historical_floods]
    })

    try:
        # Step A: Label Encode the State Name
        input_df['State Name'] = state_encoder.transform(input_df['State Name'])
        
        # Step B: Apply your Preprocessor
        processed_data = preprocessor.transform(input_df)
        
        # Step C: Predict with the ANN
        prediction = model.predict(processed_data)
        
        probability = prediction[0][0] 
        
        # Display the result
        if probability > 0.5:
            st.error(f'🚨 **HIGH RISK:** Conditions indicate a flood is likely. (Probability: {probability*100:.1f}%)')
        else:
            st.success(f'✅ **LOW RISK:** No flood expected. (Probability: {probability*100:.1f}%)')
            
    except Exception as e:
        st.warning(f"An error occurred during prediction. Check your terminal for details. Error: {e}")