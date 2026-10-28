import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Product Quality Prediction System",
    page_icon="🏭",
    layout="centered"
)

st.title("🏭 Product Quality Prediction Dashboard")
st.markdown("""
Welcome to the AI-powered Product Quality Assessment Tool. 
Enter the manufacturing process parameters below to evaluate whether the final product meets quality standards.
""")

# --- 2. Load Trained Model ---
@st.cache_resource
def load_model():
    model_path = "product_quality_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error(f"Model file not found at path: {model_path}. Please check your file directory.")
        return None

model = load_model()

# --- 3. Input Controls & Form ---
st.header("📋 Manufacturing Parameters Input")

with st.form(key="prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        temp = st.number_input(
            "Processing Temperature (°C)", 
            min_value=50.0, max_value=400.0, value=170.0, step=1.0,
            help="Normal operating range: 100 - 250 °C"
        )
        pressure = st.number_input(
            "Pressure (bar)", 
            min_value=0.0, max_value=200.0, value=5.0, step=0.1,
            help="Recommended working pressure: 2.0 - 15.0 bar"
        )
        machine_speed = st.number_input(
            "Machine Speed (RPM)", 
            min_value=100.0, max_value=5000.0, value=1000.0, step=10.0,
            help="Standard operating speed: 500 - 2000 RPM"
        )
        
    with col2:
        thickness = st.number_input(
            "Material Thickness (mm)", 
            min_value=0.1, max_value=50.0, value=10.0, step=0.1,
            help="Thickness threshold: 5.0 - 20.0 mm"
        )
        moisture = st.number_input(
            "Material Moisture (%)", 
            min_value=0.0, max_value=100.0, value=0.5, step=0.01,
            help="Optimal moisture level: < 2.0 %"
        )
        cooling_time = st.number_input(
            "Cooling Time (sec)", 
            min_value=1.0, max_value=300.0, value=30.0, step=0.5,
            help="Cooling duration: 10.0 - 60.0 seconds"
        )

    submit_button = st.form_submit_button(label="🔍 Predict Quality Status")

# --- 4. Prediction Logic & Output ---
if submit_button:
    if model is not None:
        input_data = pd.DataFrame([{
            'processing_temperature': temp,
            'pressure_bar': pressure,
            'machine_speed_rpm': machine_speed,
            'material_thickness_mm': thickness,
            'material_moisture_pct': moisture,
            'cooling_time_sec': cooling_time
        }])
        
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        st.divider()
        st.subheader("📊 Prediction Results")
        
        if prediction == 1:
            st.success("✅ **Product Status: PASSED (Good Quality)**")
            st.metric(label="Model Confidence Score", value=f"{probabilities[1]*100:.2f}%")
        else:
            st.error("❌ **Product Status: DEFECTIVE (Quality Alert)**")
            st.metric(label="Model Confidence Score", value=f"{probabilities[0]*100:.2f}%")
            st.warning("⚠️ Warning: Parameter combinations fall outside optimal manufacturing conditions.")
