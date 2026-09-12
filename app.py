import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Chennai House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Header Section
st.title("🏠 Chennai House Price Prediction")
st.markdown("Estimate Chennai house prices with our interactive tool, tailored to your preferences.")
st.divider()


# Model Loader
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "house_price_model.pkl")
    return joblib.load(model_path)


try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Ensure `house_price_model.pkl` is saved in the project root directory.")
    st.stop()

# Option Lists for Dropdowns
locations_list = sorted([
    'Ambattur', 'Ambattur INDUSTRIAL ESTATE', 'Kil Ayanambakkam', 'Kuthambakkam',
    'Mahindra World City', 'Mannivakkam', 'Mogappair', 'Moolakadai', 'Nandambakkam',
    'Oragadam Industrial Corridor', 'Ottiyambakkam', 'Pallavaram', 'Perungalathur',
    'Ragavendra Colony', 'Rathinamangalam', 'Selaiyur', 'Sembakkam', 'Thirumazhisai',
    'Thirumalpur', 'Virugambakkam'
])

builders_list = sorted([
    'Balaji', 'Casagrand Builder Private Limited', 'Chennai Gated Community',
    'DAC Promoters', 'Dee Star Properties', 'Dhivagaran', 'DJ Properties',
    'Dugar Housing Builders', 'GJ ESTATES', 'Isha Homes', 'Jones foundation private limited',
    'MAXWORTH PROPERTIES', 'MP Developers', 'MrPincode', 'Navin Housing Properties P LTD',
    'Radiance Realty Developers India Ltd', 'Traventure Homes Pvt Ltd', 'Urbanrise',
    'Velan Housing Properties', 'Venkatesh'
])

# Layout Columns (Preserving wide layout structure)
col_inputs, col_results = st.columns([2, 1], gap="large")

with col_inputs:
    st.subheader("Property Attributes")

    # Input Grid
    c1, c2 = st.columns(2)

    with c1:
        area = st.number_input("Built-up Area (sq. ft.)", min_value=300, max_value=10000, value=1200, step=50)
        bhk = st.selectbox("BHK (Bedrooms)", options=[1, 2, 3, 4, 5, 6], index=1)
        bathroom = st.selectbox("Bathrooms", options=[1, 2, 3, 4, 5], index=1)
        age = st.number_input("Property Age (Years)", min_value=0, max_value=100, value=5, step=1)

    with c2:
        status = st.selectbox("Construction Status", options=["ready", "under_construction"])
        location = st.selectbox("Location", options=locations_list, index=0)
        builder = st.selectbox("Builder Name", options=builders_list, index=0)

with col_results:
    st.subheader("Estimation")
    st.write("Click below to run model inference.")

    predict_btn = st.button("Predict Price", type="primary", use_container_width=True)

    if predict_btn:
        with st.spinner("Calculating market value..."):
            try:
                # Construct DataFrame matching model schema
                input_data = pd.DataFrame({
                    "area": [area],
                    "bhk": [bhk],
                    "bathroom": [bathroom],
                    "age": [age],
                    "status": [status],
                    "location": [location],
                    "builder": [builder]
                })

                # Run prediction
                prediction = model.predict(input_data)

                # Print raw prediction to terminal/logs for debugging
                print(f"Raw prediction: {prediction}")

                price_lakh = prediction[0]

                # Format price output
                if price_lakh < 1:
                    formatted_price = f"₹{price_lakh * 100000:,.0f}"
                elif price_lakh < 100:
                    formatted_price = f"₹{price_lakh:,.2f} Lakh"
                elif price_lakh < 100000:
                    formatted_price = f"₹{price_lakh / 100:,.2f} Crore"
                else:
                    formatted_price = f"₹{price_lakh / 100000:,.2f} Thousand Crore"

                # Render output directly in UI
                st.success(f"**Estimated Price:** {formatted_price}")
                st.metric(label="Predicted Value", value=formatted_price)

            except Exception as err:
                st.error(f"Prediction failed: {err}")
                st.exception(err)  # Prints full traceback on screen
# How to Run Locally in PyCharm Terminal:
# streamlit run app.py