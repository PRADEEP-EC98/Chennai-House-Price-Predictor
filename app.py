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

# Model Directory & Registry
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add all your trained model filenames here
MODEL_FILES = {
    "Linear Regression": "house_price_linearmodel1.pkl",
    "HistGradientBoost": "house_price_HistGradientBoostingRegressor.pkl"

}


# Dynamic Model Loader
@st.cache_resource
def load_selected_model(filename: str):
    model_path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{filename}' not found in root directory.")
    return joblib.load(model_path)


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

# Layout Columns
col_inputs, col_results = st.columns([2, 1], gap="large")

with col_inputs:
    st.subheader("Property Attributes")

    # Input Grid
    c1, c2 = st.columns(2)

    with c1:
        area = st.number_input("Built-up Area (sq. ft.)", min_value=300, max_value=10000, value=1200, step=50)
        bhk = st.selectbox("BHK (Bedrooms)", options=[1, 2, 3, 4, 5, 6], index=1)
        bathroom = st.selectbox("Bathrooms", options=[1, 2, 3, 4, 5], index=1)
        location = st.selectbox("Location", options=locations_list, index=0)

    with c2:
        status = st.selectbox("Construction Status", options=["ready", "under_construction"])
        if status == "under_construction":
            age = 0
            st.info("Property Age is automatically set to 0 for Under Construction status.")
        else:
            age = st.number_input("Property Age (Years)", min_value=0, max_value=100, value=5, step=1)
        builder = st.selectbox("Builder Name", options=builders_list, index=0)

with col_results:
    st.subheader("Estimation & Model")

    # Model Selection Dropdown
    selected_model_name = st.selectbox("Select ML Model", options=list(MODEL_FILES.keys()))
    selected_filename = MODEL_FILES[selected_model_name]

    # Load Model safely
    model = None
    try:
        model = load_selected_model(selected_filename)
        st.caption(f"Loaded: `{selected_filename}`")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info(f"Ensure `{selected_filename}` is present in your project root.")

    st.write("Click below to run model inference.")
    predict_btn = st.button("Predict Price", type="primary", use_container_width=True)

    if predict_btn:
        if model is None:
            st.error("Cannot run inference because the model failed to load.")
        else:
            with st.spinner(f"Running inference with {selected_model_name}..."):
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
                    raw_pred = model.predict(input_data)[0]

                    # Check if model outputs log-transformed values (e.g., < 20)
                    # If your model was trained on log1p(price), invert it here
                    if raw_pred < 20:
                        price_lakh = np.expm1(raw_pred)
                    else:
                        price_lakh = raw_pred

                    print(f"[{selected_model_name}] Output: {price_lakh}")

                    # Format price output (assuming price_lakh is in Lakhs)
                    if price_lakh < 1:
                        formatted_price = f"₹{price_lakh * 100000:,.0f}"
                    elif price_lakh < 100:
                        formatted_price = f"₹{price_lakh:,.2f} Lakh"
                    elif price_lakh < 100000:
                        formatted_price = f"₹{price_lakh / 100:,.2f} Crore"
                    else:
                        formatted_price = f"₹{price_lakh / 100000:,.2f} Thousand Crore"

                    # Render output
                    st.success(f"**Estimated Price ({selected_model_name}):** {formatted_price}")

                except Exception as err:
                    st.error(f"Prediction failed with {selected_model_name}: {err}")
                    st.exception(err)