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
    "Linear_Regression_with_outliers": "Models/house_price_linearmodel.pkl",
    'Linear Regression_without_outliers': "Models/house_price_linearmodel_adjusted_to_outliers.pkl"
}


# Dynamic Model Loader
def load_selected_model(filename: str):
    model_path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{filename}' not found in root directory.")
    return joblib.load(model_path)


# Option Lists for Dropdowns
locations_list = sorted(['Sembakkam', 'Selaiyur', 'Mogappair', 'Ambattur', 'Pallavaram', 'Virugambakkam', 'Thirumazhisai', 'Moolakadai', 'Ottiyambakkam', 'Perungalathur', 'Gerugambakkam', 'Anna Nagar', 'Gopalapuram', 'Pammal', 'Porur', 'Navallur', 'Sholinganallur', 'Vanagaram', 'T Nagar', 'Guindy', 'Madipakkam', 'Perumbakkam', 'Velachery', 'Medavakkam', 'Gowrivakkam', 'Chromepet', 'Thalambur', 'Thandalam', 'Ayanambakkam', 'Kanathur Reddikuppam', 'Adyar', 'Kolapakkam', 'Thaiyur', 'Thoraipakkam OMR', 'Poonamallee', 'Padur', 'East Tambaram', 'Tiruvottiyur', 'Iyappanthangal', 'Padappai', 'Mugalivakkam', 'Kelambakkam', 'Thirumullaivoyal', 'Madhavaram', 'Kundrathur', 'Vengaivasal', 'Siruseri', 'Manapakkam', 'Karapakkam', 'Sithalapakkam', 'Madambakkam', 'Perungudi', 'Vadapalani', 'Alwarpet', 'Thiruvidandhai', 'Kotturpuram', 'Velappanchavadi', 'Kilpauk', 'Besant Nagar', 'Vellakkal', 'Avadi', 'Kolathur', 'Thiruvanmiyur', 'Periyar Nagar', 'Nanmangalam', 'Anakaputhur', 'Urapakkam', 'Kovur', 'Nungambakkam', 'Maduravoyal', 'Guduvancheri', 'Koyambedu', 'Korattur', 'Iyyappanthangal', 'Madhavaram Milk Colony', 'Jeth Nagar', 'Mangadu', 'Singaperumal Koil', 'Madhanandapuram', 'Kandigai', 'Veppampattu', 'Saidapet', 'Oragadam Village Ambattur Talu', 'Pattabiram', 'Mahabalipuram', 'Ayapakkam', 'Chitlapakkam', 'Puzhal', 'Washermanpet', 'Thiruverkadu', 'Pozhichalur', 'Mannur', 'West Mambalam', 'Kattupakkam', 'Maraimalai Nagar', 'Chetpet', 'Ramapuram', 'Alandur', 'Red Hills', 'K K Nagar', 'Purasawalkam', 'Ramavaram', 'Shenoy Nagar', 'Saligramam', 'Royapettah', 'Valasaravakkam', 'Semmancheri', 'Ottiyabakkam', 'Nanganallur', 'Pallikaranai', 'Muttukadu', 'Tharapakkam', 'Egmore', 'Anna Nagar East', 'Ayanavaram', 'Raja Annamalai Puram', 'Aminjikarai', 'Purasaiwakkam', 'Periyapanicheri', 'Kattankulathur', 'Kotivakkam', 'Kumananchavadi', 'Perumalpattu', 'Kodambakkam', 'Sevvapet', 'tambaram west', 'Kovilambakkam', 'Rajakilpakkam', 'Villivakkam', 'Chengalpattu', 'Cholambedu', 'Nesapakkam', 'Padapai', 'Neelankarai', 'Vadanemmeli', 'Kumaran Nagar', 'Ashok Nagar', 'Adambakam', 'Perambur', 'Keelkattalai', 'Kodungaiyur West', 'Pudupakkam', 'Padi', 'Mogappair East', 'Elandanur', 'Pazavanthangal', 'Thiruporur', 'Moolacheri', 'Kolapakkam Vandalur', 'Veppampatttu', 'Manali', 'Jamalia', 'Kottivakkam', 'Gokulapuram', 'Iyyapa Nagar', 'West Tambaram', 'Old Pallavaram', 'Sriperumbudur', 'Ponniammanmedu', 'Kovalam', 'CIT Nagar', 'Agaramthen', 'Vengambakkam', 'Mambakkam', 'Nallambakkam', 'Peerakankaranai', 'Rathinamangalam', 'Nandambakkam', 'Jafferkhanpet', 'Arumbakkam', 'Kuthambakkam', 'Thirumalpur', 'Mahindra World City', 'Oragadam Industrial Corridor', 'Ragavendra Colony', 'Ambattur INDUSTRIAL ESTATE', 'Mannivakkam', 'Kil Ayanambakkam'])
locations_list.insert(0,"None")
builders_list = sorted(['MP Developers', 'DAC Promoters', 'Casagrand Builder Private Limited', 'Dugar Housing Builders', 'Radiance Realty Developers India Ltd', 'Traventure Homes Pvt Ltd', 'Urbanrise', 'Navin Housing Properties P LTD', 'Jones foundation private limited', 'Isha Homes', 'Kochar Homes Pvt Ltd', 'Pushkar Properties Pvt Ltd', 'Asset Tree Homes', 'Urban Tree Infrastructures', 'Olympia Group', 'Vijay Raja Homes Private Limited', 'Kamalam Builder Pvt Ltd', 'Appaswamy Real Estate', 'VNR Homes', 'PS Srijan Developers', 'Lifestyle Housing', 'Puravankara Limited', 'Jones Foundations', 'Plaza Group', 'Urbando Housing LLP', 'EK Realtors', 'The Nest Builder', 'Doshi Housing', 'Grandstyle constructions', 'vinoth builders', 'GTK Foundations', 'Baashyaam Group', 'chris', 'Krishna Constructions', 'AKS Housing Dedvelopment Pvt Ltd', 'Mayances Construction and Engineering Services', 'Ramaniyam Real Estate Builders', 'India Builders Limited', 'Ram', 'Shatapatri Estates Pvt Ltd', 'Shri Raman Developers', 'Sri Hari Developers', 'Radiance Realty Developers', 'Pacifica Companies', 'BSCPL Infrastructure Ltd', 'Khurinji Homes', 'Saradeuz Realty Constructions', 'VGK Builders Pvt Ltd', 'Amarprakash Developers Pvt Ltd', 'Hansa Estates', 'Prince Foundations Ltd', 'Budget Housing And Properties', 'SP Homes Pvt Ltd', 'Bharathi Construction', 'Karuppaswamy Builders', 'Merlin Group', 'Advaita Homes', 'Swamaan Developers', 'Poojaa Foundation', 'Prestige Estates Projects Ltd', 'Global Homes', 'seller', 'Pon Mariappan', 'HM Homes', 'INTERFACE PROPERTIES', 'R Venkatesan', 'viswaraj', 'Shiva', 'S Suresh Kumar', 'MEHTA REAL ESTATE CHENNAI LLP', 'Balamurugan', 'Alliance Group', 'Royal Square', 'Proparena', 'Propsource Realty Private Limited', 'Sandhya', 'Murali', 'Tellus Foundation', 'Jayakanthan', 'Individual Agent', 'Naveen', 'Shanmugam Property', 'Value reality', 'Swaminathan', 'smartassetsindia', 'Sri Vinayaga Real Estate', 'Nagaraj', 'Karthick', 'Prasanna', 'Vinay Asrani', 'Pragyansh', '24K Realtors', 'Kaushik associates', 'BricksBurg', 'Dinesh', 'AKS REALTY SERVICES', 'Kkk Landmark', 'SS Square Property Developers', 'Prabha Homes', 'JA Associates', 'Bala', 'Sarashwathi Construction', 'Selvakumar', 'Right Angle Properties', 'Saravan gk', 'Evrostos Properties', 'Balasubramani', 'Vishnu Foundation Ltd', 'Shree sakthivel realestate', 'Yadhav constructions real estates', 'SS PROPERTIES', 'Prop Mart Technologies', 'Info Rich', 'JD Properties', 'Mohan', 'HomeFirst', 'Elite nisha', 'THAMINA HOMES', 'MC Foundation', 'ARB HOMES', 'South Zone Realty Consulting Pvt Ltd', 'Luxclusive Homes', 'Vishal D', 'mohammed', 'Dee Star Properties', 'Chennai Gated Community', 'Balaji', 'MrPincode', 'GJ ESTATES', 'Venkatesh', 'DJ Properties', 'Dhivagaran', 'REALTY INDIA', 'MAXWORTH PROPERTIES', 'Velan Housing Properties'])
builders_list.insert(0, "None")

# Layout Columns
col_inputs, col_results = st.columns([2, 1], gap="large")

with col_inputs:
    st.subheader("Property Attributes")

    # Input Grid
    c1, c2 = st.columns(2)

    with c1:
        area = st.number_input("Built-up Area (sq. ft.)", min_value=0, max_value=10000, value=0, step=50)
        bhk = st.selectbox("BHK (Bedrooms)", options=[0,1, 2, 3, 4, 5, 6], index=0)
        bathroom = st.selectbox("Bathrooms", options=[0,1, 2, 3, 4, 5], index=0)
        location = st.selectbox("Location", options=locations_list, index=0)

    with c2:
        status = st.selectbox("Construction Status", options=["None","ready", "under_construction"], index=0)
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
                        "status": [status],
                        "location": [location],
                        "builder": [builder]
                    })

                    # Run prediction
                    price_lakh = model.predict(input_data)[0]
                    #st.write("Input Data:")
                    #st.dataframe(input_data)
                    #st.write("Raw Prediction:", price_lakh)
                    #print(f"[{selected_model_name}] Output: {price_lakh}")

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