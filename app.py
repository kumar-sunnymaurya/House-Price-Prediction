import streamlit as st
import numpy as np
import pickle
import joblib

# Load Model and Scaler
model = pickle.load(open("house-price-predction.pkl", "rb"))
scaler = joblib.load("scaler.pkl")

# Page Configuration
st.set_page_config(
    page_title=" House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.title-box{
    background: linear-gradient(90deg,#1f4e79,#4CAF50);
    padding:20px;
    border-radius:15px;
    text-align:center;
    color:white;
    margin-bottom:20px;
}

.stButton>button{
    width:100%;
    height:55px;
    border-radius:12px;
    font-size:20px;
    font-weight:bold;
}

.prediction-box{
    background-color:#d4edda;
    padding:20px;
    border-radius:10px;
    text-align:center;
    font-size:28px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="title-box">
<h1>🏠  House Price Prediction System</h1>
<p>Predict Property Prices Using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Layout
col1, col2 = st.columns(2)

with col1:

    area = st.number_input(
        "Area (sq ft)",
        min_value=500,
        value=1000
    )

    bedrooms = st.number_input(
        "Bedrooms",
        min_value=1,
        value=2
    )

    bathrooms = st.number_input(
        "Bathrooms",
        min_value=1,
        value=1
    )

    stories = st.number_input(
        "Stories",
        min_value=1,
        value=1
    )

    parking = st.number_input(
        "Parking Spaces",
        min_value=0,
        value=1
    )

with col2:

    mainroad = st.selectbox(
        "Main Road",
        ["Yes", "No"]
    )

    guestroom = st.selectbox(
        "Guest Room",
        ["Yes", "No"]
    )

    basement = st.selectbox(
        "Basement",
        ["Yes", "No"]
    )

    hotwaterheating = st.selectbox(
        "Hot Water Heating",
        ["Yes", "No"]
    )

    airconditioning = st.selectbox(
        "Air Conditioning",
        ["Yes", "No"]
    )

    prefarea = st.selectbox(
        "Preferred Area",
        ["Yes", "No"]
    )

    furnishingstatus = st.selectbox(
        "Furnishing Status",
        ["Furnished", "Semi-Furnished", "Unfurnished"]
    )

# Encoding

mainroad = 1 if mainroad == "Yes" else 0
guestroom = 1 if guestroom == "Yes" else 0
basement = 1 if basement == "Yes" else 0
hotwaterheating = 1 if hotwaterheating == "Yes" else 0
airconditioning = 1 if airconditioning == "Yes" else 0
prefarea = 1 if prefarea == "Yes" else 0

if furnishingstatus == "Furnished":
    furnishingstatus = 0
elif furnishingstatus == "Semi-Furnished":
    furnishingstatus = 1
else:
    furnishingstatus = 2

# Prediction Button

if st.button("🔍 Predict House Price"):

    input_data = np.array([[
        area,
        bedrooms,
        bathrooms,
        stories,
        mainroad,
        guestroom,
        basement,
        hotwaterheating,
        airconditioning,
        parking,
        prefarea,
        furnishingstatus
    ]])

    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    st.markdown(
        f"""
        <div class="prediction-box">
        Estimated Price<br>
        ₹ {prediction[0]:,.2f}
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")
st.caption("Developed using Streamlit, Scikit-Learn and Machine Learning")