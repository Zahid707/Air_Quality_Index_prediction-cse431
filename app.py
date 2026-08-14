import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime


# ==================================================
# Load trained model
# ==================================================

model = joblib.load("aqi_random_forest_compressed.pkl.gz")


# ==================================================
# Page configuration
# ==================================================

st.set_page_config(
    page_title="AQI Prediction System",
    page_icon="🌍",
    layout="centered"
)


# ==================================================
# Title
# ==================================================

st.title("🌍 Air Quality Prediction System")

st.write(
    "Enter environmental and weather information "
    "to predict the Air Quality Index (AQI)."
)


# ==================================================
# Environmental Measurements
# ==================================================

st.header("🌫️ Environmental Measurements")

pm25 = st.number_input(
    "PM2.5",
    min_value=0.0,
    value=44.2
)

pm10 = st.number_input(
    "PM10",
    min_value=0.0,
    value=93.2
)

o3 = st.number_input(
    "O3",
    min_value=0.0,
    value=20.6
)

no2 = st.number_input(
    "NO2",
    min_value=0.0,
    value=15.4
)

so2 = st.number_input(
    "SO2",
    min_value=0.0,
    value=28.2
)

co = st.number_input(
    "CO",
    min_value=0.0,
    value=1.05
)

temperature = st.number_input(
    "Temperature",
    value=21.0
)

humidity = st.number_input(
    "Humidity",
    min_value=0.0,
    max_value=100.0,
    value=71.0
)

wind_speed = st.number_input(
    "Wind Speed",
    min_value=0.0,
    value=8.4
)

pressure = st.number_input(
    "Pressure",
    min_value=900.0,
    max_value=1100.0,
    value=1011.7
)


# ==================================================
# Date and Time
# ==================================================

st.header("📅 Date and Time")

selected_date = st.date_input(
    "Select date",
    value=datetime.now().date()
)

selected_time = st.time_input(
    "Select time",
    value=datetime.now().time()
)

selected_datetime = datetime.combine(
    selected_date,
    selected_time
)


# ==================================================
# Feature Engineering
# ==================================================

hour = selected_datetime.hour
day_of_week = selected_datetime.weekday()
month = selected_datetime.month

hour_sin = np.sin(
    2 * np.pi * hour / 24
)

hour_cos = np.cos(
    2 * np.pi * hour / 24
)

month_sin = np.sin(
    2 * np.pi * month / 12
)

month_cos = np.cos(
    2 * np.pi * month / 12
)


# ==================================================
# Create model input
# ==================================================

input_data = pd.DataFrame({
    "PM2.5": [pm25],
    "PM10": [pm10],
    "O3": [o3],
    "NO2": [no2],
    "SO2": [so2],
    "CO": [co],
    "Temperature": [temperature],
    "Humidity": [humidity],
    "Wind_Speed": [wind_speed],
    "Pressure": [pressure],
    "hour": [hour],
    "day_of_week": [day_of_week],
    "month": [month],
    "hour_sin": [hour_sin],
    "hour_cos": [hour_cos],
    "month_sin": [month_sin],
    "month_cos": [month_cos]
})


# ==================================================
# AQI Reference Table
# ==================================================

st.header("📊 AQI Reference Guide")

aqi_reference = pd.DataFrame({
    "AQI": [
        "0–50",
        "51–100",
        "101–150",
        "151–200",
        "201–300",
        "301–500"
    ],
    "Air Quality": [
        "🟢 Good",
        "🟡 Moderate",
        "🟠 Unhealthy for Sensitive Groups",
        "🔴 Unhealthy",
        "🟣 Very Unhealthy",
        "🟤 Hazardous"
    ],
    "Meaning": [
        "Little or no health risk",
        "Acceptable; some sensitive people may be affected",
        "Sensitive groups may experience effects",
        "Everyone may begin experiencing health effects",
        "Health alert",
        "Serious health effects likely"
    ]
})

st.table(aqi_reference)


# ==================================================
# AQI Category Function
# ==================================================

def get_aqi_category(aqi):

    if aqi <= 50:
        return (
            "🟢 Good",
            "Little or no health risk",
            "#2e7d32",
            "white"
        )

    elif aqi <= 100:
        return (
            "🟡 Moderate",
            "Acceptable; some sensitive people may be affected",
            "#f9a825",
            "black"
        )

    elif aqi <= 150:
        return (
            "🟠 Unhealthy for Sensitive Groups",
            "Sensitive groups may experience effects",
            "#ef6c00",
            "white"
        )

    elif aqi <= 200:
        return (
            "🔴 Unhealthy",
            "Everyone may begin experiencing health effects",
            "#c62828",
            "white"
        )

    elif aqi <= 300:
        return (
            "🟣 Very Unhealthy",
            "Health alert",
            "#6a1b9a",
            "white"
        )

    elif aqi <= 500:
        return (
            "🟤 Hazardous",
            "Serious health effects likely",
            "#5d4037",
            "white"
        )

    else:
        return (
            "⚠️ Beyond AQI Scale",
            "AQI is above the displayed 0–500 scale",
            "#212121",
            "white"
        )


# ==================================================
# Prediction
# ==================================================

st.header("🔮 AQI Prediction")

if st.button(
    "Predict AQI",
    use_container_width=True
):

    prediction = float(
        model.predict(input_data)[0]
    )

    category, meaning, background, text_color = get_aqi_category(
        prediction
    )

    # Colored prediction box

    st.markdown(
        f"""
        <div style="
            background-color: {background};
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            margin-top: 15px;
            margin-bottom: 20px;
            color: {text_color};
        ">

            <div style="
                font-size: 20px;
                font-weight: bold;
            ">
                PREDICTED AIR QUALITY INDEX
            </div>

            <div style="
                font-size: 52px;
                font-weight: bold;
                margin: 10px 0;
            ">
                {prediction:.2f}
            </div>

            <div style="
                font-size: 25px;
                font-weight: bold;
            ">
                {category}
            </div>

            <div style="
                font-size: 17px;
                margin-top: 10px;
            ">
                {meaning}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # Optional model input display

    with st.expander("🔍 View Model Input Features"):

        st.dataframe(
            input_data,
            use_container_width=True
        )