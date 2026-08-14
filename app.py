import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AQI Prediction System",
    page_icon="🌍",
    layout="centered"
)


# ==================================================
# LOAD MODEL
# ==================================================

model = joblib.load("aqi_random_forest_compressed.pkl.gz")


# ==================================================
# TITLE
# ==================================================

st.title("🌍 Air Quality Prediction System")

st.write(
    "Enter environmental and weather information "
    "to predict the Air Quality Index (AQI)."
)


# ==================================================
# INPUTS
# ==================================================

st.header("🌫️ Environmental Measurements")

pm25 = st.number_input("PM2.5", min_value=0.0, value=44.2)
pm10 = st.number_input("PM10", min_value=0.0, value=93.2)
o3 = st.number_input("O3", min_value=0.0, value=20.6)
no2 = st.number_input("NO2", min_value=0.0, value=15.4)
so2 = st.number_input("SO2", min_value=0.0, value=28.2)
co = st.number_input("CO", min_value=0.0, value=1.05)

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
# DATE AND TIME
# ==================================================

st.header("📅 Date and Time")

selected_date = st.date_input(
    "Date",
    value=datetime.now().date()
)

selected_time = st.time_input(
    "Time",
    value=datetime.now().time()
)

selected_datetime = datetime.combine(
    selected_date,
    selected_time
)


# ==================================================
# FEATURE ENGINEERING
# ==================================================

hour = selected_datetime.hour
day_of_week = selected_datetime.weekday()
month = selected_datetime.month

hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)

month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)


# ==================================================
# MODEL INPUT
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
# AQI CATEGORY
# ==================================================

def get_aqi_info(aqi):

    if aqi <= 50:
        return "🟢 Good", "Little or no health risk", "green"

    elif aqi <= 100:
        return "🟡 Moderate", \
               "Acceptable; some sensitive people may be affected", \
               "orange"

    elif aqi <= 150:
        return "🟠 Unhealthy for Sensitive Groups", \
               "Sensitive groups may experience effects", \
               "darkorange"

    elif aqi <= 200:
        return "🔴 Unhealthy", \
               "Everyone may begin experiencing health effects", \
               "red"

    elif aqi <= 300:
        return "🟣 Very Unhealthy", \
               "Health alert", \
               "purple"

    else:
        return "🟤 Hazardous", \
               "Serious health effects likely", \
               "brown"


# ==================================================
# PREDICTION
# ==================================================

st.header("🔮 AQI Prediction")

if st.button(
    "Predict AQI",
    type="primary",
    use_container_width=True
):

    prediction = float(model.predict(input_data)[0])

    # Determine AQI category and colors

    if prediction <= 50:
        category = "🟢 Good"
        meaning = "Little or no health risk"
        background = "#2e7d32"
        text_color = "#ffffff"

    elif prediction <= 100:
        category = "🟡 Moderate"
        meaning = "Acceptable; some sensitive people may be affected"
        background = "#f9a825"
        text_color = "#000000"

    elif prediction <= 150:
        category = "🟠 Unhealthy for Sensitive Groups"
        meaning = "Sensitive groups may experience effects"
        background = "#ef6c00"
        text_color = "#ffffff"

    elif prediction <= 200:
        category = "🔴 Unhealthy"
        meaning = "Everyone may begin experiencing health effects"
        background = "#c62828"
        text_color = "#ffffff"

    elif prediction <= 300:
        category = "🟣 Very Unhealthy"
        meaning = "Health alert"
        background = "#6a1b9a"
        text_color = "#ffffff"

    else:
        category = "🟤 Hazardous"
        meaning = "Serious health effects likely"
        background = "#5d4037"
        text_color = "#ffffff"

    # Colored AQI prediction box

    st.markdown(
        f"""
        <div style="
            background-color: {background};
            color: {text_color};
            padding: 24px;
            border-radius: 12px;
            text-align: center;
            margin-top: 20px;
            margin-bottom: 25px;
        ">
            <div style="
                font-size: 18px;
                font-weight: 600;
            ">
                PREDICTED AIR QUALITY INDEX
            </div>

            <div style="
                font-size: 48px;
                font-weight: 700;
                margin: 8px 0;
            ">
                {prediction:.2f}
            </div>

            <div style="
                font-size: 22px;
                font-weight: 600;
            ">
                {category}
            </div>

            <div style="
                font-size: 16px;
                margin-top: 8px;
            ">
                {meaning}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ==================================================
# AQI REFERENCE TABLE
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

st.dataframe(
    aqi_reference,
    hide_index=True,
    use_container_width=True
)
