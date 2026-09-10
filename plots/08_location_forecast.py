import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Delhi Air Intelligence",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #eef7ff, #f8f9fc);
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    color: #172033;
    margin-bottom: 5px;
}

.sub-title {
    text-align: center;
    font-size: 17px;
    color: #667085;
    margin-bottom: 25px;
}

.section-title {
    font-size: 25px;
    font-weight: 700;
    color: #172033;
    margin-top: 20px;
    margin-bottom: 15px;
}

.explanation-box {
    background: white;
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0px 4px 18px rgba(0,0,0,0.08);
    margin-top: 15px;
}

.flow-box {
    background: #f8fafc;
    border-radius: 12px;
    padding: 15px;
    text-align: center;
    border: 1px solid #e4e7ec;
    margin: 7px;
}

.arrow {
    text-align: center;
    font-size: 25px;
    font-weight: bold;
}

.risk-high {
    color: #d92d20;
    font-weight: bold;
}

.risk-medium {
    color: #f79009;
    font-weight: bold;
}

.risk-low {
    color: #12b76a;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# FILE PATHS
# ============================================================

FORECAST_FILE = "outputs/forecast_72h.csv"

LOCATION_FILE = "data/delhi_locations.csv"

LOCATION_FORECAST_FILE = "outputs/location_forecast_72h.csv"

MAIN_DATA_FILE = "data/delhincr.csv"


# ============================================================
# CHECK OVERALL FORECAST
# ============================================================

if not os.path.exists(FORECAST_FILE):

    st.error("❌ forecast_72h.csv was not found.")

    st.info(
        "Please run your overall 72-hour prediction script first."
    )

    st.stop()


# ============================================================
# LOAD OVERALL FORECAST
# ============================================================

forecast = pd.read_csv(
    FORECAST_FILE
)

forecast["timestamp"] = pd.to_datetime(
    forecast["timestamp"],
    errors="coerce"
)

forecast["predicted_pm25"] = pd.to_numeric(
    forecast["predicted_pm25"],
    errors="coerce"
)

forecast["predicted_aqi"] = pd.to_numeric(
    forecast["predicted_aqi"],
    errors="coerce"
)

forecast = forecast.dropna(
    subset=["timestamp"]
)

forecast = forecast.sort_values(
    "timestamp"
).reset_index(drop=True)


# ============================================================
# LOAD LOCATION DATASET
# ============================================================

location_data = None

if os.path.exists(LOCATION_FILE):

    location_data = pd.read_csv(
        LOCATION_FILE
    )

    if "date_ist" in location_data.columns:

        location_data["date_ist"] = pd.to_datetime(
            location_data["date_ist"],
            errors="coerce"
        )


# ============================================================
# LOAD LOCATION 72-HOUR FORECAST
# ============================================================

location_forecast = None

if os.path.exists(LOCATION_FORECAST_FILE):

    location_forecast = pd.read_csv(
        LOCATION_FORECAST_FILE
    )

    location_forecast["timestamp"] = pd.to_datetime(
        location_forecast["timestamp"],
        errors="coerce"
    )

    location_forecast["predicted_pm25"] = pd.to_numeric(
        location_forecast["predicted_pm25"],
        errors="coerce"
    )

    location_forecast["predicted_aqi"] = pd.to_numeric(
        location_forecast["predicted_aqi"],
        errors="coerce"
    )

    location_forecast["lat"] = pd.to_numeric(
        location_forecast["lat"],
        errors="coerce"
    )

    location_forecast["lon"] = pd.to_numeric(
        location_forecast["lon"],
        errors="coerce"
    )

    location_forecast = location_forecast.dropna(
        subset=[
            "timestamp",
            "location",
            "predicted_aqi"
        ]
    )

else:

    st.warning(
        "⚠️ Location forecast file was not found. "
        "Run the location-wise 72-hour forecasting script."
    )


# ============================================================
# LOAD MAIN DATA
# ============================================================

main_data = None

if os.path.exists(MAIN_DATA_FILE):

    main_data = pd.read_csv(
        MAIN_DATA_FILE
    )

    if "timestamp" in main_data.columns:

        main_data["timestamp"] = pd.to_datetime(
            main_data["timestamp"],
            errors="coerce"
        )

        main_data = main_data.sort_values(
            "timestamp"
        )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🌫️ Delhi Air Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'AI-Powered Air Pollution Monitoring & 72-Hour Forecasting System'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# FORECAST START
# ============================================================

latest_time = forecast["timestamp"].min()


col1, col2, col3 = st.columns(3)


with col1:

    st.info(
        "📅 Forecast Start\n\n"
        + latest_time.strftime("%d %B %Y")
    )


with col2:

    st.info(
        "🕐 Forecast Time\n\n"
        + latest_time.strftime("%I:%M %p")
    )


with col3:

    st.info(
        "📍 Region\n\nDelhi NCR"
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🌫️ Delhi Air Intelligence"
)

st.sidebar.markdown(
    "### Dashboard Controls"
)

page = st.sidebar.radio(
    "Select View",
    [
        "🏠 Overview",
        "🗺️ Delhi AQI Map",
        "📈 72-Hour Forecast",
        "🔎 Why Is AQI High?",
        "📊 Location Analysis",
        "🤖 Model Information"
    ]
)


# ============================================================
# AQI CATEGORY
# ============================================================

def get_aqi_category(aqi):

    if aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Satisfactory"

    elif aqi <= 200:
        return "Moderate"

    elif aqi <= 300:
        return "Poor"

    elif aqi <= 400:
        return "Very Poor"

    else:
        return "Severe"


# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(aqi):

    if aqi <= 100:
        return "🟢 Low Risk"

    elif aqi <= 200:
        return "🟡 Moderate Risk"

    elif aqi <= 300:
        return "🟠 High Risk"

    elif aqi <= 400:
        return "🔴 Very High Risk"

    else:
        return "🔴 Severe Risk"


# ============================================================
# RISK SYMBOL
# ============================================================

def get_risk_symbol(aqi):

    if aqi <= 100:
        return "🟢"

    elif aqi <= 200:
        return "🟡"

    elif aqi <= 300:
        return "🟠"

    else:
        return "🔴"


# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.markdown(
        '<div class="section-title">'
        '🌤️ Current Air Quality'
        '</div>',
        unsafe_allow_html=True
    )


    current = forecast.iloc[0]


    pm25 = float(
        current["predicted_pm25"]
    )


    aqi = float(
        current["predicted_aqi"]
    )


    category = get_aqi_category(
        aqi
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "🌫️ PM2.5",
            f"{pm25:.1f} µg/m³"
        )


    with c2:

        st.metric(
            "📊 Predicted AQI",
            f"{aqi:.0f}"
        )


    with c3:

        st.metric(
            "⚠️ Category",
            category
        )


    with c4:

        st.metric(
            "⏱️ Forecast",
            "72 Hours"
        )


    # --------------------------------------------------------
    # MESSAGE
    # --------------------------------------------------------

    if aqi <= 50:

        st.success(
            "🟢 Air quality is GOOD."
        )

    elif aqi <= 100:

        st.info(
            "🟡 Air quality is SATISFACTORY."
        )

    elif aqi <= 200:

        st.warning(
            "🟠 Air quality is MODERATE."
        )

    elif aqi <= 300:

        st.warning(
            "🔴 Air quality is POOR."
        )

    elif aqi <= 400:

        st.error(
            "🟣 Air quality is VERY POOR."
        )

    else:

        st.error(
            "⚫ Air quality is SEVERE."
        )


    # --------------------------------------------------------
    # OVERALL PM2.5
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '📈 72-Hour PM2.5 Trend'
        '</div>',
        unsafe_allow_html=True
    )


    fig = px.line(
        forecast,
        x="timestamp",
        y="predicted_pm25",
        markers=True,
        title="Predicted PM2.5"
    )


    fig.update_layout(
        xaxis_title="Date & Time",
        yaxis_title="PM2.5 (µg/m³)",
        hovermode="x unified"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # --------------------------------------------------------
    # OVERALL AQI
    # --------------------------------------------------------

    st.markdown(
        "### 📊 72-Hour AQI Trend"
    )


    fig_aqi = px.line(
        forecast,
        x="timestamp",
        y="predicted_aqi",
        markers=True,
        title="Predicted AQI"
    )


    fig_aqi.update_layout(
        xaxis_title="Date & Time",
        yaxis_title="AQI",
        hovermode="x unified"
    )


    st.plotly_chart(
        fig_aqi,
        use_container_width=True
    )


# ============================================================
# DELHI AQI MAP
# ============================================================

elif page == "🗺️ Delhi AQI Map":

    st.markdown(
        '<div class="section-title">'
        '🗺️ Delhi NCR — Next 72-Hour AQI Risk Map'
        '</div>',
        unsafe_allow_html=True
    )


    if location_forecast is None:

        st.error(
            "Location forecast data is not available."
        )

        st.stop()


    # ========================================================
    # CALCULATE MAX AQI FOR EACH LOCATION
    # ========================================================

    location_summary = (
        location_forecast
        .groupby("location")
        .agg(
            max_aqi=("predicted_aqi", "max"),
            average_aqi=("predicted_aqi", "mean"),
            max_pm25=("predicted_pm25", "max"),
            lat=("lat", "first"),
            lon=("lon", "first")
        )
        .reset_index()
    )


    # --------------------------------------------------------
    # FIND TIME OF MAX AQI
    # --------------------------------------------------------

    max_times = []

    for location in location_summary["location"]:

        temp = location_forecast[
            location_forecast["location"]
            == location
        ]

        max_row = temp.loc[
            temp["predicted_aqi"].idxmax()
        ]

        max_times.append(
            max_row["timestamp"]
        )


    location_summary["max_aqi_time"] = max_times


    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

    location_summary["aqi_category"] = (
        location_summary["max_aqi"]
        .apply(get_aqi_category)
    )


    location_summary["risk"] = (
        location_summary["max_aqi"]
        .apply(get_risk_level)
    )


    location_summary["risk_symbol"] = (
        location_summary["max_aqi"]
        .apply(get_risk_symbol)
    )


    # --------------------------------------------------------
    # MAP
    # --------------------------------------------------------

    st.markdown(
        "### 📍 Predicted Maximum AQI in Next 72 Hours"
    )


    fig_map = px.scatter_map(
        location_summary,

        lat="lat",
        lon="lon",

        size="max_aqi",

        color="max_aqi",

        hover_name="location",

        hover_data={
            "max_aqi": ":.0f",
            "average_aqi": ":.0f",
            "max_pm25": ":.1f",
            "aqi_category": True,
            "risk": True,
            "max_aqi_time": True,
            "lat": False,
            "lon": False
        },

        color_continuous_scale=[
            [0.00, "green"],
            [0.20, "yellow"],
            [0.40, "orange"],
            [0.60, "red"],
            [0.80, "purple"],
            [1.00, "black"]
        ],

        zoom=9.5,

        center={
            "lat": 28.6139,
            "lon": 77.2090
        },

        height=650
    )


    fig_map.update_layout(

        map_style="open-street-map",

        margin={
            "r": 0,
            "t": 0,
            "l": 0,
            "b": 0
        },

        coloraxis_colorbar={
            "title": "Maximum<br>AQI"
        }
    )


    st.plotly_chart(
        fig_map,
        use_container_width=True
    )


    # ========================================================
    # HIGHEST RISK LOCATION
    # ========================================================

    highest_location = location_summary.loc[
        location_summary["max_aqi"].idxmax()
    ]


    st.markdown(
        "### 🚨 Highest Risk Location"
    )


    h1, h2, h3, h4 = st.columns(4)


    with h1:

        st.metric(
            "📍 Location",
            highest_location["location"]
        )


    with h2:

        st.metric(
            "📊 Maximum AQI",
            f"{highest_location['max_aqi']:.0f}"
        )


    with h3:

        st.metric(
            "🌫️ Maximum PM2.5",
            f"{highest_location['max_pm25']:.1f}"
        )


    with h4:

        st.metric(
            "⚠️ Risk",
            highest_location["risk"]
        )


    st.info(
        "The map shows the highest predicted AQI "
        "for each location during the next 72 hours."
    )


    # ========================================================
    # LOCATION RISK TABLE
    # ========================================================

    st.markdown(
        "### 🚦 Location-wise 72-Hour Risk"
    )


    risk_table = location_summary.copy()


    risk_table = risk_table.sort_values(
        "max_aqi",
        ascending=False
    )


    risk_table["Rank"] = range(
        1,
        len(risk_table) + 1
    )


    risk_table["Maximum AQI"] = (
        risk_table["max_aqi"]
        .round(0)
    )


    risk_table["Average AQI"] = (
        risk_table["average_aqi"]
        .round(0)
    )


    risk_table["Maximum PM2.5"] = (
        risk_table["max_pm25"]
        .round(1)
    )


    risk_table["Peak Time"] = (
        risk_table["max_aqi_time"]
        .dt.strftime("%d %b %I:%M %p")
    )


    display_table = risk_table[
        [
            "Rank",
            "risk_symbol",
            "location",
            "Maximum AQI",
            "Average AQI",
            "Maximum PM2.5",
            "aqi_category",
            "Peak Time"
        ]
    ].copy()


    display_table.columns = [
        "Rank",
        "Risk",
        "Location",
        "Maximum AQI",
        "Average AQI",
        "Maximum PM2.5",
        "Category",
        "Peak Time"
    ]


    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 72-HOUR OVERALL FORECAST
# ============================================================

elif page == "📈 72-Hour Forecast":

    st.markdown(
        '<div class="section-title">'
        '📈 AI 72-Hour Forecast'
        '</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # PM2.5
    # --------------------------------------------------------

    fig1 = px.line(
        forecast,
        x="timestamp",
        y="predicted_pm25",
        markers=True,
        title="🌫️ Predicted PM2.5"
    )


    fig1.update_layout(
        hovermode="x unified"
    )


    st.plotly_chart(
        fig1,
        use_container_width=True
    )


    # --------------------------------------------------------
    # AQI
    # --------------------------------------------------------

    fig2 = px.line(
        forecast,
        x="timestamp",
        y="predicted_aqi",
        markers=True,
        title="📊 Predicted AQI"
    )


    fig2.update_layout(
        hovermode="x unified"
    )


    st.plotly_chart(
        fig2,
        use_container_width=True
    )


    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

    st.markdown(
        "### ⚠️ Forecast AQI Categories"
    )


    category_count = (
        forecast["predicted_aqi"]
        .apply(get_aqi_category)
        .value_counts()
        .reset_index()
    )


    category_count.columns = [
        "Category",
        "Hours"
    ]


    fig3 = px.bar(
        category_count,
        x="Category",
        y="Hours",
        title="Number of Forecast Hours in Each AQI Category"
    )


    st.plotly_chart(
        fig3,
        use_container_width=True
    )


    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    st.markdown(
        "### 🕐 Complete 72-Hour Forecast"
    )


    table = forecast[
        [
            "timestamp",
            "forecast_hour",
            "predicted_pm25",
            "predicted_aqi"
        ]
    ].copy()


    table["aqi_category"] = (
        table["predicted_aqi"]
        .apply(get_aqi_category)
    )


    table.columns = [
        "Date & Time",
        "Forecast Hour",
        "Predicted PM2.5",
        "Predicted AQI",
        "AQI Category"
    ]


    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# WHY IS AQI HIGH
# ============================================================

elif page == "🔎 Why Is AQI High?":

    st.markdown(
        '<div class="section-title">'
        '🔎 Why Is AQI High?'
        '</div>',
        unsafe_allow_html=True
    )


    if main_data is None:

        st.warning(
            "Main dataset was not found."
        )

        st.stop()


    # --------------------------------------------------------
    # LATEST DATA
    # --------------------------------------------------------

    latest = main_data.iloc[-1]


    def get_value(column, default=0):

        if column in latest.index:

            try:

                value = float(
                    latest[column]
                )

                if np.isnan(value):
                    return default

                return value

            except:

                return default

        return default


    pm25_now = get_value(
        "pm2_5_ugm3"
    )

    wind = get_value(
        "wind_speed_10m_kmh"
    )

    pbl = get_value(
        "boundary_layer_height_m"
    )

    inversion = get_value(
        "inversion_strength_c"
    )

    fire = get_value(
        "upwind_stubble_fire_count"
    )

    humidity = get_value(
        "relative_humidity"
    )


    # --------------------------------------------------------
    # CONDITIONS
    # --------------------------------------------------------

    st.markdown(
        "### 🌤️ Atmospheric Conditions"
    )


    c1, c2, c3, c4, c5 = st.columns(5)


    with c1:

        st.metric(
            "PM2.5",
            f"{pm25_now:.1f}"
        )


    with c2:

        st.metric(
            "Wind",
            f"{wind:.1f} km/h"
        )


    with c3:

        st.metric(
            "PBL Height",
            f"{pbl:.0f} m"
        )


    with c4:

        st.metric(
            "Inversion",
            f"{inversion:.1f}°C"
        )


    with c5:

        st.metric(
            "Upwind Fires",
            f"{fire:.0f}"
        )


    # ========================================================
    # FLOW
    # ========================================================

    st.markdown(
        "### 🔄 Pollution Explanation Flow"
    )


    st.markdown("""
    <div class="explanation-box">

    <div class="flow-box">
    🌫️ <b>PM2.5 Level</b><br>
    Current particulate pollution is monitored
    </div>

    <div class="arrow">⬇️</div>

    <div class="flow-box">
    💨 <b>Wind Conditions</b><br>
    Low wind can reduce pollutant dispersion
    </div>

    <div class="arrow">⬇️</div>

    <div class="flow-box">
    ⬆️ <b>Atmospheric Boundary Layer</b><br>
    Low PBL can restrict vertical mixing
    </div>

    <div class="arrow">⬇️</div>

    <div class="flow-box">
    🌡️ <b>Temperature Inversion</b><br>
    Stable air can trap pollution near the surface
    </div>

    <div class="arrow">⬇️</div>

    <div class="flow-box">
    🔥 <b>Upwind Fire Activity</b><br>
    Regional fire activity can contribute smoke
    </div>

    <div class="arrow">⬇️</div>

    <div class="flow-box">
    📈 <b>Historical Pollution</b><br>
    Recent pollution patterns are considered
    </div>

    <div class="arrow">⬇️</div>

    <div class="flow-box">
    🤖 <b>Machine Learning Model</b><br>
    Weather + pollution + atmospheric + fire signals
    </div>

    <div class="arrow">⬇️</div>

    <div class="flow-box">
    🌫️ <b>Predicted PM2.5</b>
    </div>

    <div class="arrow">⬇️</div>

    <div class="flow-box">
    📊 <b>AQI Calculation</b>
    </div>

    <div class="arrow">⬇️</div>

    <div class="flow-box">
    ⚠️ <b>Predicted AQI</b>
    </div>

    </div>
    """, unsafe_allow_html=True)


    # ========================================================
    # INTERPRETATION
    # ========================================================

    st.markdown(
        "### 🧠 System Interpretation"
    )


    reasons = []


    if wind < 5:

        reasons.append(
            "💨 **Low wind speed:** pollutants may disperse more slowly."
        )


    if pbl > 0 and pbl < 500:

        reasons.append(
            "⬆️ **Low boundary-layer height:** vertical dispersion may be limited."
        )


    if inversion > 0:

        reasons.append(
            "🌡️ **Temperature inversion detected:** "
            "stable atmospheric conditions can trap pollution."
        )


    if fire > 0:

        reasons.append(
            "🔥 **Upwind fire activity detected:** "
            "regional smoke may contribute to pollution."
        )


    if humidity > 80:

        reasons.append(
            "💧 **High humidity:** humid conditions can contribute "
            "to haze and particulate growth."
        )


    if pm25_now > 100:

        reasons.append(
            "🌫️ **Elevated PM2.5:** particulate pollution is already high."
        )


    if len(reasons) == 0:

        st.success(
            "No major pollution-trapping indicators "
            "were detected by these simple rules."
        )

    else:

        for reason in reasons:

            st.warning(
                reason
            )


    st.caption(
        "These explanations describe atmospheric conditions associated "
        "with pollution. They should not be interpreted as proof that "
        "one individual factor caused the AQI value."
    )


# ============================================================
# LOCATION ANALYSIS
# ============================================================

elif page == "📊 Location Analysis":

    st.markdown(
        '<div class="section-title">'
        '📊 Location-wise 72-Hour Risk Analysis'
        '</div>',
        unsafe_allow_html=True
    )


    if location_forecast is None:

        st.error(
            "Location forecast file not found."
        )

        st.stop()


    # --------------------------------------------------------
    # LOCATIONS
    # --------------------------------------------------------

    locations = sorted(
        location_forecast["location"]
        .dropna()
        .unique()
    )


    selected = st.selectbox(
        "📍 Select Location",
        locations
    )


    selected_data = location_forecast[
        location_forecast["location"]
        == selected
    ].copy()


    selected_data = selected_data.sort_values(
        "timestamp"
    )


    # --------------------------------------------------------
    # MAX AQI
    # --------------------------------------------------------

    max_row = selected_data.loc[
        selected_data["predicted_aqi"].idxmax()
    ]


    max_aqi = float(
        max_row["predicted_aqi"]
    )


    max_pm25 = float(
        selected_data["predicted_pm25"].max()
    )


    average_aqi = float(
        selected_data["predicted_aqi"].mean()
    )


    peak_time = max_row["timestamp"]


    category = get_aqi_category(
        max_aqi
    )


    risk = get_risk_level(
        max_aqi
    )


    # --------------------------------------------------------
    # LOCATION RANK
    # --------------------------------------------------------

    location_max = (
        location_forecast
        .groupby("location")["predicted_aqi"]
        .max()
        .sort_values(
            ascending=False
        )
    )


    rank = (
        location_max.index
        .get_loc(selected)
        + 1
    )


    total_locations = len(
        location_max
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    c1, c2, c3, c4, c5 = st.columns(5)


    with c1:

        st.metric(
            "📊 Maximum AQI",
            f"{max_aqi:.0f}"
        )


    with c2:

        st.metric(
            "🌫️ Maximum PM2.5",
            f"{max_pm25:.1f}"
        )


    with c3:

        st.metric(
            "📈 Average AQI",
            f"{average_aqi:.0f}"
        )


    with c4:

        st.metric(
            "🏆 Risk Rank",
            f"{rank} / {total_locations}"
        )


    with c5:

        st.metric(
            "⚠️ Risk",
            risk
        )


    # --------------------------------------------------------
    # PEAK INFORMATION
    # --------------------------------------------------------

    st.info(
        "🚨 The predicted AQI for "
        + selected
        + " reaches a maximum of "
        + f"{max_aqi:.0f}"
        + " at "
        + peak_time.strftime("%d %B %Y, %I:%M %p")
        + "."
    )


    # --------------------------------------------------------
    # AQI CATEGORY
    # --------------------------------------------------------

    st.markdown(
        "### ⚠️ Predicted Risk Category"
    )


    if max_aqi <= 100:

        st.success(
            "🟢 Lower pollution risk — "
            + category
        )

    elif max_aqi <= 200:

        st.info(
            "🟡 Moderate pollution risk — "
            + category
        )

    elif max_aqi <= 300:

        st.warning(
            "🟠 High pollution risk — "
            + category
        )

    else:

        st.error(
            "🔴 Very high pollution risk — "
            + category
        )


    # ========================================================
    # PM2.5 FORECAST
    # ========================================================

    st.markdown(
        "### 🌫️ PM2.5 Forecast — " + selected
    )


    fig_pm = px.line(
        selected_data,
        x="timestamp",
        y="predicted_pm25",
        markers=True,
        title="Next 72 Hours PM2.5"
    )


    fig_pm.update_layout(
        xaxis_title="Date & Time",
        yaxis_title="Predicted PM2.5",
        hovermode="x unified"
    )


    st.plotly_chart(
        fig_pm,
        use_container_width=True
    )


    # ========================================================
    # AQI FORECAST
    # ========================================================

    st.markdown(
        "### 📊 AQI Forecast — " + selected
    )


    fig_location_aqi = px.line(
        selected_data,
        x="timestamp",
        y="predicted_aqi",
        markers=True,
        title="Next 72 Hours AQI"
    )


    fig_location_aqi.update_layout(
        xaxis_title="Date & Time",
        yaxis_title="Predicted AQI",
        hovermode="x unified"
    )


    st.plotly_chart(
        fig_location_aqi,
        use_container_width=True
    )


    # ========================================================
    # FORECAST TABLE
    # ========================================================

    st.markdown(
        "### 🕐 72-Hour Location Forecast"
    )


    location_table = selected_data[
        [
            "timestamp",
            "forecast_hour",
            "predicted_pm25",
            "predicted_aqi"
        ]
    ].copy()


    location_table["aqi_category"] = (
        location_table["predicted_aqi"]
        .apply(get_aqi_category)
    )


    location_table["risk"] = (
        location_table["predicted_aqi"]
        .apply(get_risk_level)
    )


    location_table.columns = [
        "Date & Time",
        "Forecast Hour",
        "Predicted PM2.5",
        "Predicted AQI",
        "Category",
        "Risk"
    ]


    st.dataframe(
        location_table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "🤖 Model Information":

    st.markdown(
        '<div class="section-title">'
        '🤖 AI Forecasting System'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown("""
    <div class="explanation-box">

    <h3>🧠 How the system works</h3>

    <p>
    The system combines historical pollution, weather,
    atmospheric and fire-related information to forecast
    PM2.5 and AQI.
    </p>

    <br>

    <b>Step 1 — Historical Data</b>

    <br>⬇️<br>

    Weather + Pollution + Fire Data

    <br>⬇️<br>

    <b>Step 2 — Feature Engineering</b>

    <br>⬇️<br>

    PM2.5 Lags + Rolling Averages +
    Atmospheric Conditions + Fire History +
    Time Features

    <br>⬇️<br>

    <b>Step 3 — Machine Learning</b>

    <br>⬇️<br>

    Random Forest + HistGradientBoosting

    <br>⬇️<br>

    <b>Step 4 — 72 Forecast Horizons</b>

    <br>⬇️<br>

    +1h → +2h → +3h → ... → +72h

    <br>⬇️<br>

    <b>Step 5 — Location-wise Forecasting</b>

    <br>⬇️<br>

    Anand Vihar + Connaught Place + Dwarka +
    IGI Airport + Okhla Phase III + Rohini

    <br>⬇️<br>

    <b>Step 6 — PM2.5 → AQI</b>

    <br>⬇️<br>

    <b>Delhi NCR Air Quality Risk Map</b>

    </div>
    """, unsafe_allow_html=True)


    # ========================================================
    # DATASET ARCHITECTURE
    # ========================================================

    st.markdown(
        "### 🗂️ Dataset Architecture"
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.info("""
        **Dataset 1 — Forecasting**

        `delhincr.csv`

        • Pollution
        • Weather
        • Fire activity
        • Atmospheric variables

        Used for overall forecasting.
        """)


    with c2:

        st.success("""
        **Dataset 2 — Locations**

        `delhi_locations.csv`

        • Locations
        • Latitude
        • Longitude
        • AQI
        • PM2.5
        • PM10
        • Weather

        Used for spatial monitoring.
        """)


    with c3:

        st.warning("""
        **Location Forecast**

        `location_forecast_72h.csv`

        • Location
        • Forecast hour
        • Predicted PM2.5
        • Predicted AQI
        • AQI category

        Used for 72-hour location risk.
        """)


    # ========================================================
    # LOCATION MODEL RESULTS
    # ========================================================

    if location_forecast is not None:

        st.markdown(
            "### 📈 Location Forecast Summary"
        )


        summary = (
            location_forecast
            .groupby("location")
            .agg(
                maximum_aqi=(
                    "predicted_aqi",
                    "max"
                ),

                average_aqi=(
                    "predicted_aqi",
                    "mean"
                ),

                maximum_pm25=(
                    "predicted_pm25",
                    "max"
                )
            )
            .reset_index()
        )


        summary = summary.sort_values(
            "maximum_aqi",
            ascending=False
        )


        summary["Category"] = (
            summary["maximum_aqi"]
            .apply(get_aqi_category)
        )


        summary["Risk"] = (
            summary["maximum_aqi"]
            .apply(get_risk_level)
        )


        summary.columns = [
            "Location",
            "Maximum AQI",
            "Average AQI",
            "Maximum PM2.5",
            "Category",
            "Risk"
        ]


        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")


st.markdown(
    """
    <div style="text-align:center;color:#667085;">

    🌫️ <b>Delhi Air Intelligence</b><br>

    Air Pollution–Weather Coupled Forecasting System<br>

    SIH Prototype

    </div>
    """,
    unsafe_allow_html=True
)