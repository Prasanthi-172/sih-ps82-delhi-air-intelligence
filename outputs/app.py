import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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

.card {
    background: white;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0px 4px 18px rgba(0,0,0,0.08);
    margin-bottom: 18px;
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

.risk-card {
    background: white;
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0px 4px 18px rgba(0,0,0,0.08);
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# FILE PATHS
# ============================================================

FORECAST_FILE = "outputs/forecast_72h.csv"
LOCATION_FORECAST_FILE = "outputs/location_forecast_72h.csv"
LOCATION_FILE = "data/delhi_locations.csv"
MAIN_DATA_FILE = "data/delhincr.csv"

METRICS_FILE = "outputs/05_metrics_by_horizon.csv"
SUMMARY_FILE = "outputs/05_summary_24_48_72.csv"
IMPORTANCE_FILE = "outputs/07_feature_importance.csv"


# ============================================================
# CHECK MAIN FORECAST
# ============================================================

if not os.path.exists(FORECAST_FILE):

    st.error("❌ forecast_72h.csv was not found.")

    st.info(
        "Please run your main 72-hour forecasting script first."
    )

    st.stop()


# ============================================================
# LOAD MAIN FORECAST
# ============================================================

forecast = pd.read_csv(FORECAST_FILE)

if "timestamp" in forecast.columns:

    forecast["timestamp"] = pd.to_datetime(
        forecast["timestamp"],
        errors="coerce"
    )

    forecast = forecast.dropna(
        subset=["timestamp"]
    )

    forecast = forecast.sort_values(
        "timestamp"
    ).reset_index(drop=True)


# ============================================================
# LOAD LOCATION DATA
# ============================================================

location_data = None

if os.path.exists(LOCATION_FILE):

    location_data = pd.read_csv(
        LOCATION_FILE
    )


# ============================================================
# LOAD LOCATION FORECAST
# ============================================================

location_forecast = None

if os.path.exists(LOCATION_FORECAST_FILE):

    location_forecast = pd.read_csv(
        LOCATION_FORECAST_FILE
    )

    if "timestamp" in location_forecast.columns:

        location_forecast["timestamp"] = pd.to_datetime(
            location_forecast["timestamp"],
            errors="coerce"
        )

    if "predicted_aqi" in location_forecast.columns:

        location_forecast["predicted_aqi"] = pd.to_numeric(
            location_forecast["predicted_aqi"],
            errors="coerce"
        )

    if "predicted_pm25" in location_forecast.columns:

        location_forecast["predicted_pm25"] = pd.to_numeric(
            location_forecast["predicted_pm25"],
            errors="coerce"
        )

    if "location" in location_forecast.columns:

        location_forecast = location_forecast.dropna(
            subset=["location"]
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
# LOAD MODEL METRICS
# ============================================================

metrics = None

if os.path.exists(METRICS_FILE):

    try:
        metrics = pd.read_csv(METRICS_FILE)
    except Exception:
        metrics = None


summary_metrics = None

if os.path.exists(SUMMARY_FILE):

    try:
        summary_metrics = pd.read_csv(SUMMARY_FILE)
    except Exception:
        summary_metrics = None


# ============================================================
# LOAD FEATURE IMPORTANCE
# ============================================================

feature_importance = None

if os.path.exists(IMPORTANCE_FILE):

    try:
        feature_importance = pd.read_csv(
            IMPORTANCE_FILE
        )
    except Exception:
        feature_importance = None


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

if "timestamp" in forecast.columns and len(forecast) > 0:

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

st.sidebar.title("🌫️ Delhi Air Intelligence")

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
        return "🟢 LOW RISK"

    elif aqi <= 200:
        return "🟡 MODERATE RISK"

    elif aqi <= 300:
        return "🟠 HIGH RISK"

    elif aqi <= 400:
        return "🔴 VERY HIGH RISK"

    else:
        return "⚫ SEVERE RISK"


# ============================================================
# RISK COLOR
# ============================================================

def get_risk_color(aqi):

    if aqi <= 100:
        return "green"

    elif aqi <= 200:
        return "yellow"

    elif aqi <= 300:
        return "orange"

    elif aqi <= 400:
        return "red"

    else:
        return "black"


# ============================================================
# LOCATION FORECAST SUMMARY
# ============================================================

def create_location_summary():

    if location_forecast is None:
        return None

    if "location" not in location_forecast.columns:
        return None

    if "predicted_aqi" not in location_forecast.columns:
        return None

    data = location_forecast.copy()

    data["predicted_aqi"] = pd.to_numeric(
        data["predicted_aqi"],
        errors="coerce"
    )

    if "predicted_pm25" not in data.columns:

        data["predicted_pm25"] = np.nan

    data["predicted_pm25"] = pd.to_numeric(
        data["predicted_pm25"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["location", "predicted_aqi"]
    )

    if len(data) == 0:
        return None

    summary = data.groupby(
        "location"
    ).agg(
        max_aqi=("predicted_aqi", "max"),
        average_aqi=("predicted_aqi", "mean"),
        max_pm25=("predicted_pm25", "max"),
        average_pm25=("predicted_pm25", "mean")
    ).reset_index()

    summary["aqi_category"] = summary[
        "max_aqi"
    ].apply(get_aqi_category)

    summary["risk"] = summary[
        "max_aqi"
    ].apply(get_risk_level)

    summary["risk_color"] = summary[
        "max_aqi"
    ].apply(get_risk_color)

    summary = summary.sort_values(
        "max_aqi",
        ascending=False
    ).reset_index(drop=True)

    summary["rank"] = range(
        1,
        len(summary) + 1
    )

    return summary


location_summary = create_location_summary()


# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.markdown(
        '<div class="section-title">🌤️ Current Air Quality</div>',
        unsafe_allow_html=True
    )

    current = forecast.iloc[0]

    pm25 = float(
        current["predicted_pm25"]
    )

    aqi = float(
        current["predicted_aqi"]
    )

    category = current.get(
        "aqi_category",
        get_aqi_category(aqi)
    )


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
    # MAIN FORECAST
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">📈 72-Hour PM2.5 Trend</div>',
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
    # FUTURE LOCATION RISK
    # --------------------------------------------------------

    if location_summary is not None:

        st.markdown(
            '<div class="section-title">'
            '📍 Next 72-Hour Location Risk'
            '</div>',
            unsafe_allow_html=True
        )

        top = location_summary.iloc[0]

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "🔴 Highest Risk Location",
                top["location"]
            )

        with c2:

            st.metric(
                "Maximum Predicted AQI",
                f"{top['max_aqi']:.0f}"
            )

        with c3:

            st.metric(
                "Risk",
                top["risk"]
            )


# ============================================================
# DELHI AQI MAP
# ============================================================

elif page == "🗺️ Delhi AQI Map":

    st.markdown(
        '<div class="section-title">'
        '🗺️ Delhi Location-wise 72-Hour AQI Risk Map'
        '</div>',
        unsafe_allow_html=True
    )


    if location_forecast is None:

        st.error(
            "❌ location_forecast_72h.csv was not found."
        )

        st.info(
            "Run 08_location_forecast.py first."
        )

        st.stop()


    if location_data is None:

        st.error(
            "❌ delhi_locations.csv was not found."
        )

        st.stop()


    if location_summary is None:

        st.error(
            "❌ Location forecast data is incomplete."
        )

        st.stop()


    summary = location_summary.copy()


    # --------------------------------------------------------
    # GET COORDINATES
    # --------------------------------------------------------

    coordinates = location_data[
        [
            "location",
            "lat",
            "lon"
        ]
    ].drop_duplicates(
        subset=["location"]
    )


    coordinates["lat"] = pd.to_numeric(
        coordinates["lat"],
        errors="coerce"
    )

    coordinates["lon"] = pd.to_numeric(
        coordinates["lon"],
        errors="coerce"
    )


    map_data = summary.merge(
        coordinates,
        on="location",
        how="left"
    )


    map_data = map_data.dropna(
        subset=["lat", "lon"]
    )


    # --------------------------------------------------------
    # LOCATION FILTER
    # --------------------------------------------------------

    selected_location = st.selectbox(
        "📍 Select Location",
        ["All Locations"] +
        sorted(
            map_data["location"].unique()
        )
    )


    if selected_location != "All Locations":

        selected_map = map_data[
            map_data["location"] ==
            selected_location
        ].copy()

    else:

        selected_map = map_data.copy()


    # --------------------------------------------------------
    # MAP
    # --------------------------------------------------------

    if len(selected_map) > 0:

        fig_map = px.scatter_map(
            selected_map,
            lat="lat",
            lon="lon",
            size="max_aqi",
            color="max_aqi",
            hover_name="location",

            hover_data={
                "max_aqi": ":.0f",
                "average_aqi": ":.0f",
                "max_pm25": ":.1f",
                "average_pm25": ":.1f",
                "aqi_category": True,
                "risk": True,
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
                "title": "Max AQI"
            }
        )


        st.plotly_chart(
            fig_map,
            use_container_width=True,
            key="future_location_map"
        )


    # --------------------------------------------------------
    # RISK LEGEND
    # --------------------------------------------------------

    st.markdown(
        "### 🚦 Future AQI Risk Levels"
    )

    r1, r2, r3, r4, r5 = st.columns(5)

    with r1:
        st.success("🟢 AQI ≤ 100\nLow Risk")

    with r2:
        st.warning("🟡 AQI 101–200\nModerate")

    with r3:
        st.warning("🟠 AQI 201–300\nHigh")

    with r4:
        st.error("🔴 AQI 301–400\nVery High")

    with r5:
        st.error("⚫ AQI > 400\nSevere")


    # --------------------------------------------------------
    # HIGHEST RISK LOCATION
    # --------------------------------------------------------

    st.markdown(
        "### 🚨 Highest-Risk Location in Next 72 Hours"
    )


    highest = summary.iloc[0]


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "📍 Location",
            highest["location"]
        )


    with c2:

        st.metric(
            "Maximum AQI",
            f"{highest['max_aqi']:.0f}"
        )


    with c3:

        st.metric(
            "Maximum PM2.5",
            f"{highest['max_pm25']:.1f}"
        )


    with c4:

        st.metric(
            "Risk",
            highest["risk"]
        )


    # --------------------------------------------------------
    # LOCATION RANKING
    # --------------------------------------------------------

    st.markdown(
        "### 🏆 Location Risk Ranking"
    )


    ranking = summary[
        [
            "rank",
            "location",
            "max_aqi",
            "average_aqi",
            "max_pm25",
            "risk",
            "aqi_category"
        ]
    ].copy()


    ranking.columns = [
        "Rank",
        "Location",
        "Max Forecast AQI",
        "Average AQI",
        "Max PM2.5",
        "Risk",
        "AQI Category"
    ]


    st.dataframe(
        ranking,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 72-HOUR FORECAST
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

    category_count = forecast[
        "aqi_category"
    ].value_counts().reset_index()

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
    # COMPLETE FORECAST
    # --------------------------------------------------------

    st.markdown(
        "### 🕐 Complete Forecast"
    )

    table = forecast.copy()


    rename_map = {

        "timestamp":
            "Date & Time",

        "forecast_hour":
            "Forecast Hour",

        "predicted_pm25":
            "Predicted PM2.5",

        "predicted_aqi":
            "Predicted AQI",

        "aqi_category":
            "AQI Category"
    }


    table = table.rename(
        columns=rename_map
    )


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
            "Main historical dataset was not found."
        )

        st.info(
            "The explanation section requires data/delhincr.csv."
        )

    else:

        # ----------------------------------------------------
        # GET LATEST AVAILABLE RECORD
        # ----------------------------------------------------

        if "timestamp" in main_data.columns:

            valid_data = main_data.dropna(
                subset=["timestamp"]
            )

        else:

            valid_data = main_data.copy()


        if len(valid_data) == 0:

            st.warning(
                "No valid historical records were found."
            )

        else:

            row = valid_data.iloc[-1]


            # ------------------------------------------------
            # READ IMPORTANT VARIABLES
            # ------------------------------------------------

            def get_value(column):

                if column in row.index:

                    try:
                        return float(row[column])
                    except Exception:
                        return np.nan

                return np.nan


            pm25_value = get_value(
                "pm2_5_ugm3"
            )

            wind_value = get_value(
                "wind_speed_10m_kmh"
            )

            boundary_value = get_value(
                "boundary_layer_height_m"
            )

            inversion_value = get_value(
                "inversion_strength_c"
            )

            fire_value = get_value(
                "upwind_stubble_fire_count"
            )

            humidity_value = get_value(
                "relative_humidity"
            )


            # ------------------------------------------------
            # CURRENT CONDITIONS
            # ------------------------------------------------

            st.markdown(
                "### 🌫️ Atmospheric Conditions"
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                if not np.isnan(pm25_value):

                    st.metric(
                        "PM2.5",
                        f"{pm25_value:.1f} µg/m³"
                    )

                else:

                    st.metric(
                        "PM2.5",
                        "N/A"
                    )


            with c2:

                if not np.isnan(wind_value):

                    st.metric(
                        "Wind Speed",
                        f"{wind_value:.1f} km/h"
                    )

                else:

                    st.metric(
                        "Wind Speed",
                        "N/A"
                    )


            with c3:

                if not np.isnan(humidity_value):

                    st.metric(
                        "Relative Humidity",
                        f"{humidity_value:.1f}%"
                    )

                else:

                    st.metric(
                        "Relative Humidity",
                        "N/A"
                    )


            # ------------------------------------------------
            # PHYSICAL EXPLANATION
            # ------------------------------------------------

            st.markdown(
                "### 🔬 Pollution Formation & Accumulation"
            )

            st.markdown(
                '<div class="explanation-box">',
                unsafe_allow_html=True
            )

            f1, f2, f3 = st.columns(3)

            with f1:

                st.markdown(
                    '<div class="flow-box">'
                    '🔥 Pollution Sources'
                    '<br><br>'
                    'Vehicles • Industry • Biomass Burning'
                    '</div>',
                    unsafe_allow_html=True
                )

            with f2:

                st.markdown(
                    '<div class="flow-box">'
                    '🌬️ Atmospheric Conditions'
                    '<br><br>'
                    'Wind • Humidity • PBL • Temperature'
                    '</div>',
                    unsafe_allow_html=True
                )

            with f3:

                st.markdown(
                    '<div class="flow-box">'
                    '🌫️ Pollution Accumulation'
                    '<br><br>'
                    'PM2.5 increases → AQI increases'
                    '</div>',
                    unsafe_allow_html=True
                )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


            # ------------------------------------------------
            # EXPLANATION RULES
            # ------------------------------------------------

            st.markdown(
                "### 🧠 Why Can Pollution Become High?"
            )


            explanations = []


            if not np.isnan(wind_value):

                if wind_value < 5:

                    explanations.append(
                        "🌬️ **Low wind speed:** "
                        "weak winds reduce the dispersion of pollutants."
                    )

                elif wind_value < 10:

                    explanations.append(
                        "🌬️ **Moderate wind:** "
                        "pollutant dispersion may still be limited."
                    )

                else:

                    explanations.append(
                        "🌬️ **Stronger wind:** "
                        "generally helps disperse pollutants."
                    )


            if not np.isnan(boundary_value):

                if boundary_value < 500:

                    explanations.append(
                        "⬇️ **Low planetary boundary layer height:** "
                        "pollutants are trapped closer to the surface."
                    )

                else:

                    explanations.append(
                        "⬆️ **Higher planetary boundary layer:** "
                        "provides more atmospheric volume for dispersion."
                    )


            if not np.isnan(inversion_value):

                if inversion_value > 2:

                    explanations.append(
                        "🌡️ **Atmospheric inversion:** "
                        "stable air can suppress vertical mixing."
                    )

                else:

                    explanations.append(
                        "🌡️ **Weak inversion:** "
                        "there is less evidence of strong temperature inversion."
                    )


            if not np.isnan(fire_value):

                if fire_value > 0:

                    explanations.append(
                        "🔥 **Upwind stubble-fire activity detected:** "
                        "regional biomass burning can contribute to pollution."
                    )

                else:

                    explanations.append(
                        "🔥 **No detected upwind stubble-fire activity "
                        "in the available record.**"
                    )


            if not np.isnan(humidity_value):

                if humidity_value > 80:

                    explanations.append(
                        "💧 **High humidity:** "
                        "humid conditions can contribute to particulate growth "
                        "and haze."
                    )


            if len(explanations) == 0:

                st.info(
                    "Not enough atmospheric variables are available "
                    "to generate a detailed explanation."
                )

            else:

                for reason in explanations:

                    st.warning(
                        reason
                    )


# ============================================================
# LOCATION ANALYSIS
# ============================================================

elif page == "📊 Location Analysis":

    st.markdown(
        '<div class="section-title">'
        '📊 Location-wise Future Air Quality Analysis'
        '</div>',
        unsafe_allow_html=True
    )


    if location_forecast is None:

        st.error(
            "❌ location_forecast_72h.csv was not found."
        )

        st.info(
            "Run 08_location_forecast.py first."
        )

        st.stop()


    if location_summary is None:

        st.error(
            "❌ Location forecast data is incomplete."
        )

        st.stop()


    # --------------------------------------------------------
    # SELECT LOCATION
    # --------------------------------------------------------

    selected_location = st.selectbox(
        "📍 Choose a Delhi NCR Location",
        sorted(
            location_summary["location"].unique()
        )
    )


    selected_data = location_forecast[
        location_forecast["location"] ==
        selected_location
    ].copy()


    selected_data = selected_data.sort_values(
        "timestamp"
    )


    selected_summary = location_summary[
        location_summary["location"] ==
        selected_location
    ].iloc[0]


    # --------------------------------------------------------
    # LOCATION METRICS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "📍 Location",
            selected_location
        )


    with c2:

        st.metric(
            "Maximum Forecast AQI",
            f"{selected_summary['max_aqi']:.0f}"
        )


    with c3:

        st.metric(
            "Average AQI",
            f"{selected_summary['average_aqi']:.0f}"
        )


    with c4:

        st.metric(
            "Risk",
            selected_summary["risk"]
        )


    # --------------------------------------------------------
    # PM2.5 METRICS
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            "Maximum PM2.5",
            f"{selected_summary['max_pm25']:.1f} µg/m³"
        )


    with c2:

        st.metric(
            "Average PM2.5",
            f"{selected_summary['average_pm25']:.1f} µg/m³"
        )


    with c3:

        st.metric(
            "AQI Category",
            selected_summary["aqi_category"]
        )


    # --------------------------------------------------------
    # PEAK TIME
    # --------------------------------------------------------

    peak_row = selected_data.loc[
        selected_data["predicted_aqi"].idxmax()
    ]


    st.markdown(
        "### 🚨 Peak Pollution Period"
    )


    p1, p2, p3 = st.columns(3)


    with p1:

        st.metric(
            "Peak AQI",
            f"{peak_row['predicted_aqi']:.0f}"
        )


    with p2:

        st.metric(
            "Peak PM2.5",
            f"{peak_row['predicted_pm25']:.1f}"
        )


    with p3:

        st.metric(
            "Peak Time",
            peak_row["timestamp"].strftime(
                "%d %b %Y, %I:%M %p"
            )
        )


    # --------------------------------------------------------
    # AQI CHART
    # --------------------------------------------------------

    fig_aqi = px.line(
        selected_data,
        x="timestamp",
        y="predicted_aqi",
        markers=True,
        title=(
            "📊 "
            + selected_location
            + " - Predicted AQI"
        )
    )

    fig_aqi.update_layout(
        xaxis_title="Date & Time",
        yaxis_title="Predicted AQI",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig_aqi,
        use_container_width=True
    )


    # --------------------------------------------------------
    # PM2.5 CHART
    # --------------------------------------------------------

    fig_pm = px.line(
        selected_data,
        x="timestamp",
        y="predicted_pm25",
        markers=True,
        title=(
            "🌫️ "
            + selected_location
            + " - Predicted PM2.5"
        )
    )

    fig_pm.update_layout(
        xaxis_title="Date & Time",
        yaxis_title="PM2.5 (µg/m³)",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig_pm,
        use_container_width=True
    )


    # --------------------------------------------------------
    # LOCATION FORECAST TABLE
    # --------------------------------------------------------

    st.markdown(
        "### 🕐 Location Forecast"
    )


    location_table = selected_data.copy()


    location_table = location_table.rename(
        columns={
            "timestamp": "Date & Time",
            "predicted_aqi": "Predicted AQI",
            "predicted_pm25": "Predicted PM2.5"
        }
    )


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
        '🤖 AI Model & System Information'
        '</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # SYSTEM PIPELINE
    # --------------------------------------------------------

    st.markdown(
        "### 🔄 Forecasting Pipeline"
    )


    st.markdown(
        """
        <div class="explanation-box">

        <div class="flow-box">
        📥 Historical Weather + Pollution Data
        </div>

        <div class="arrow">↓</div>

        <div class="flow-box">
        🧹 Data Validation & Cleaning
        </div>

        <div class="arrow">↓</div>

        <div class="flow-box">
        ⚙️ Feature Engineering
        </div>

        <div class="arrow">↓</div>

        <div class="flow-box">
        🤖 Machine Learning Models
        </div>

        <div class="arrow">↓</div>

        <div class="flow-box">
        🔮 72-Hour AQI Forecast
        </div>

        <div class="arrow">↓</div>

        <div class="flow-box">
        🗺️ Location-wise Risk Analysis
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    st.markdown(
        "### 🧠 Models Used"
    )


    model_table = pd.DataFrame({

        "Model": [
            "Persistence Baseline",
            "Random Forest",
            "HistGradientBoosting"
        ],

        "Purpose": [
            "Baseline comparison",
            "Non-linear ensemble prediction",
            "Gradient boosting prediction"
        ]
    })


    st.dataframe(
        model_table,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # ACTUAL METRICS
    # --------------------------------------------------------

    st.markdown(
        "### 📈 Model Performance"
    )


    if metrics is not None:

        st.dataframe(
            metrics,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Model performance file was not found."
        )


    if summary_metrics is not None:

        st.markdown(
            "### 📊 24 / 48 / 72 Hour Summary"
        )

        st.dataframe(
            summary_metrics,
            use_container_width=True,
            hide_index=True
        )


    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.markdown(
        "### 🎯 Feature Importance"
    )


    if feature_importance is not None:

        st.dataframe(
            feature_importance,
            use_container_width=True,
            hide_index=True
        )

        # Try to find feature and importance columns

        feature_column = None
        importance_column = None

        for col in feature_importance.columns:

            lower = col.lower()

            if "feature" in lower:

                feature_column = col

            if (
                "importance" in lower
                or "importance" in lower
            ):

                importance_column = col


        if (
            feature_column is not None
            and importance_column is not None
        ):

            plot_data = feature_importance.copy()

            plot_data[importance_column] = pd.to_numeric(
                plot_data[importance_column],
                errors="coerce"
            )

            plot_data = plot_data.dropna(
                subset=[importance_column]
            )

            plot_data = plot_data.sort_values(
                importance_column,
                ascending=False
            ).head(15)

            fig_imp = px.bar(
                plot_data,
                x=importance_column,
                y=feature_column,
                orientation="h",
                title="Top 15 Important Features"
            )

            st.plotly_chart(
                fig_imp,
                use_container_width=True
            )

    else:

        st.info(
            "Feature importance file was not found."
        )


    # --------------------------------------------------------
    # DATA ARCHITECTURE
    # --------------------------------------------------------

    st.markdown(
        "### 🗃️ Dataset Architecture"
    )


    data_info = pd.DataFrame({

        "Component": [
            "Historical Dataset",
            "Weather Variables",
            "Pollution Variables",
            "Fire Variables",
            "Location Data",
            "Forecast Horizon"
        ],

        "Description": [
            "Delhi NCR historical observations",
            "Temperature, humidity, wind, pressure and related variables",
            "PM2.5 and other pollutant measurements",
            "Regional fire / stubble-burning indicators when available",
            "Delhi NCR monitoring locations",
            "72 hours"
        ]
    })


    st.dataframe(
        data_info,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # LOCATION FORECAST STATUS
    # --------------------------------------------------------

    st.markdown(
        "### 📍 Location Forecast Status"
    )


    if location_forecast is not None:

        unique_locations = (
            location_forecast["location"]
            .nunique()
        )

        st.success(
            "✅ Location forecasting is available."
        )

        st.info(
            f"Forecast generated for "
            f"{unique_locations} locations."
        )

    else:

        st.warning(
            "⚠️ Location forecast file is not available."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <hr>

    <div style="text-align:center; color:#667085;">

    🌫️ <b>Delhi Air Intelligence</b><br>

    Air Pollution–Weather Coupled Forecasting System<br>

    Delhi NCR Focus • 72-Hour Forecasting

    </div>
    """,
    unsafe_allow_html=True
)