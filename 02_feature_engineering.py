"""
02_feature_engineering.py
--------------------------
PURPOSE:
Turn the raw hourly dataset into a set of FEATURES a machine learning model
can actually learn from.

WHY THIS MATTERS:
A model looking only at "the weather and pollution right now" cannot predict
72 hours ahead very well. Pollution builds up and decays over time, so we
give the model MEMORY by creating "lag" and "rolling average" features:
    - pm25_lag_6h  = what was PM2.5 six hours ago?
    - pm25_avg_24h = what has PM2.5 averaged over the last 24 hours?

All of these features only ever look BACKWARDS in time from the current row.
This is critical: if we accidentally used future information, our model
would look great in testing but fail completely in the real world
(this mistake is called "data leakage").

INPUT:  data/delhincr.csv               (raw, untouched)
OUTPUT: data/02_features.csv            (new file, engineered features)

Run this after 01_validate_data.py:
    python 02_feature_engineering.py
"""

"""
02_feature_engineering.py
--------------------------
PURPOSE:
Create useful historical, weather, fire, atmospheric, trend and interaction
features for PM2.5 forecasting.

IMPORTANT:
All historical features use only information available at or before the
prediction time. Future PM2.5 values are NOT used here.

INPUT:
    data/delhincr.csv

OUTPUT:
    data/02_features.csv
"""

import pandas as pd
import numpy as np
import os


RAW_FILE = "data/delhincr.csv"
OUTPUT_FILE = "data/02_features.csv"


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

# PM2.5 historical values
PM25_LAGS = [1, 2, 3, 6, 12, 18, 24, 36, 48, 72]

# Rolling PM2.5 history
PM25_ROLLING_WINDOWS = [3, 6, 12, 24, 48, 72, 168]


# Fire history
FIRE_LAGS = [1, 6, 12, 24]

FIRE_COLUMNS = [
    "fire_count",
    "punjab_fire_count",
    "haryana_fire_count",
    "uttar_pradesh_fire_count",
    "uttarakhand_fire_count",
    "delhi_ncr_fire_count",
    "upwind_stubble_fire_count",
]


# Atmospheric variables
ATMOSPHERIC_COLUMNS = [
    "boundary_layer_height_m",
    "wind_speed_10m_kmh",
    "inversion_strength_c",
    "is_inversion",
    "low_pbl_flag",
    "trapping_condition",
]

ATMOSPHERIC_LAGS = [1, 3, 6, 12, 24]


def main():

    print("Loading dataset...")

    if not os.path.exists(RAW_FILE):
        print("ERROR: Dataset not found:")
        print(RAW_FILE)
        return

    df = pd.read_csv(RAW_FILE)

    # ------------------------------------------------------------
    # BASIC PREPARATION
    # ------------------------------------------------------------

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values("timestamp").reset_index(drop=True)

    print("Creating features...")


    # ============================================================
    # 1. PM2.5 LAG FEATURES
    # ============================================================

    for lag in PM25_LAGS:

        df[f"pm25_lag_{lag}h"] = (
            df["pm2_5_ugm3"].shift(lag)
        )


    # ============================================================
    # 2. PM2.5 ROLLING AVERAGES
    # ============================================================

    for window in PM25_ROLLING_WINDOWS:

        df[f"pm25_avg_{window}h"] = (
            df["pm2_5_ugm3"]
            .shift(1)
            .rolling(
                window=window,
                min_periods=1
            )
            .mean()
        )


    # ============================================================
    # 3. PM2.5 EXPONENTIALLY WEIGHTED AVERAGES
    # ============================================================

    df["pm25_ewm_12h"] = (
        df["pm2_5_ugm3"]
        .shift(1)
        .ewm(
            span=12,
            min_periods=1
        )
        .mean()
    )

    df["pm25_ewm_24h"] = (
        df["pm2_5_ugm3"]
        .shift(1)
        .ewm(
            span=24,
            min_periods=1
        )
        .mean()
    )


    # ============================================================
    # 4. PM2.5 TREND / MOMENTUM
    # ============================================================

    # Change during previous 1 hour
    df["pm25_change_1h"] = (
        df["pm2_5_ugm3"].shift(1)
        -
        df["pm2_5_ugm3"].shift(2)
    )

    # Change during previous 3 hours
    df["pm25_diff_3h"] = (
        df["pm2_5_ugm3"].shift(1)
        -
        df["pm2_5_ugm3"].shift(4)
    )

    # Change during previous 6 hours
    df["pm25_diff_6h"] = (
        df["pm2_5_ugm3"].shift(1)
        -
        df["pm2_5_ugm3"].shift(7)
    )

    # Change during previous 12 hours
    df["pm25_change_12h"] = (
        df["pm2_5_ugm3"].shift(1)
        -
        df["pm2_5_ugm3"].shift(13)
    )

    # Change during previous 24 hours
    df["pm25_diff_24h"] = (
        df["pm2_5_ugm3"].shift(1)
        -
        df["pm2_5_ugm3"].shift(25)
    )

    df["pm25_change_24h"] = (
        df["pm2_5_ugm3"].shift(1)
        -
        df["pm2_5_ugm3"].shift(25)
    )


    # ============================================================
    # 5. PM2.5 ANOMALY FEATURES
    # ============================================================

    df["pm25_anomaly_vs_72h_avg"] = (
        df["pm2_5_ugm3"]
        -
        df["pm25_avg_72h"]
    )

    df["pm25_ratio_vs_72h_avg"] = (
        df["pm2_5_ugm3"]
        /
        df["pm25_avg_72h"].replace(0, np.nan)
    )


    # ============================================================
    # 6. FIRE HISTORY
    # ============================================================

    for col in FIRE_COLUMNS:

        for lag in FIRE_LAGS:

            df[f"{col}_last_{lag}h"] = (
                df[col]
                .shift(1)
                .rolling(
                    window=lag,
                    min_periods=1
                )
                .sum()
            )


    # ============================================================
    # 7. LONGER FIRE HISTORY
    # ============================================================

    for col in [
        "punjab_fire_count",
        "haryana_fire_count",
        "upwind_stubble_fire_count"
    ]:

        for lag in [48, 72]:

            df[f"{col}_last_{lag}h"] = (
                df[col]
                .shift(1)
                .rolling(
                    window=lag,
                    min_periods=1
                )
                .sum()
            )


    # ============================================================
    # 8. FIRE TREND
    # ============================================================

    df["punjab_fire_change_24h"] = (
        df["punjab_fire_count"].shift(1)
        -
        df["punjab_fire_count"].shift(25)
    )

    df["haryana_fire_change_24h"] = (
        df["haryana_fire_count"].shift(1)
        -
        df["haryana_fire_count"].shift(25)
    )

    df["upwind_fire_change_24h"] = (
        df["upwind_stubble_fire_count"].shift(1)
        -
        df["upwind_stubble_fire_count"].shift(25)
    )


    # ============================================================
    # 9. ATMOSPHERIC HISTORY
    # ============================================================

    for col in ATMOSPHERIC_COLUMNS:

        for lag in ATMOSPHERIC_LAGS:

            df[f"{col}_lag_{lag}h"] = (
                df[col].shift(lag)
            )


    # ============================================================
    # 10. WIND DIRECTION
    # ============================================================

    wind_rad = np.deg2rad(
        df["wind_direction_10m_deg"]
    )

    df["wind_direction_sin"] = np.sin(wind_rad)

    df["wind_direction_cos"] = np.cos(wind_rad)


    # ============================================================
    # 11. WIND COMPONENTS
    # ============================================================

    df["wind_u"] = (
        df["wind_speed_10m_kmh"]
        *
        np.sin(wind_rad)
    )

    df["wind_v"] = (
        df["wind_speed_10m_kmh"]
        *
        np.cos(wind_rad)
    )


    # ============================================================
    # 12. CALM WIND
    # ============================================================

    df["calm_wind_flag"] = (
        df["wind_speed_10m_kmh"] < 5
    ).astype(int)


    # ============================================================
    # 13. DEW POINT SPREAD
    # ============================================================

    df["dew_point_spread_c"] = (
        df["temperature_2m_c"]
        -
        df["dew_point_2m_c"]
    )


    # ============================================================
    # 14. PRESSURE TREND
    # ============================================================

    df["pressure_trend_24h"] = (
        df["pressure_msl_hpa"].shift(1)
        -
        df["pressure_msl_hpa"].shift(25)
    )


    # ============================================================
    # 15. PHYSICAL INTERACTION FEATURES
    # ============================================================

    # Fire + inversion
    df["fire_x_inversion"] = (
        df["upwind_stubble_fire_count_last_24h"]
        *
        df["inversion_strength_c"].clip(lower=0)
    )


    # Fire + weak wind
    df["fire_x_wind"] = (
        df["upwind_stubble_fire_count_last_24h"]
        /
        (df["wind_speed_10m_kmh"] + 1)
    )


    # PM2.5 + inversion
    df["pm25_x_inversion"] = (
        df["pm2_5_ugm3"]
        *
        df["inversion_strength_c"].clip(lower=0)
    )


    # PM2.5 + low PBL
    df["pm25_x_low_pbl"] = (
        df["pm2_5_ugm3"]
        *
        df["low_pbl_flag"]
    )


    # ============================================================
    # 16. TIME FEATURES
    # ============================================================

    df["day"] = (
        df["timestamp"].dt.day
    )

    df["day_of_year"] = (
        df["timestamp"].dt.dayofyear
    )


    # ============================================================
    # 17. CYCLICAL TIME FEATURES
    # ============================================================

    df["hour_sin"] = (
        np.sin(
            2 * np.pi * df["hour"] / 24
        )
    )

    df["hour_cos"] = (
        np.cos(
            2 * np.pi * df["hour"] / 24
        )
    )

    df["month_sin"] = (
        np.sin(
            2 * np.pi * df["month"] / 12
        )
    )

    df["month_cos"] = (
        np.cos(
            2 * np.pi * df["month"] / 12
        )
    )


    # ============================================================
    # 18. ONE-HOT ENCODING
    # ============================================================

    if "dominant_region" in df.columns:

        df = pd.get_dummies(
            df,
            columns=["dominant_region"],
            prefix="region"
        )


    # trapping condition
    if "trapping_condition" in df.columns:

        df["trapping_condition"] = (
            df["trapping_condition"]
            .astype(int)
        )


    # ============================================================
    # 19. CLEAN DATAFRAME
    # ============================================================

    # De-fragment dataframe
    df = df.copy()


    # ============================================================
    # 20. SAVE
    # ============================================================

    os.makedirs("data", exist_ok=True)

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("Feature engineering complete.")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Saved to: {OUTPUT_FILE}")
    print()
    print("Original delhincr.csv was NOT modified.")


if __name__ == "__main__":
    main()