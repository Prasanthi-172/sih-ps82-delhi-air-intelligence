"""
03_create_targets.py
---------------------
PURPOSE:
Create the columns we are actually trying to predict: PM2.5 at each of the
next 72 hours.

HOW IT WORKS:
For every row (which represents "now", time T), we look FORWARD in the
data and grab the PM2.5 value that actually happened at T+1h, T+2h, ...,
T+72h, and store each one in its own column:

    pm25_future_1h, pm25_future_2h, ..., pm25_future_72h

This uses pandas' .shift(-n), which pulls a value from n rows AHEAD.

IMPORTANT: These columns are TARGETS (the answers), not features (the
model must never see them as an input). Script 04 will keep these separate.

Because we shift forward, the LAST 72 rows of the dataset won't have a
complete set of future values (there's no data beyond December 31st to look
ahead to). We drop those incomplete rows -- this is normal and expected.

INPUT:  data/02_features.csv
OUTPUT: data/03_features_with_targets.csv

Run this after 02_feature_engineering.py:
    python 03_create_targets.py
"""

import pandas as pd
import os

INPUT_FILE = "data/02_features.csv"
OUTPUT_FILE = "data/03_features_with_targets.csv"

FORECAST_HORIZON = 72  # hours


def main():
    print("Loading dataset...")
    df = pd.read_csv(INPUT_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    print("Creating targets...")

    for h in range(1, FORECAST_HORIZON + 1):
        df[f"pm25_future_{h}h"] = df["pm2_5_ugm3"].shift(-h)

    rows_before = len(df)

    # Drop rows near the end of the dataset that don't have a full set of
    # 72 future values to predict (there's no "future" data beyond the
    # last recorded hour).
    target_cols = [f"pm25_future_{h}h" for h in range(1, FORECAST_HORIZON + 1)]
    df = df.dropna(subset=target_cols).reset_index(drop=True)

    rows_after = len(df)
    print(f"Rows before dropping incomplete targets: {rows_before}")
    print(f"Rows after dropping incomplete future-targets: {rows_after}")
    print(f"(Dropped {rows_before - rows_after} rows from the very end of the year "
          f"-- expected, since we can't look 72 hours past Dec 31st.)")

    # ------------------------------------------------------------------
    # Also drop rows at the VERY START of the year that don't have a full
    # lag history yet (e.g. pm25_lag_24h needs 24 hours of PAST data,
    # which doesn't exist for the first day of the year). This is the
    # mirror-image problem of the one above, just at the other end of
    # the timeline.
    # ------------------------------------------------------------------
    rows_before_start_drop = len(df)
    df = df.dropna().reset_index(drop=True)
    rows_after_start_drop = len(df)
    print(f"Dropped {rows_before_start_drop - rows_after_start_drop} rows from the very "
          f"start of the year (not enough history yet for lag features -- expected).")

    os.makedirs("data", exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
