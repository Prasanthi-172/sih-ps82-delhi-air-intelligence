"""
01_validate_data.py
--------------------
PURPOSE:
Before we do ANY machine learning, we must make sure our raw data is clean.
This script checks the raw file (data/delhincr.csv) for common problems:
    - missing rows / missing hours
    - duplicate timestamps
    - missing values (NaN)
    - unsorted timestamps

IMPORTANT: This script NEVER modifies delhincr.csv. It only reads it and
writes a validation report. If everything passes, later scripts will use
delhincr.csv as their starting point.

Run this file first:
    python 01_validate_data.py
"""

import pandas as pd
import os
import sys

RAW_FILE = "data/delhincr.csv"
REPORT_FILE = "outputs/01_validation_report.txt"

def main():
    print("Loading dataset...")

    if not os.path.exists(RAW_FILE):
        print(f"ERROR: Could not find {RAW_FILE}")
        print("Make sure delhincr.csv is inside the data/ folder.")
        sys.exit(1)
    df = pd.read_csv(RAW_FILE)
    report_lines = []
    report_lines.append("DATA VALIDATION REPORT")
    report_lines.append("=" * 50)
    # 1. Basic shape check
    report_lines.append(f"Rows: {df.shape[0]}")
    report_lines.append(f"Columns: {df.shape[1]}")
    # 2. Parse timestamp column
    # We try the standard format first, then fall back to automatic parsing.
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    except Exception as e:
        print(f"ERROR: Could not parse the timestamp column: {e}")
        sys.exit(1)

    # 3. Check timestamps are sorted (oldest to newest)
    is_sorted = df["timestamp"].is_monotonic_increasing
    report_lines.append(f"Timestamps sorted (oldest->newest): {is_sorted}")
    if not is_sorted:
        report_lines.append("WARNING: Timestamps are NOT sorted. Sort them before continuing.")

    # 4. Check for duplicate timestamps
    n_duplicates = df["timestamp"].duplicated().sum()
    report_lines.append(f"Duplicate timestamps: {n_duplicates}")

    # 5. Check for missing hours (a full year of hourly data should have no gaps)
    full_range = pd.date_range(start=df["timestamp"].min(),
                                end=df["timestamp"].max(),
                                freq="h")
    missing_hours = full_range.difference(df["timestamp"])
    report_lines.append(f"Missing hours in the timeline: {len(missing_hours)}")

    # 6. Check for missing values (NaN) in every column
    na_counts = df.isna().sum()
    total_na = na_counts.sum()
    report_lines.append(f"Total missing (NaN) values across all columns: {total_na}")
    if total_na > 0:
        report_lines.append("Columns with missing values:")
        for col, count in na_counts[na_counts > 0].items():
            report_lines.append(f"   - {col}: {count} missing")

    # 7. Check for infinite values (can break some ML models)
    numeric_df = df.select_dtypes(include=["float64", "int64"])
    n_inf = ((numeric_df == float("inf")) | (numeric_df == float("-inf"))).sum().sum()
    report_lines.append(f"Infinite values found: {n_inf}")

    # ---- Final verdict ----
    all_good = (n_duplicates == 0) and (len(missing_hours) == 0) and (total_na == 0) \
        and (n_inf == 0) and is_sorted

    report_lines.append("=" * 50)
    if all_good:
        report_lines.append("RESULT: Dataset passed all validation checks.")
    else:
        report_lines.append("RESULT: Dataset has issues. Review warnings above before modeling.")

    # Print to screen
    for line in report_lines:
        print(line)

    # Save report to file
    os.makedirs("outputs", exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        f.write("\n".join(report_lines))

    print(f"\nValidation report saved to {REPORT_FILE}")
    print("NOTE: delhincr.csv was only read, never modified.")


if __name__ == "__main__":
    main()
