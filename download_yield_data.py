"""
USDA Crop Yield Data Downloader
-------------------------------

This script downloads county-level crop yield data from the USDA NASS Quick Stats API
for a given list of U.S. states, crops, and a specified year range.

Key Features:
- Pulls yield, production, and area data for each crop-year-state combination.
- Handles pagination, retries, and request errors gracefully.
- Saves raw data as a CSV file with relevant fields, including `county_fips`, `year`, `commodity_desc`, `Value`, and other descriptors.
- Optionally pivots the data to organize multiple statistics (e.g., YIELD, AREA HARVESTED) into separate columns for ML-ready analysis.

Inputs:
- --states: List of U.S. state abbreviations (e.g., ["CA", "IA"])
- --crops: List of crop names (e.g., ["CORN", "SOYBEANS"])
- --start_year: Start year for the data (e.g., 2010)
- --end_year: End year (inclusive) for the data (e.g., 2024)

Outputs:
- Raw data saved to `crop_yield_{start_year}_{end_year}_raw.csv` in the configured DATA_PATH
- (Optional) Pivoted ML-ready data saved to `crop_yield_{start_year}_{end_year}.csv`

Typical Usage:
--------------
python download_yield_data.py --states IA CA --crops CORN SOYBEANS --start_year 2010 --end_year 2024

Note:
- Data is fetched only for crops that are reported at the county level under `source_desc="SURVEY"` and `sector_desc="CROPS"`.
"""

import requests
import argparse
import pandas as pd
import time
import random
import utils
from config import NASS_BASE_URL, NASS_API_KEY, DATA_PATH, logging
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_sleep_log
)
from requests.exceptions import RequestException, HTTPError

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json"
}


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=10),
    retry=retry_if_exception_type((RequestException, HTTPError)),
    before_sleep=before_sleep_log(logging.getLogger(), logging.WARNING),
    reraise=True
)
def make_request_with_retry(params):
    response = requests.get(NASS_BASE_URL, params=params, headers=HEADERS)
    logging.debug(f"URL: {response.url}")
    response.raise_for_status()
    return response.json().get("data", [])


def fetch_all_pages(params):
    all_data = []
    offset = 0
    limit = 5000
    while True:
        paginated = {**params, "limit": limit, "offset": offset}
        try:
            chunk = make_request_with_retry(paginated)
            if not chunk:
                break
            all_data.extend(chunk)
            logging.info(f"Fetched {len(chunk)} rows from offset {offset}")
            offset += len(chunk)
            if len(chunk) < limit:
                break
        except Exception as e:
            logging.warning(f"Skipping offset {offset} due to error: {e}")
            break
        time.sleep(random.uniform(3.0, 5.0))
    return all_data


def get_usda_crop_yield(api_key, state_alpha, crops, start_year, end_year):
    all_records = []
    for year in range(start_year, end_year + 1):
        for state in state_alpha:
            for crop in crops:
                params = {
                    "key": api_key,
                    "source_desc": "SURVEY",
                    "sector_desc": "CROPS",
                    "agg_level_desc": "COUNTY",
                    "state_alpha": state,
                    "year": str(year),
                    "commodity_desc": crop,
                    "format": "JSON"
                }

                logging.info(f"Fetching all stats for {year} - {state} - {crop}...")
                data = fetch_all_pages(params)
                if data:
                    all_records.extend(data)
                else:
                    logging.warning(f"No data for {year} - {state} - {crop}")

    df = pd.DataFrame(all_records).drop_duplicates()
    if not df.empty:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["Value"] = df["Value"].str.replace(",", "", regex=False)
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        df["county_fips"] = df["state_fips_code"].astype(str).str.zfill(2) + df["county_code"].astype(str).str.zfill(3)
    return df


def pivot_usda_crop_data(df):
    if df.empty:
        return df

    required = {"county_fips", "year", "commodity_desc", "statisticcat_desc", "Value",
                "unit_desc", "class_desc", "prodn_practice_desc", "util_practice_desc", "CV (%)"}
    if missing := required - set(df.columns):
        raise ValueError(f"Missing columns: {missing}")

    df["CV (%)"] = pd.to_numeric(df["CV (%)"], errors="coerce")
    pivot = df.pivot_table(index=["county_fips", "year", "commodity_desc"],
                           columns="statisticcat_desc", values="Value", aggfunc="mean").reset_index()
    pivot.columns.name = None

    group_keys = ["county_fips", "year", "commodity_desc"]
    for col in ["unit_desc", "class_desc", "prodn_practice_desc", "util_practice_desc"]:
        pivot[col] = df.groupby(group_keys)[col].first().values

    pivot["cv_mean"] = df.groupby(group_keys)["CV (%)"].mean().values
    return pivot


def main():
    parser = argparse.ArgumentParser(description="Fetch yield data for various crops")
    parser.add_argument("--states", nargs="+",
                        default=["IA", "CA", "IL", "NE", "MN", "TX", "AR", "LA", "WA", "OR", "ID"],
                        help="List of US state abbreviations")
    parser.add_argument("--crops", nargs="+",
                        default=["CORN", "SOYBEANS", "WHEAT", "RICE", "BARLEY", "SORGHUM"],
                        help="List of crops")
    parser.add_argument("--start_year", type=int, default=2010, help="Start year for data")
    parser.add_argument("--end_year", type=int, default=2024, help="End year for data")
    args = parser.parse_args()

    raw_df = get_usda_crop_yield(NASS_API_KEY, args.states, args.crops, args.start_year, args.end_year)
    utils.save_df(raw_df, f"{DATA_PATH}/crop_yield_{args.start_year}_{args.end_year}_raw.csv")

    # Temporarily disable this as previous line is added to save the raw data
    ml_df = pivot_usda_crop_data(raw_df)
    utils.save_df(ml_df, f"{DATA_PATH}/crop_yield_{args.start_year}_{args.end_year}.csv")


if __name__ == "__main__":
    main()
