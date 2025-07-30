"""
USDA Crop Yield Data Downloader
-------------------------------

This script downloads county-level crop yield data from the USDA NASS Quick Stats API
for a given list of U.S. states, crops, and a specified year range.

Inputs:
- --states: List of U.S. state abbreviations (e.g., ["CA", "IA"])
- --crops: List of crop names (e.g., ["CORN", "SOYBEANS"])
- --start_year: Start year for the data (e.g., 2010)
- --end_year: End year (inclusive) for the data (e.g., 2014)

Outputs:
- Pivoted ML-ready data which will get joined with other data such as climate data, vegetation indices and soil data. 
Typical Usage:
--------------
python download_yield_data.py --states IA CA --crops CORN SOYBEANS --start_year 2010 --end_year 2014

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


def pivot_yield_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    exclude_util_practice = ["SILAGE"]      # Conflicting with different unit_desc for the same crop
    keep_stats = ["AREA HARVESTED", "AREA PLANTED", "PRODUCTION", "YIELD"]

    df = df[~df["util_practice_desc"].isin(exclude_util_practice)]
    df = df[df["statisticcat_desc"].isin(keep_stats)]
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    # Pivot to wide format
    pivot = df.pivot_table(
        index=["county_fips", "year", "commodity_desc"],
        columns="statisticcat_desc",
        values="Value",
        aggfunc="mean"
    ).reset_index()
    pivot.columns.name = None

    # Metadata from any row
    meta = (
        df.groupby(["county_fips", "year", "commodity_desc"])[
            ["class_desc", "prodn_practice_desc", "util_practice_desc", "CV (%)"]
        ]
        .first()
        .reset_index()
        .rename(columns={"CV (%)": "cv_mean"})
    )

    # Yield unit only from YIELD rows
    yield_units = (
        df[df["statisticcat_desc"] == "YIELD"]
        .groupby(["county_fips", "year", "commodity_desc"])["unit_desc"]
        .first()
        .reset_index()
        .rename(columns={"unit_desc": "yield_unit"})
    )

    return (
        pivot
        .merge(meta, on=["county_fips", "year", "commodity_desc"], how="left")
        .merge(yield_units, on=["county_fips", "year", "commodity_desc"], how="left")
    )


def main():
    parser = argparse.ArgumentParser(description="Fetch yield data for various crops")
    parser.add_argument("--states", nargs="+",
                        default=["IA", "CA", "IL", "NE", "MN", "TX", "AR", "LA", "WA", "OR", "ID"],
                        help="List of US state abbreviations")
    parser.add_argument("--crops", nargs="+",
                        default=["CORN", "SOYBEANS", "WHEAT", "RICE", "BARLEY", "SORGHUM"],
                        help="List of crops")
    parser.add_argument("--start_year", type=int, default=2010, help="Start year for data")
    parser.add_argument("--end_year", type=int, default=2014, help="End year for data")
    args = parser.parse_args()

    raw_df = get_usda_crop_yield(NASS_API_KEY, args.states, args.crops, args.start_year, args.end_year)
    utils.save_df(raw_df, f"{DATA_PATH}/crop_yield_new_{args.start_year}_{args.end_year}_raw.csv")

    ml_df = pivot_yield_data(raw_df)
    utils.save_df(ml_df, f"{DATA_PATH}/final_crop_yield_data_{args.start_year}_{args.end_year}.csv")


if __name__ == "__main__":
    main()
