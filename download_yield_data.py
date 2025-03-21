import logging
import requests
import itertools
import us
import json
import pandas as pd
import utils
from config import NASS_BASE_URL, NASS_API_KEY, CROP_YIELD_DATA

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_usda_crop_yield(api_key,
                        state_alpha=["CA"],
                        crops=["CORN"],
                        statisticcat_descs=["YIELD"],
                        start_year=2020,
                        end_year=2020
                        ):
                        #group_descs = ["FIELD CROPS"],

    base_url = NASS_BASE_URL
    all_records = []

    # Create the cartesian product for years, states, commodities, groups, and statistic categories
    combinations = itertools.product(
        range(start_year, end_year + 1),
        state_alpha,
        crops,
        statisticcat_descs
    )

    for year, state_alpha, crop, stat_desc in combinations:
        params = {
            "key": api_key,
            "source_desc": "SURVEY",
            "sector_desc": "CROPS",
            "agg_level_desc": "COUNTY",
            "state_alpha": state_alpha,
            "year": str(year),
            "commodity_desc": crop,
            "statisticcat_desc": stat_desc,
            "format": "JSON"
        }

        logging.info(f"Downloading data for {year} - {state_alpha} - {stat_desc} - {crop}  ...")
        response = requests.get(base_url, params=params)
        logging.info(f"url: {response.url}")
        try:
            response.raise_for_status()
        except Exception as e:
            print(f"Request failed for {year} - {state_alpha} - {stat_desc} - {crop}: {e}")
            continue

        data = response.json()
        records = data.get("data", [])
        if records:
            all_records.extend(records)
        else:
            print(f"No data returned for {year} - {state_alpha} - {stat_desc} - {crop}")

    # Convert the results to a DataFrame
    df = pd.DataFrame(all_records)
    if not df.empty:
        # Convert "year" column to numeric if possible
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

        # Clean the "Value" column: remove commas and convert to numeric
        df["Value"] = df["Value"].str.replace(",", "", regex=False)
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

        # Combine state_fips and county_code to make a 5 digit county_fips
        df["county_fips"] = df["state_fips_code"].astype(str) + df["county_code"].str.zfill(3)
      
    return df


if __name__ == "__main__":
    df = get_usda_crop_yield(api_key=NASS_API_KEY, state_alpha=["IA", "CA"], crops=["CORN"], start_year=2020, end_year=2021)
    utils.save_df(df, CROP_YIELD_DATA)
