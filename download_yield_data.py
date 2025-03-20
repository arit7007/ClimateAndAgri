import pandas as pd
import logging
import requests
import json
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_usda_crop_yield(state='CALIFORNIA', crop="CORN", start_year=2010, end_year=2023, limit=50000):
  all_data = []

  for year in range(start_year, end_year + 1):
    logging.info(f"Fetching USDA data for {crop} in {state}, Year: {year}")
    params = {
            "key": NASS_API_KEY,
            "source_desc": "SURVEY",
            "sector_desc": "CROPS",
            "group_desc": "FIELD CROPS",
            "commodity_desc": crop,
            "agg_level_desc": "COUNTY",
            "state_name": state,
            "year": year,
            "format": "json",
            "limit": limit
    }
            #"unit_desc": "BU / ACRE",

    try:
        response = requests.get(NASS_BASE_URL, params=params)
        response.raise_for_status()  # Raises HTTPError for bad responses

        try:
            response_json = response.json()
        except json.JSONDecodeError as e:
            logging.warning(f"JSON Decode Error for {year}: {e}\nResponse Text: {response.text[:500]}")
            continue

        if "error" in response_json:
            logging.warning(f"USDA API Error for {year}: {response_json['error']}")
            continue

        data = response_json.get("data", [])
        if data:
          df = pd.DataFrame(data)
          all_data.append(df)
          logging.info(f"Retrieved {len(df)} records for {year}.")
        else:
          logging.warning(f"No data found for {year}.")

    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP Request Error for {year}: {e}")
        logging.error(f"Response Text: {response.text[:500] if 'response' in locals() else 'No Response'}")

if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    logging.info(f"Available columns in yield df: {combined_df.columns.tolist()}")
    combined_df.to_csv(CROP_YIELD_DATA, index=False)
    logging.info(f"USDA Data saved: {CROP_YIELD_DATA}")
    return combined_df
else:
    logging.info("No data was retrieved for yield data.")
    return None


if __name__ == "__main__":
    get_usda_crop_yield(state="CALIFORNIA", crop="CORN", start_year=2020, end_year=2021)
