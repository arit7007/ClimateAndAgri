"""
climate_utils.py

Reusable helpers for NOAA climate data processing:
--------------------------------------------------
- Generian NOAA data download 
- Map stations to counties locally using GeoPandas + pygris
- Pivot to monthly station-level format
- Growing season lookup

These functions help prepare climate data for merging with yield data
using county_fips and year as keys.
"""

import requests
import time
import pandas as pd
import geopandas as gpd
import pygris

from ratelimit import limits, sleep_and_retry
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
from requests.exceptions import ReadTimeout, RequestException

from config import NOAA_TOKEN, NOAA_BASE_URL
from utils import logging


# === Generic download from NOAA ===
@sleep_and_retry
@limits(calls=5, period=1)
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    retry=retry_if_exception_type((ReadTimeout, RequestException)),
    before_sleep=before_sleep_log(logging.getLogger(), logging.WARNING)
)
def fetch_noaa_data(endpoint: str, params: dict) -> dict:
    headers = {"token": NOAA_TOKEN}
    try:
        response = requests.get(endpoint, headers=headers, params=params, timeout=(10, 90))
        response.raise_for_status()
    except Exception as e:
        # Rebuild the URL that was attempted
        request = requests.Request('GET', endpoint, headers=headers, params=params).prepare()
        logging.error(f"Request failed. Final URL:\n{request.url}")
        raise e  # re-raise to trigger retry logic
    if response.status_code == 429:
        logging.error(f"429 Rate Limit. Response: {response.text}")
        raise SystemExit("NOAA API limit exceeded.")
    return response.json()


# === Download station metadata from NOAA so it can be used for county-level mapping ===
def download_station_metadata(state_fips: str, dataset_id: str = "GSOM") -> pd.DataFrame:
    all_stations = []
    offset = 0
    while True:
        params = {
            "datasetid": dataset_id,
            "locationid": f"FIPS:{state_fips}",
            "limit": 1000,
            "offset": offset
        }
        data = fetch_noaa_data(f"{NOAA_BASE_URL}/stations", params)
        if not data.get("results"):
            break
        all_stations.extend(data["results"])
        offset += len(data["results"])
        time.sleep(1.2)
    df = pd.DataFrame(all_stations)
    if not df.empty:
        df = df.rename(columns={"id": "station"})
    return df


# === Mapping station with county FIPS ===
def enrich_station_with_county_fips(stations_df: pd.DataFrame, state_abbr_list: list) -> pd.DataFrame:
    """
    Map stations to counties locally using lat/lon and pygris county shapes.
    """
    #if "id" in stations_df.columns and "station" not in stations_df.columns:
    #    stations_df = stations_df.rename(columns={"id": "station"})

    if "longitude" not in stations_df.columns or "latitude" not in stations_df.columns:
        raise ValueError("Stations DataFrame must include 'longitude' and 'latitude' columns.")

    counties_gdf = pygris.counties(state=state_abbr_list).to_crs("EPSG:4326")

    stations_gdf = gpd.GeoDataFrame(
        stations_df,
        geometry=gpd.points_from_xy(stations_df.longitude, stations_df.latitude),
        crs="EPSG:4326"
    )

    enriched_df = gpd.sjoin(stations_gdf, counties_gdf, how="left", predicate="within")
    #enriched_df["county_fips"] = joined["STATEFP"] + joined["COUNTYFP"]
    enriched_df["county_fips"] = enriched_df["GEOID"]   # Supposed to be same as STATEFP + COUNTYFP

    return enriched_df.reset_index(drop=True)


# === Pivot monthly station-level ===
def pivot_monthly_station_level(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    grouped = df.groupby(
        ["county_fips", "station", "year", "month", "datatype"]
    )["value"].mean().reset_index()

    pivot = grouped.pivot_table(
        index=["county_fips", "station", "year", "month"],
        columns="datatype", values="value"
    ).reset_index()

    pivot.columns.name = None
    return pivot


# === Growing months helper ===
CROP_GROWING_MONTHS = {
    "CORN": list(range(4, 10)),  # April–Sept
    "SOYBEANS": list(range(5, 10)),  # May–Sept
    "WHEAT": [10, 11, 3, 4, 5],  # Example for winter wheat
    "RICE": list(range(5, 11)),  # May–Oct
    "BARLEY": list(range(4, 9)),  # April–Aug
    "SORGHUM": list(range(5, 10))  # May–Sept
}


def get_growing_months(crop_name: str) -> list:
    """
    Return list of months for the crop growing season.
    Defaults to full year if not found.
    """
    return CROP_GROWING_MONTHS.get(crop_name.upper(), list(range(1, 13)))
