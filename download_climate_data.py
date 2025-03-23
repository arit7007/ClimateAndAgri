import requests
import logging
import pandas as pd
import geopandas as gpd
from typing import List, Dict, Any
from ratelimit import limits, sleep_and_retry
from tenacity import retry, stop_after_attempt, wait_exponential
from requests.exceptions import RequestException, ReadTimeout
from config import NOAA_TOKEN, NOAA_BASE_URL, CLIMATE_DATA_WITH_FIPS
from utils import fetch_state_fips, save_df

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(messages)s")

DATASET_ID = "GSOM"
ELEMENTS = ["TMIN", "TMAX", "PRCP", "SNOW", "SNWD", "HTDD", "CLDD",
            "TSUN", "AWND", "EMXT", "EMNT", "RHAV"]
BATCH_SIZE = 1000
TIMEOUT = 20
COUNTY_SHAPEFILE = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_county_500k.zip"


@sleep_and_retry
@limits(calls=5, period=1)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_noaa_data_with_retries(end_point: str, params: Dict[str, Any]):
    headers = {"token": NOAA_TOKEN}
    try:
        response = requests.get(end_point, headers=headers, params=params, timeout=TIMEOUT)
        logging.debug(f"url: {response.url}")
        response.raise_for_status()
        return response.json()
    except ReadTimeout:
        logging.warning(f"Read timeout error for {end_point}. Retrying...")
        raise
    except RequestException as e:
        logging.error(f"Request failed: {str(e)}")
        return None


def fetch_weather_stations_by_fips(fips: str) -> pd.DataFrame:
    stations = []
    offset = 0

    while True:
        params = {
                "datasetid": DATASET_ID,
                "locationid": f"FIPS:{fips}",
                "limit": BATCH_SIZE,
                "offset": offset,
        }

        data = fetch_noaa_data_with_retries(f"{NOAA_BASE_URL}/stations", params)
        if data and "results" in data:
            stations.extend(data["results"])
            offset += len(data["results"])
        else:
            break

    stations_df = pd.DataFrame(stations)
    if "id" not in stations_df:
        return pd.DataFrame()

    stations_df.rename(columns={"id": "station", "name": "city"}, inplace=True)
    stations_df["city"] = stations_df["city"].str.split(",").str[0]
    return stations_df


def fetch_weather_stations_for_states(state_abbr_list: List[str]) -> pd.DataFrame:
    fips_list = fetch_state_fips(state_abbr_list)
    logging.info(f"Fips for {state_abbr_list} is {fips_list}")
    stations = []
    for fips in fips_list:
        df = fetch_weather_stations_by_fips(fips)
        stations.append(df)
    return pd.concat(stations, ignore_index=True)


def map_stations_to_us_counties(stations_df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Mapping stations to counties")
    counties_gdf = gpd.read_file(COUNTY_SHAPEFILE).to_crs("EPSG:4326")
    stations_gdf = gpd.GeoDataFrame(
        stations_df,
        geometry=gpd.points_from_xy(stations_df.longitude, stations_df.latitude),
        crs="EPSG:4326"
    )

    stations_with_county: pd.DataFrame = gpd.sjoin(stations_gdf, counties_gdf, how="left", predicate="within")

    logging.debug(f"Columns in stations_with_county: {stations_with_county.columns}")
    stations_with_county = stations_with_county[
                ["station", "longitude", "latitude", "elevation", "city",
                "STATE_NAME", "STATEFP", "COUNTYFP", "NAME"]
            ]

    stations_with_county["county_fips"] = stations_with_county["STATEFP"] + stations_with_county["COUNTYFP"]

    stations_with_county.rename(columns={"STATE_NAME": "state", "STATEFP": "state_fips",
                                         "COUNTYFP": "city_fips", "NAME": "county"}, inplace=True)

    stations_with_county.dropna(subset=["county_fips"], inplace=True)

    return stations_with_county


def fetch_climate_data_for_fips(state_fips: str, start_year: int, end_year: int) -> pd.DataFrame:
    climate_data = []

    for year in range(start_year, end_year + 1):
        logging.info(f"Getting climate data for fips: {state_fips} and year: {year}")
        offset = 0

        while True:
            params = {
                "datasetid": DATASET_ID,
                "locationid": f"FIPS:{state_fips.zfill(2)}",
                "startdate": f"{start_year}-01-01",
                "enddate": f"{end_year}-12-31",
                "limit": BATCH_SIZE,
                "offset": offset,
                "units": "metric",
                "datatypeid": ",".join(ELEMENTS)
            }

            data = fetch_noaa_data_with_retries(f"{NOAA_BASE_URL}/data", params)
            if data and "results" in data:
                climate_data.extend(data["results"])
                offset += len(data["results"])
            else:
                break

    climate_df = pd.DataFrame(climate_data)
    if climate_df.empty:
        return pd.DataFrame()

    return climate_df


def fetch_stations_with_county_info(state_abbr_list: List[str]) -> pd.DataFrame:
    stations_df = fetch_weather_stations_for_states(state_abbr_list)
    return map_stations_to_us_counties(stations_df)


def fetch_climate_data_for_states(state_abbr_list: List[str], start_year: int, end_year: int) -> pd.DataFrame:
    logging.info(f"Getting climate data")
    climate_data = []
    state_fips = fetch_state_fips(state_abbr_list)
    for fips in state_fips:
        df = fetch_climate_data_for_fips(fips, start_year, end_year)
        climate_data.append(df)

    climate_df = pd.concat(climate_data, ignore_index=True)
    logging.debug(f"Climate data before pivot: {climate_df.head(5)}")

    climate_pivot = climate_df.pivot_table(
        index=['date', 'station'],
        columns='datatype',
        values='value',
        aggfunc="mean"
    ).reset_index()
    logging.debug(f"Climate data after pivot: {climate_pivot.head(5)}")

    return climate_pivot


def fetch_climate_data_with_county_info(state_abbr_list: List[str], start_year: int, end_year: int) -> pd.DataFrame:
    climate_df = fetch_climate_data_for_states(state_abbr_list, start_year, end_year)
    county_df = fetch_stations_with_county_info(state_abbr_list)

    return pd.merge(climate_df, county_df, on="station", how="left")


if __name__ == "__main__":
    state_abbr_list = ['IA', 'CA']
    df = fetch_climate_data_with_county_info(state_abbr_list, 2020, 2020)
    save_df(df, CLIMATE_DATA_WITH_FIPS)
