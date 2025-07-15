"""
Climate Data Pipeline

Downloads NOAA GSOM climate data, maps stations to county FIPS,
pivots to monthly station-level format, and aggregates to county-year level.
"""

import argparse
import time
from pathlib import Path
from typing import List

import pandas as pd

from climate_utils import (
    download_station_metadata,
    fetch_noaa_data,
    enrich_station_with_county_fips
)
from config import NOAA_BASE_URL, DATA_PATH
from utils import fetch_state_fips, save_df, logging

# === Constants ===
DATASET_ID = "GSOM"
DATATYPE = [
    "TMIN", "TMAX", "PRCP", "SNOW", "SNWD",
    "HTDD", "CLDD", "TSUN", "AWND", "EMXT", "EMNT", "RHAV"
]
BATCH_SIZE = 1000

CHECKPOINT_FILE = DATA_PATH / "checkpoint_climate_state_year.csv"
CLIMATE_RAW_FILE = DATA_PATH / "climate_raw_combined.csv"
CLIMATE_RAW_FILE_PIVOTED = DATA_PATH / "climate_raw_combined_pivoted.csv"

STATION_CHECKPOINT_FILE = DATA_PATH / "checkpoint_stations.csv"
STATIONS_FILE = DATA_PATH / "station_metadata.csv"
STATIONS_FILE_WITH_COUNTY = DATA_PATH / "station_metadata_with_county.csv"


# === Checkpoint helpers ===
def load_checkpoint(path: Path) -> set:
    if not path.exists():
        return set()
    df = pd.read_csv(path, dtype=str)
    return set((row["state_fips"], int(row["year"])) for _, row in df.iterrows())


def update_checkpoint(path: Path, state_fips: str, year: int) -> None:
    df = pd.DataFrame([[state_fips, year]], columns=["state_fips", "year"])
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, mode="w", header=True, index=False)


def load_station_checkpoint(path: Path) -> set:
    if not path.exists():
        return set()
    df = pd.read_csv(path, dtype=str)
    return set(df["state_fips"])



def update_station_checkpoint(path: Path, state_fips: str):
    df = pd.DataFrame([[state_fips]], columns=["state_fips"])
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, mode="w", header=True, index=False)

# Pivot from row-wise format to column-wise at the DATATYPE column (e.g.  is PRCP, 
def pivot_climate_data(df):
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    grouped = df.groupby(
        ["station", "year", "month", "datatype"]
    )["value"].mean().reset_index()

    pivoted = grouped.pivot_table(
        index=["station", "year", "month"],
        columns="datatype",
        values="value"
    ).reset_index()

    return pivoted


def aggregate_climate_by_period(pivoted_df: pd.DataFrame, months: list) -> pd.DataFrame:
    df = pivoted_df[pivoted_df["month"].isin(months)].copy()

    grouping_cols = ["county_fips", "year"]
    # Only use the columns of interest
    agg_cols = [col for col in DATATYPE if col in df.columns]

    logging.info(f"Aggregating columns: {agg_cols}")

    grouped = df.groupby(grouping_cols)[agg_cols].mean().reset_index()
    return grouped


# === Download raw NOAA climate data for one state/year ===
def download_climate_data_by_state(state_fips: str, year: int) -> pd.DataFrame:
    all_data = []
    offset = 0
    while True:
        params = {
            "datasetid": DATASET_ID,
            "locationid": f"FIPS:{state_fips}",
            "startdate": f"{year}-01-01",
            "enddate": f"{year}-12-31",
            "limit": BATCH_SIZE,
            "offset": offset,
            "units": "metric",
            "datatypeid": ",".join(DATATYPE)
        }
        data = fetch_noaa_data(f"{NOAA_BASE_URL}/data", params)
        if not data.get("results"):
            break
        all_data.extend(data["results"])
        offset += len(data["results"])
        logging.info(f"Processed {offset} rows for FIPS {state_fips} year {year}")
        time.sleep(1.2)
    return pd.DataFrame(all_data)


# === Main pipeline ===
def climate_data_pipeline(states: List[str], start_year: int, end_year: int):
    state_fips_list = fetch_state_fips(states)
    climate_checkpoint = load_checkpoint(CHECKPOINT_FILE)

    # === Download raw climate data ===
    for state_abbr, state_fips in zip(states, state_fips_list):
        for year in range(start_year, end_year + 1):
            if (state_fips, year) in climate_checkpoint:
                logging.info(f"Skipping {state_abbr} {year} (checkpointed)")
                continue

            logging.info(f"Downloading {state_abbr} {year}...")
            df = download_climate_data_by_state(state_fips, year)
            if not df.empty:
                df["state_abbr"] = state_abbr
                if not CLIMATE_RAW_FILE.exists():
                    df.to_csv(CLIMATE_RAW_FILE, mode="w", index=False)
                else:
                    df.to_csv(CLIMATE_RAW_FILE, mode="a", index=False, header=False)

                update_checkpoint(CHECKPOINT_FILE, state_fips, year)
                logging.info(f"Added {len(df)} rows for {state_abbr} {year}")
            else:
                logging.info(f"No data for {state_abbr} {year} — skipping write and checkpoint.")

    logging.info(f"Reading raw climate data from {CLIMATE_RAW_FILE.resolve()}...")
    climate_df = pd.read_csv(CLIMATE_RAW_FILE)
    climate_df.drop_duplicates(inplace=True)

    # === Pivot climate data to have the various climate elements as column for ML ready ===
    pivoted_climate_df = pivot_climate_data(climate_df)
    save_df(pivoted_climate_df, CLIMATE_RAW_FILE_PIVOTED)
    logging.info(f"Pivoted raw climate data saved to {CLIMATE_RAW_FILE_PIVOTED.resolve()}...")

    # === Download station metadata with checkpoint ===
    logging.info("Downloading and enriching station metadata...")
    if STATIONS_FILE.exists():
        stations_df = pd.read_csv(STATIONS_FILE)
        logging.info(f"Loaded existing stations metadata ({len(stations_df)} rows).")
    else:
        stations_df = pd.DataFrame()

    station_checkpoint = load_station_checkpoint(STATION_CHECKPOINT_FILE)
    logging.info(
        f"Station checkpoint: {len(station_checkpoint)} states done, {len(state_fips_list) - len(station_checkpoint)} to download.")

    new_stations = []
    for state_fips in state_fips_list:
        if state_fips in station_checkpoint:
            logging.info(f"Skipping FIPS {state_fips} (checkpointed)")
            continue
        logging.info(f"Downloading stations for FIPS {state_fips}...")
        df = download_station_metadata(state_fips)
        if not df.empty:
            df["state_fips"] = state_fips
            new_stations.append(df)
            update_station_checkpoint(STATION_CHECKPOINT_FILE, state_fips)
            logging.info(f"Downloaded {len(df)} stations for FIPS {state_fips}")

    if new_stations:
        combined_new = pd.concat(new_stations, ignore_index=True)
        stations_df = pd.concat([stations_df, combined_new], ignore_index=True)
        stations_df.drop_duplicates(subset=["station"], inplace=True)
    else:
        logging.info("No new stations downloaded.")

    save_df(stations_df, STATIONS_FILE)
    logging.info(f"Saved stations metadata to {STATIONS_FILE}")

    # === Map stations to counties and add county-level FIPS ===
    enriched_stations_df = enrich_station_with_county_fips(stations_df, states)
    save_df(enriched_stations_df, STATIONS_FILE_WITH_COUNTY)
    logging.info(f"Saved enriched stations metadata with county fips to {STATIONS_FILE_WITH_COUNTY.resolve()}")

    # === Join climate data with county FIPS ===
    logging.info("Joining pivoted climate data with enriched stations data...")
    joined_df = pivoted_climate_df.merge(enriched_stations_df, on="station", how="left")
    save_df(joined_df, DATA_PATH / f"climate_with_county_{start_year}_{end_year}.csv")
    logging.info(
        f"Climate records: {len(pivoted_climate_df)} | Stations: {len(enriched_stations_df)} | Joined: {len(joined_df)}")

    # === Aggregate county-year ===
    logging.info("Aggregating to county-year...")
    grow_months = list(range(1, 13))  # aggregating for the whole year, whithout consideration for crop-specific growing season
    county_df = aggregate_climate_by_period(joined_df, months=grow_months)
    save_df(county_df, DATA_PATH / f"climate_with_county_aggregated_{start_year}_{end_year}.csv")

    logging.info("Pipeline complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process NOAA climate data pipeline")
    parser.add_argument("--states", nargs="+",
                        default=["IA", "CA", "IL", "NE", "MN", "TX", "AR", "LA", "WA", "OR", "ID"],
                        help="List of US state abbreviations")
    parser.add_argument("--start_year", type=int, default=2010)
    parser.add_argument("--end_year", type=int, default=2024)
    args = parser.parse_args()

    climate_data_pipeline(args.states, args.start_year, args.end_year)
