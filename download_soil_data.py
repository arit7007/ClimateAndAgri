import pandas as pd
import numpy as np
import tempfile
import rasterio
import argparse
from tqdm import tqdm
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from requests.exceptions import RequestException, HTTPError
from pathlib import Path

from soilgrids import SoilGrids
from utils import get_county_geometries
from config import DATA_PATH, logging

CHECKPOINT_FILE = DATA_PATH / "soil_checkpoint.csv"
OUTPUT_FILE = DATA_PATH / "county_soil_features.csv"
RESOLUTION = 256


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionResetError, OSError, RequestException, HTTPError))
)
def download_soil_layer(soil_grids, service_id, coverage_id, bbox, crs, resolution, outpath):
    return soil_grids.get_coverage_data(
        service_id=service_id,
        coverage_id=coverage_id,
        west=bbox[0], south=bbox[1], east=bbox[2], north=bbox[3],
        crs=crs,
        width=resolution,
        height=resolution,
        output=outpath
    )


def load_checkpoint(path: Path) -> set:
    if not path.exists():
        return set()
    df = pd.read_csv(path, dtype=str)
    return set(df["county_fips"])


def update_checkpoint(path: Path, county_fips: str) -> None:
    df = pd.DataFrame([[county_fips]], columns=["county_fips"])
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def process_county_static(county, soil_grids, soil_properties, resolution):
    county_fips = county['county_fips']
    county_name = county['county']
    bbox = county.geometry.bounds if county.geometry.is_valid else county.geometry.buffer(0).bounds

    county_data = {
        'county_fips': county_fips,
        'county_name': county_name,
        'state_abbr': county['state_abbr'],
        'state_fips': county['state_fips']
    }

    for prop_name, service_id, coverage_id in soil_properties:
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp.close()
            tmp_path = Path(tmp.name)
            try:
                download_soil_layer(
                    soil_grids, service_id, coverage_id,
                    bbox=bbox,
                    crs="urn:ogc:def:crs:EPSG::4326",
                    resolution=resolution,
                    outpath=str(tmp_path)
                )
                with rasterio.open(tmp_path) as src:
                    data = src.read(1)
                    county_data[f"{prop_name}_mean"] = float(np.nanmean(data))
            except Exception as e:
                logging.info(f"[{county_name}] {prop_name} failed: {e}")
                county_data[f"{prop_name}_mean"] = np.nan
            finally:
                tmp_path.unlink(missing_ok=True)

    return county_data


def get_soil_data(state_abbr_list: List[str], resolution=256, max_workers=8) -> None:
    counties = get_county_geometries(state_abbr_list)
    checkpointed = load_checkpoint(CHECKPOINT_FILE)
    counties = counties[~counties['county_fips'].isin(checkpointed)]

    soil_grids = SoilGrids()
    soil_properties = [
        ("clay", "clay", "clay_0-5cm_mean"),
        ("silt", "silt", "silt_0-5cm_mean"),
        ("sand", "sand", "sand_0-5cm_mean"),
        ("ph", "phh2o", "phh2o_0-5cm_mean"),
        ("soc", "soc", "soc_0-5cm_mean"),
        ("bdod", "bdod", "bdod_0-5cm_mean")
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(total=len(counties), desc="Downloading Soil Data") as pbar:
        futures = {}
        for _, row in counties.iterrows():
            futures[executor.submit(process_county_static, row, soil_grids, soil_properties, resolution)] = row['county_fips']

        batch_results = []
        for future in as_completed(futures):
            county_fips = futures[future]
            try:
                result = future.result()
                batch_results.append(result)
            except Exception as exc:
                logging.info(f"Failed for {county_fips}: {exc}")
            finally:
                pbar.update(1)

            if batch_results:
                df_batch = pd.DataFrame(batch_results)
                if OUTPUT_FILE.exists():
                    df_batch.to_csv(OUTPUT_FILE, mode="a", header=False, index=False)
                else:
                    df_batch.to_csv(OUTPUT_FILE, mode="w", header=True, index=False)

                for entry in batch_results:
                    update_checkpoint(CHECKPOINT_FILE, entry['county_fips'])

                batch_results.clear()


def main():
    parser = argparse.ArgumentParser(description="Download soil data by county for given states")
    parser.add_argument("--states", nargs="+", default=["CA", "WA"], help="List of US state abbreviations")
    args = parser.parse_args()

    get_soil_data(args.states)
    logging.info("Soil data download complete.")


if __name__ == "__main__":
    main()
