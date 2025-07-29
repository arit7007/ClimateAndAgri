import pandas as pd
from datetime import datetime
import argparse
import ee
from shapely.geometry import mapping
from pathlib import Path

from utils import get_county_geometries, save_df, logging
from config import GEE_PROJECT_ID, DATA_PATH

CHECKPOINT_PATH = DATA_PATH / "gee_ndvi_checkpoint.csv"


def authenticate_gee(project_id=GEE_PROJECT_ID, service_account_path=None):
    if service_account_path:
        credentials = ee.ServiceAccountCredentials(None, service_account_path)
        ee.Initialize(credentials, project=project_id)
    else:
        try:
            ee.Initialize(project=project_id)
        except Exception:
            ee.Authenticate()
            ee.Initialize(project=project_id)
    logging.info("Google Earth Engine authenticated.")


def load_checkpoint(path: Path) -> set:
    if not path.exists():
        return set()
    df = pd.read_csv(path, dtype=str)
    return set((row["county_fips"], int(row["year"]), int(row["month"])) for _, row in df.iterrows())


def update_checkpoint(path: Path, county_fips: str, year: int, month: int) -> None:
    df = pd.DataFrame([[county_fips, year, month]], columns=["county_fips", "year", "month"])
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def aggregate_yearly(input_path: Path, output_path: Path):
    df = pd.read_csv(input_path)
    #yearly = df.groupby(["county_fips", "year"], as_index=False).agg(
    yearly = df.groupby(["state_fips", "county_fips", "county_name", "year"], as_index=False).agg(
        ndvi_mean_year=("ndvi_mean", "mean"),
        ndvi_min_year=("ndvi_mean", "min"),
        ndvi_max_year=("ndvi_mean", "max"),
        evi_mean_year=("evi_mean", "mean"),
        evi_min_year=("evi_mean", "min"),
        evi_max_year=("evi_mean", "max")
    )
    yearly.to_csv(output_path, index=False)
    logging.info(f"Saved yearly NDVI aggregated data to {output_path}")


def aggregate_ndvi_by_period(ndvi_df: pd.DataFrame, grow_months: list) -> pd.DataFrame:
    df = ndvi_df[ndvi_df["month"].isin(grow_months)].copy()
    grouping_cols = ["state_fips", "county_fips", "county_name", "year"]
    grouped = df.groupby(grouping_cols, as_index=False).agg(
        ndvi_mean_year=("ndvi_mean", "mean"),
        ndvi_min_year=("ndvi_mean", "min"),
        ndvi_max_year=("ndvi_mean", "max"),
        evi_mean_year=("evi_mean", "mean"),
        evi_min_year=("evi_mean", "min"),
        evi_max_year=("evi_mean", "max")
    )

    return grouped


def download_ndvi(states, start_year, end_year, checkpoint_path: Path, monthly_data_path: Path, yearly_data_path: Path,
                  batch_size=10):
    counties = get_county_geometries(states)
    collection = ee.ImageCollection('MODIS/061/MOD13A3').select(['NDVI', 'EVI'])
    completed = load_checkpoint(checkpoint_path)

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            logging.info(f"Processing {year}-{month:02d} for all counties in batches")
            start = datetime(year, month, 1)
            end = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)
            image = collection.filterDate(start, end).first().multiply(0.0001)

            for i in range(0, len(counties), batch_size):
                batch = counties.iloc[i:i + batch_size]
                features = []
                for _, row in batch.iterrows():
                    county_fips = row.county_fips
                    state_fips = row.state_fips
                    county_name = row.county
                    if (county_fips, year, month) in completed:
                        continue
                    ee_geom = ee.Geometry(mapping(row.geometry))
                    features.append(ee.Feature(ee_geom, {
                        'state_fips': state_fips,
                        'county_fips': county_fips,
                        'county_name': county_name,
                        'year': year,
                        'month': month
                    }))

                if not features:
                    continue

                try:
                    fc = ee.FeatureCollection(features)
                    reduced = image.reduceRegions(
                        collection=fc,
                        reducer=ee.Reducer.mean(),
                        scale=1000,
                        tileScale=4
                    ).getInfo()

                    batch_results = []
                    for f in reduced['features']:
                        props = f['properties']
                        batch_results.append({
                            "state_fips": props['state_fips'],
                            "county_fips": props['county_fips'],
                            "county_name": props['county_name'],
                            "year": props['year'],
                            "month": props['month'],
                            "ndvi_mean": props.get("NDVI"),
                            "evi_mean": props.get("EVI")
                        })

                    df_batch = pd.DataFrame(batch_results)
                    if not df_batch.empty:
                        if monthly_data_path.exists():
                            df_batch.to_csv(monthly_data_path, mode="a", header=False, index=False)
                        else:
                            df_batch.to_csv(monthly_data_path, mode="w", header=True, index=False)

                    for f in batch_results:
                        update_checkpoint(checkpoint_path, f['county_fips'], f['year'], f['month'])

                except Exception as e:
                    logging.warning(f"Batch {i}-{i + batch_size} failed for {year}-{month:02d}: {e}")

    ndvi_mnthly_df = pd.read_csv(monthly_data_path)
    ndvi_yearly_df = aggregate_ndvi_by_period(ndvi_mnthly_df, grow_months=list(range(4, 10)))   # aggregate for growing months
    save_df(ndvi_yearly_df, yearly_data_path)


def main():
    parser = argparse.ArgumentParser(description="Download monthly NDVI and EVI data using Google Earth Engine")
    parser.add_argument("--states", nargs="+", default=["CA"], help="List of US state abbreviations")
    parser.add_argument("--start_year", type=int, default=2018)
    parser.add_argument("--end_year", type=int, default=2018)
    parser.add_argument("--batch_size", type=int, default=10, help="Number of counties to process per API call")
    args = parser.parse_args()

    ndvi_data_path_monthly = DATA_PATH / f"gee_ndvi_monthly_{args.start_year}_{args.end_year}.csv"
    ndvi_data_path_yearly = DATA_PATH / f"final_ndvi_data_Apr_Sep_{args.start_year}_{args.end_year}.csv"

    authenticate_gee()
    download_ndvi(
        args.states,
        args.start_year,
        args.end_year,
        checkpoint_path=Path(CHECKPOINT_PATH),
        monthly_data_path=ndvi_data_path_monthly,
        yearly_data_path=ndvi_data_path_yearly,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
