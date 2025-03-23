import logging
import os
import us
import pandas as pd
from typing import List

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def fetch_state_fips(state_alpha_list: List[str]) -> List[str]:
    return [us.states.lookup(abbr).fips for abbr in state_alpha_list]


def save_df(df: pd.DataFrame, file_name):
    extn = os.path.splitext(file_name)[1]

    if extn == ".csv":
        df.to_csv(file_name, index=False)
    elif extn == '.parquet':
        df.to_parquet(file_name)
    else:
        logging.error(f"Invalid file extension. Only .csv or .parquet are recognized")

    logging.info(f"File is saved at {file_name}")
