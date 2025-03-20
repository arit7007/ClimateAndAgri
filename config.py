# Keeps the configuration parameters used across programs

import os
import logging
from dotenv import load_dotenv

DATA_PATH = "../data/"
CLIMATE_DATA = DATA_PATH + "climate_data.csv"
CLIMATE_DATA_WITH_FIPS = DATA_PATH + "climate_data_with_fips.csv"
CROP_YIELD_DATA = DATA_PATH + "crop_yield_data.csv"
CLIMATE_YIELD_DATA = DATA_PATH + "climate_and_yield_data.csv"

logging.basicConfig(level=logging.WARN, format="%(asctime)s - %(levelname)s - %(message)s")

# API Endpoints
NASS_BASE_URL = "https://quickstats.nass.usda.gov/api/api_GET/"
NOAA_BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2/"

load_dotenv()
# API Keys
NASS_API_KEY = os.getenv("NASS_API_KEY")
NOAA_TOKEN = os.getenv("NOAA_TOKEN")

#logging.info(f"Loaded USDA API Key: {NASS_API_KEY[:5]}*****" if NASS_API_KEY else "USDA NASS API Key is not set")
#logging.info(f"Loaded NOAA API Key: {NOAA_TOKEN[:5]}*****" if NOAA_TOKEN else "NOAA Token is not set")

