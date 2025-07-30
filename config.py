import os
import logging
from dotenv import load_dotenv
from pathlib import Path

DATA_PATH = Path("../data")
COUNTY_SHAPEFILE = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_county_500k.zip"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# API Endpoints
NASS_BASE_URL = "https://quickstats.nass.usda.gov/api/api_GET/"
NOAA_BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2/"

load_dotenv()
# API Keys
NASS_API_KEY = os.getenv("NASS_API_KEY")
NOAA_TOKEN = os.getenv("NOAA_TOKEN")
SH_CLIENT_ID = os.getenv("SH_CLIENT_ID")
SH_CLIENT_SECRET = os.getenv("SH_CLIENT_SECRET")
GEE_PROJECT_ID=os.getenv("GEE_PROJECT_ID")

#logging.info(f"Loaded USDA API Key: {NASS_API_KEY[:5]}*****" if NASS_API_KEY else "USDA NASS API Key is not set")
#logging.info(f"Loaded NOAA API Key: {NOAA_TOKEN[:5]}*****" if NOAA_TOKEN else "NOAA Token is not set")
