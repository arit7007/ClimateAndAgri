# Climate-Aware Crop Yield Prediction Using Machine Learning

This project predicts U.S. county-level crop yields using climate, soil, and satellite-derived vegetation data. Built with open datasets (NOAA, USDA NASS, MODIS, SoilGrids), this modular pipeline uses machine learning to study both *what drives crop yield* and *how to forecast it accurately*. For the initial analysis, a uniform growing season from April to September was applied across all crops, without incorporating crop-specific growing periods.

---

## Project Goals

- Understand which **environmental conditions** (climate, soil) impact specific crops
- Use **NDVI/EVI** as predictive signals for in-season plant health
- Develop a **scalable pipeline** for yield forecasting and agricultural insight
- Visualize model performance across crops using Random Forests and feature importance

---

## Project Structure

### Core Scripts

| File | Description |
|------|-------------|
| `ml.py` | Main ML pipeline: builds Random Forest models, computes R²/MAE, ranks features per crop |
| `config.py` | Central config: paths, API keys, logging format |
| `utils.py` | Shared utilities: save CSVs, get FIPS codes, logging setup |

### Climate Data

| File | Description |
|------|-------------|
| `download_climate_data.py` | Downloads NOAA GSOM climate data, maps stations to counties, and aggregates monthly weather stats (e.g. PRCP, TMAX, HTDD) to the growing season |
| `climate_utils.py` | Helper functions for NOAA API calls, spatial joins, pivoting monthly weather data, and crop-specific growing month lookup |

### NDVI Data

| File | Description |
|------|-------------|
| `download_ndvi_data.py` | Uses Google Earth Engine (MODIS/061/MOD13A3) to download NDVI/EVI by county and year. Aggregates April–Sept values and checkpoints progress |

### Soil Data

| File | Description |
|------|-------------|
| `download_soil_data.py` | Uses ISRIC SoilGrids to extract surface-level properties (clay, silt, pH, SOC, bulk density) per county. Parallelized with `ThreadPoolExecutor` and includes checkpointing |

### Yield Data

| File | Description |
|------|-------------|
| `download_yield_data.py` | Downloads USDA NASS crop yield data at the county level. |

---

## Sample Findings

- **NDVI/EVI** improves *prediction accuracy* but doesn’t explain causality.
- **Soil pH and bulk density** are key drivers of wheat yield.
- **Corn and soybeans** respond well to vegetation indices, suggesting NDVI is a strong proxy for in-season health.
- Models using **all features** (climate + soil + NDVI) consistently outperform climate-only baselines.

---

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt

2. Set your .env:

   ```bash
   NOAA_TOKEN=your_noaa_token
   NASS_API_KEY=your_nass_key
   SH_CLIENT_ID=your_sentinelhub_id
   SH_CLIENT_SECRET=your_sentinelhub_secret
   GEE_PROJECT_ID=your_gee_project_id

3. Download various data:

   ```bash
   python download_climate_data.py --states CA IA IL --start_year 2010 --end_year 2014
   python download_yield_data.py --states CA IA IL --crops CORN WHEAT --start_year 2010 --end_year 2014
   python download_ndvi_data.py --states CA IA IL --start_year 2010 --end_year 2014
   python download_soil_data.py --states CA IA IL

4. Combine the final data from the above programs and run the ml:

   ```bash
   python ml.py

## Output Files
model_performance_summary.csv — R² and MAE scores per crop and feature set

feature_importance_by_crop.csv — Ranked top features for each crop

r2_comparison_chart.png — Bar chart comparing model performance (e.g., climate-only vs. all-features)

## Packages used
   ```bash
   pandas
   pygris
   geopandas
   tenacity
   scikit-learn
```

### Data Sources

- [USDA NASS Quick Stats API](https://quickstats.nass.usda.gov/)
- [NOAA GSOM Climate Data](https://www.ncei.noaa.gov/products/land-based-station/global-summary-of-the-month)
- [MODIS NDVI via Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD13A3)
- [ISRIC SoilGrids](https://soilgrids.org/)

## Authors
Arit Prince – Student, American High School

Arya Prince – Data Science Major, UC Berkeley

## Contact
Feel free to open an issue or contact us directly if you're interested in collaboration or would like to expand this work to other crops or geographies.

> Disclaimer: Portions of the code have been generated or optimized using vibe code
