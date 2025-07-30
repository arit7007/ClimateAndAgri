import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

from config import DATA_PATH
from utils import save_df, logging

MIN_SAMPLES_PER_CROP = 1000
MIN_PERCENT_OF_TOTAL = 0.05
SAVE_FEATURE_IMPORTANCE = True

# Feature of interest
climate_features = ['TMIN', 'TMAX', 'PRCP', 'SNOW', 'HTDD', 'CLDD', 'EMXT', 'EMNT']
soil_features = ['clay_mean', 'silt_mean', 'sand_mean', 'ph_mean', 'soc_mean', 'bdod_mean']
ndvi_features = [
    'ndvi_mean_year', 'ndvi_min_year', 'ndvi_max_year',
    'evi_mean_year', 'evi_min_year', 'evi_max_year'
]

feature_sets = {
    'Climate': climate_features,
    'Climate + NDVI': climate_features + ndvi_features,
    'Climate + Soil': climate_features + soil_features,
    'All': climate_features + soil_features + ndvi_features
}


def run_random_forest_by_crop(df, crops, feature_cols):
    results = {}
    importance_rows = []

    for crop in crops:
        crop_df = df[df['commodity_desc'] == crop].dropna(subset=['YIELD'])
        if len(crop_df) < MIN_SAMPLES_PER_CROP:
            continue

        X = crop_df[feature_cols].fillna(0)
        y = crop_df['YIELD']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        importances = dict(zip(feature_cols, model.feature_importances_))
        top_features = sorted(importances.items(), key=lambda x: -x[1])[:5]
        logging.info(f"{crop:<12} | Top features: {top_features}")

        results[crop] = {
            'R2': r2_score(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'Importances': importances
        }

        for feat, score in importances.items():
            importance_rows.append({'Crop': crop, 'Feature': feat, 'Importance': score})

    if SAVE_FEATURE_IMPORTANCE:
        feature_df = pd.DataFrame(importance_rows)
        save_df(feature_df, DATA_PATH / "feature_importance_by_crop.csv")

    return results


def plot_r2_comparison(results_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.2
    crops = results_df['Crop']
    x = np.arange(len(crops))

    ax.bar(x - bar_width, results_df['R2_Climate'], width=bar_width, label='Climate')
    ax.bar(x, results_df['R2_NDVI_Only'], width=bar_width, label='+NDVI')
    ax.bar(x + bar_width, results_df['R2_Soil'], width=bar_width, label='+Soil')
    ax.bar(x + 2 * bar_width, results_df['R2_All'], width=bar_width, label='All')

    ax.set_ylabel("R² Score")
    ax.set_title("Crop Yield Prediction: R² Comparison")
    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(crops)
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(DATA_PATH / "r2_comparison_chart.png")
    plt.close()


def main():
    # Read the climate data
    climate_df = pd.read_csv(DATA_PATH / "final_climate_data.csv")

    # Read the crop yield data
    yield_df = pd.read_csv(DATA_PATH / "final_crop_yield_data.csv")
    yield_df = yield_df[(yield_df['YIELD'] > 0)].copy()

    # Read the soil data
    soil_df = pd.read_csv(DATA_PATH / "county_soil_features.csv")
    soil_df = soil_df[(soil_df['ph_mean'] > 0) & (soil_df['ph_mean'] < 14) &
                      (soil_df['clay_mean'] > 0) & (soil_df['sand_mean'] > 0)].copy()  # Filter out illogical data

    # Read the vegetation index data
    ndvi_df = pd.read_csv(DATA_PATH / "final_ndvi_data.csv")

    # Though source data set has county_fips as 5 digits, format it again before joining
    for df_ in [climate_df, yield_df, soil_df, ndvi_df]:
        df_['county_fips'] = df_['county_fips'].astype(float).astype(int).astype(str).str.zfill(5)

    climate_df[climate_df.columns.difference(["county_fips", "year"])] = \
        climate_df[climate_df.columns.difference(["county_fips", "year"])].apply(pd.to_numeric, errors="coerce")

    df = (climate_df
          .merge(yield_df, on=['county_fips', 'year'], how='inner')
          .merge(soil_df.drop_duplicates('county_fips'), on='county_fips', how='left')
          .merge(ndvi_df[ndvi_features + ['county_fips', 'year']].drop_duplicates(['county_fips', 'year']),
                 on=['county_fips', 'year'], how='left'))

    drop_cols = ['TSUN', 'AWND', 'RHAV', 'cv_mean', 'unit_desc', 'class_desc',
                 'prodn_practice_desc', 'util_practice_desc', 'county_name',
                 'state_abbr', 'state_fips']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    save_df(df, DATA_PATH / "cleaned_model_data.csv")

    # Filter only the crops with enough representation
    total_rows = len(df)
    crop_counts = df['commodity_desc'].value_counts()

    filtered_crops = [crop for crop, count in crop_counts.items()
                      if count >= MIN_SAMPLES_PER_CROP and count / total_rows >= MIN_PERCENT_OF_TOTAL]

    logging.info(f"Crops selected for modeling: {filtered_crops}")

    all_results = {}
    for label, features in feature_sets.items():
        logging.info(f"=== Running model: {label} ===")
        all_results[label] = run_random_forest_by_crop(df, filtered_crops, features)

    results_df = pd.DataFrame([{
        'Crop': crop,
        'n_samples': crop_counts[crop],
        'R2_Climate': all_results['Climate'].get(crop, {}).get('R2'),
        'MAE_Climate': all_results['Climate'].get(crop, {}).get('MAE'),
        'R2_NDVI_Only': all_results['Climate + NDVI'].get(crop, {}).get('R2'),
        'MAE_NDVI_Only': all_results['Climate + NDVI'].get(crop, {}).get('MAE'),
        'R2_Soil': all_results['Climate + Soil'].get(crop, {}).get('R2'),
        'MAE_Soil': all_results['Climate + Soil'].get(crop, {}).get('MAE'),
        'R2_All': all_results['All'].get(crop, {}).get('R2'),
        'MAE_All': all_results['All'].get(crop, {}).get('MAE'),
    } for crop in filtered_crops])

    results_df['ΔR2_NDVI'] = results_df['R2_NDVI_Only'] - results_df['R2_Climate']
    results_df['ΔR2_Soil'] = results_df['R2_Soil'] - results_df['R2_Climate']
    results_df['ΔR2_All'] = results_df['R2_All'] - results_df['R2_Climate']

    save_df(results_df, DATA_PATH / "model_performance_summary.csv")
    plot_r2_comparison(results_df)


if __name__ == "__main__":
    main()
