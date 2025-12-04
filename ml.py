"""
ml.py - Crop Yield Prediction with Random Forest

Author: Arit Prince
Updated: December 2024

This script trains Random Forest models to predict crop yields using
climate, soil, and vegetation features. Uses spatial cross-validation
(GroupKFold by county) to ensure proper evaluation of generalization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

from config import DATA_PATH
from utils import save_df, logging

# === Configuration ===
MIN_SAMPLES_PER_CROP = 1000
MIN_PERCENT_OF_TOTAL = 0.05
N_ESTIMATORS = 100
N_SPLITS = 5  # For cross-validation
RANDOM_STATE = 42

# === Feature Definitions ===
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


def get_available_features(df, feature_list):
    """Return only features that exist in the dataframe."""
    available = [f for f in feature_list if f in df.columns]
    missing = set(feature_list) - set(available)
    if missing:
        logging.warning(f"Missing features: {missing}")
    return available


def run_spatial_cv(df, crop, feature_cols, n_splits=N_SPLITS):
    """
    Spatial cross-validation using GroupKFold.
    
    Ensures NO county appears in both training and test sets.
    This prevents the model from memorizing county-specific patterns.
    
    Args:
        df: DataFrame with features and target
        crop: Crop name to filter
        feature_cols: List of feature column names
        n_splits: Number of CV folds
    
    Returns:
        dict with R2_mean, R2_std, MAE_mean, feature_importances
    """
    crop_df = df[df['commodity_desc'] == crop].dropna(subset=['YIELD']).copy()
    
    if len(crop_df) < MIN_SAMPLES_PER_CROP:
        logging.warning(f"Skipping {crop}: only {len(crop_df)} samples")
        return None
    
    available_features = get_available_features(crop_df, feature_cols)
    if not available_features:
        logging.warning(f"No features available for {crop}")
        return None
    
    X = crop_df[available_features].copy()
    y = crop_df['YIELD'].copy()
    groups = crop_df['county_fips']
    
    # Use median imputation for missing values
    X = X.fillna(X.median())
    
    # Spatial cross-validation
    gkf = GroupKFold(n_splits=n_splits)
    
    r2_scores = []
    mae_scores = []
    all_importances = []
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS, 
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test_scaled)
        r2_scores.append(r2_score(y_test, y_pred))
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        all_importances.append(model.feature_importances_)
    
    # Average across folds
    mean_importances = np.mean(all_importances, axis=0)
    importances_dict = dict(zip(available_features, mean_importances))
    
    return {
        'R2': np.mean(r2_scores),
        'R2_std': np.std(r2_scores),
        'MAE': np.mean(mae_scores),
        'MAE_std': np.std(mae_scores),
        'Importances': importances_dict,
        'n_samples': len(crop_df),
        'n_counties': crop_df['county_fips'].nunique()
    }


def plot_r2_comparison(results_df, output_name="r2_comparison_chart.png"):
    """Plot R² comparison across feature sets."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bar_width = 0.2
    crops = results_df['Crop'].tolist()
    x = np.arange(len(crops))
    
    colors = ['#1E90FF', '#FFA500', '#228B22', '#FF6B6B']
    labels = ['Climate', '+ NDVI', '+ Soil', 'All Features']
    
    for i, (col, color, label) in enumerate(zip(
        ['R2_Climate', 'R2_NDVI', 'R2_Soil', 'R2_All'],
        colors, labels
    )):
        values = results_df[col].fillna(0)
        ax.bar(x + i * bar_width, values, width=bar_width, 
               label=label, color=color, alpha=0.8)
    
    ax.set_ylabel("R² Score", fontsize=12)
    ax.set_xlabel("Crop", fontsize=12)
    ax.set_title("Crop Yield Prediction: R² by Feature Set\n(5-Fold Spatial Cross-Validation)", 
                 fontsize=14)
    ax.set_xticks(x + bar_width * 1.5)
    ax.set_xticklabels(crops, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, col in enumerate(['R2_Climate', 'R2_NDVI', 'R2_Soil', 'R2_All']):
        for j, val in enumerate(results_df[col].fillna(0)):
            if val > 0:
                ax.text(j + i * bar_width, val + 0.02, f'{val:.2f}', 
                       ha='center', va='bottom', fontsize=8, rotation=90)
    
    plt.tight_layout()
    plt.savefig(DATA_PATH / output_name, dpi=150)
    plt.close()
    logging.info(f"Saved plot to {DATA_PATH / output_name}")


def main():
    """Main pipeline: load data, run models, save results."""
    
    logging.info("="*70)
    logging.info("CROP YIELD PREDICTION WITH SPATIAL CROSS-VALIDATION")
    logging.info("="*70)
    
    # === Load Data ===
    logging.info("\nLoading data files...")
    
    climate_df = pd.read_csv(DATA_PATH / "final_climate_data_Apr_Sep_2010_2024.csv")
    yield_df = pd.read_csv(DATA_PATH / "final_crop_yield_data_2010_2024.csv")
    yield_df = yield_df[yield_df['YIELD'] > 0].copy()
    
    soil_df = pd.read_csv(DATA_PATH / "final_soil_features.csv")
    soil_df = soil_df[
        (soil_df['ph_mean'] > 0) & (soil_df['ph_mean'] < 14) &
        (soil_df['clay_mean'] > 0) & (soil_df['sand_mean'] > 0)
    ].copy()
    
    ndvi_df = pd.read_csv(DATA_PATH / "final_ndvi_data_Apr_Sep_2010_2024.csv")
    
    # Standardize FIPS codes
    for df_ in [climate_df, yield_df, soil_df, ndvi_df]:
        df_['county_fips'] = df_['county_fips'].astype(float).astype(int).astype(str).str.zfill(5)
    
    # Convert climate columns to numeric
    climate_numeric_cols = climate_df.columns.difference(['county_fips', 'year'])
    climate_df[climate_numeric_cols] = climate_df[climate_numeric_cols].apply(
        pd.to_numeric, errors='coerce'
    )
    
    # === Merge All Data Sources ===
    logging.info("Merging data sources...")
    
    df = (climate_df
          .merge(yield_df, on=['county_fips', 'year'], how='inner')
          .merge(soil_df.drop_duplicates('county_fips'), on='county_fips', how='left')
          .merge(ndvi_df[ndvi_features + ['county_fips', 'year']].drop_duplicates(['county_fips', 'year']),
                 on=['county_fips', 'year'], how='left'))
    
    # Drop unused columns
    drop_cols = ['TSUN', 'AWND', 'RHAV', 'cv_mean', 'unit_desc', 'class_desc',
                 'prodn_practice_desc', 'util_practice_desc', 'county_name',
                 'state_abbr', 'state_fips']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)
    
    logging.info(f"Final dataset: {len(df)} rows, {df['county_fips'].nunique()} counties")
    save_df(df, DATA_PATH / "cleaned_model_data.csv")
    
    # === Filter Crops ===
    total_rows = len(df)
    crop_counts = df['commodity_desc'].value_counts()
    
    filtered_crops = [
        crop for crop, count in crop_counts.items()
        if count >= MIN_SAMPLES_PER_CROP and count / total_rows >= MIN_PERCENT_OF_TOTAL
    ]
    
    logging.info(f"\nCrops selected: {filtered_crops}")
    logging.info(f"Sample sizes: {dict(crop_counts[filtered_crops])}")
    
    # === Run Spatial CV for Each Feature Set ===
    results = {label: {} for label in feature_sets}
    importance_rows = []
    
    for label, features in feature_sets.items():
        logging.info(f"\n--- Feature Set: {label} ---")
        for crop in filtered_crops:
            result = run_spatial_cv(df, crop, features)
            if result:
                results[label][crop] = result
                logging.info(f"{crop:<12} R² = {result['R2']:.3f} ± {result['R2_std']:.3f}")
                
                # Save feature importances for 'All' features
                if label == 'All':
                    for feat, score in result['Importances'].items():
                        importance_rows.append({
                            'Crop': crop,
                            'Feature': feat,
                            'Importance': score
                        })
    
    # === Save Results ===
    results_rows = []
    for crop in filtered_crops:
        row = {
            'Crop': crop,
            'n_samples': crop_counts[crop],
            'R2_Climate': results['Climate'].get(crop, {}).get('R2'),
            'R2_Climate_std': results['Climate'].get(crop, {}).get('R2_std'),
            'R2_NDVI': results['Climate + NDVI'].get(crop, {}).get('R2'),
            'R2_NDVI_std': results['Climate + NDVI'].get(crop, {}).get('R2_std'),
            'R2_Soil': results['Climate + Soil'].get(crop, {}).get('R2'),
            'R2_Soil_std': results['Climate + Soil'].get(crop, {}).get('R2_std'),
            'R2_All': results['All'].get(crop, {}).get('R2'),
            'R2_All_std': results['All'].get(crop, {}).get('R2_std'),
            'MAE_All': results['All'].get(crop, {}).get('MAE'),
        }
        results_rows.append(row)
    
    results_df = pd.DataFrame(results_rows)
    
    # Calculate improvements
    results_df['Delta_R2_NDVI'] = results_df['R2_NDVI'] - results_df['R2_Climate']
    results_df['Delta_R2_Soil'] = results_df['R2_Soil'] - results_df['R2_Climate']
    results_df['Delta_R2_All'] = results_df['R2_All'] - results_df['R2_Climate']
    
    save_df(results_df, DATA_PATH / "model_performance_summary.csv")
    logging.info(f"\nSaved results to model_performance_summary.csv")
    
    # Save feature importances
    if importance_rows:
        importance_df = pd.DataFrame(importance_rows)
        save_df(importance_df, DATA_PATH / "feature_importance_by_crop.csv")
        logging.info(f"Saved feature importances to feature_importance_by_crop.csv")
    
    # Generate plot
    plot_r2_comparison(results_df)
    
    # === Print Summary ===
    logging.info("\n" + "="*70)
    logging.info("RESULTS SUMMARY")
    logging.info("="*70)
    
    print("\n{:<12} {:>8} {:>10} {:>10} {:>10} {:>10}".format(
        'Crop', 'N', 'Climate', '+NDVI', '+Soil', 'All'))
    print("-" * 62)
    
    for _, row in results_df.iterrows():
        print("{:<12} {:>8} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f}".format(
            row['Crop'],
            row['n_samples'],
            row['R2_Climate'] or 0,
            row['R2_NDVI'] or 0,
            row['R2_Soil'] or 0,
            row['R2_All'] or 0
        ))
    
    print("-" * 62)
    print("{:<12} {:>8} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f}".format(
        'AVERAGE', '',
        results_df['R2_Climate'].mean(),
        results_df['R2_NDVI'].mean(),
        results_df['R2_Soil'].mean(),
        results_df['R2_All'].mean()
    ))
    
    # Print improvement summary
    avg_climate = results_df['R2_Climate'].mean()
    avg_all = results_df['R2_All'].mean()
    improvement = (avg_all - avg_climate) / avg_climate * 100
    
    print(f"\n📊 Key Finding:")
    print(f"   Climate-only R²: {avg_climate:.2f}")
    print(f"   All features R²: {avg_all:.2f}")
    print(f"   Improvement: +{improvement:.0f}%")
    
    logging.info("\n✅ Pipeline complete!")


if __name__ == "__main__":
    main()
