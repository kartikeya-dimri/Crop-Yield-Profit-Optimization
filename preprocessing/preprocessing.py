import pandas as pd
import numpy as np
import os

# --- Configuration ---
# Get absolute paths to avoid relative path issues
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = SCRIPT_DIR # Save output CSVs in the current 'preprocessing' folder

CROP_YIELD_FILE = os.path.join(DATA_DIR, 'crop_yield.csv')
MSP_FILE = os.path.join(DATA_DIR, 'minimum-support-prices.csv')

# --- 1. Preprocessing for Crop Yield Dataset (Yield Prediction) ---

def preprocess_crop_yield(df):
    """
    Handles transformations, encoding, and feature selection for the Crop Yield data.
    """
    print("Starting preprocessing for Crop Yield Dataset...")

    # 1. Feature Transformation (Handling Skewness and Outliers)
    # Apply log(1+x) to highly skewed features (Area, Production, Fertilizer, Pesticide)
    # This stabilizes variance and makes relationships more linear.
    skewed_cols = ['Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
    
    # Add a small constant (1) before log to handle zero values
    for col in skewed_cols:
        df[f'log_{col}'] = np.log1p(df[col])
        
    # The target variable (Yield) should also be transformed for OLS/Ridge/Lasso
    df['log_Yield'] = np.log1p(df['Yield'])
    
    # 2. Categorical Encoding (One-Hot Encoding)
    # Convert 'Crop', 'Season', and 'State' into numerical features
    categorical_cols = ['Crop_Year', 'Season', 'State', 'Crop']
    df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # 3. Final Feature Selection
    # Keep only the log-transformed features and the one-hot encoded columns
    # Drop the original numerical and Production columns
    cols_to_drop = skewed_cols + ['Yield', 'Production'] 
    
    # Keep 'Production' in the drop list, as we use log_Area, log_Production, and log_Yield.
    # Production is highly collinear with Area, but the log-transformed version might be useful.
    # For now, we keep the log versions of all features derived from the EDA.
    df_final = df_processed.drop(columns=cols_to_drop, errors='ignore')

    print(f"Crop Yield Data processed: Shape {df_final.shape}")
    return df_final


# --- 2. Preprocessing for MSP Dataset (Price Prediction) ---

def preprocess_msp(df):
    """
    Handles time-series feature engineering and encoding for the MSP data.
    """
    print("\nStarting preprocessing for MSP Dataset...")

    # 1. Feature Engineering (Extracting Time-Series Feature)
    # Convert the 'year' range (e.g., '2022-2023') into a single numerical feature (2022)
    df['start_year'] = df['year'].str.split('-').str[0].astype(int)
    
    # 2. Rename Target Variable
    df = df.rename(columns={'min_support_price': 'MSP'})
    
    # 3. Categorical Encoding (One-Hot Encoding)
    # Convert 'crop' and 'season' into numerical features
    categorical_cols = ['season', 'crop']
    df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # 4. Final Feature Selection
    # Drop the original categorical and ID/Year columns
    df_final = df_processed.drop(columns=['id', 'year'], errors='ignore')

    print(f"MSP Data processed: Shape {df_final.shape}")
    return df_final


# --- Main Execution ---

if __name__ == '__main__':
    try:
        # Load datasets
        df_yield = pd.read_csv(CROP_YIELD_FILE)
        df_msp = pd.read_csv(MSP_FILE)

        # Preprocess both
        df_yield_processed = preprocess_crop_yield(df_yield)
        df_msp_processed = preprocess_msp(df_msp)

        # Save processed files
        yield_output_path = os.path.join(OUTPUT_DIR, 'crop_yield_TRAIN.csv')
        msp_output_path = os.path.join(OUTPUT_DIR, 'msp_TRAIN.csv')
        
        df_yield_processed.to_csv(yield_output_path, index=False)
        df_msp_processed.to_csv(msp_output_path, index=False)

        print(f"\nSUCCESS: Processed Crop Yield data saved to: {yield_output_path}")
        print(f"SUCCESS: Processed MSP data saved to: {msp_output_path}")

    except FileNotFoundError as e:
        print(f"\nERROR: One or more data files were not found. Please ensure the path is correct: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")