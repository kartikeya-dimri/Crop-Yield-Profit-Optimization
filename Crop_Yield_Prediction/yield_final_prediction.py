import pandas as pd
import numpy as np
import os
import pickle
from xgboost import XGBRegressor # Assuming XGBoost is the best model

# --- Configuration ---
# Get absolute paths to avoid relative path issues
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PREPROCESSING_DIR = os.path.join(PROJECT_ROOT, 'preprocessing')
TRAIN_DATA_PATH = os.path.join(PREPROCESSING_DIR, 'crop_yield_TRAIN.csv')

# Output Configuration
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'Final_Predictions')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'crop_yield_final_predictions.csv')

# --- 1. Load Data and Train Final Model on 100% Data ---

def train_final_model_and_predict():
    """Trains the chosen model on 100% of the data and generates predictions."""
    print("--- Training Final Yield Model on 100% Data ---")
    
    try:
        # Load the full processed data
        print(f"Loading data from: {TRAIN_DATA_PATH}")
        df_full = pd.read_csv(TRAIN_DATA_PATH)
            
    except FileNotFoundError as e:
        print(f"Error: Required file not found: {e}")
        return

    # Define Target (y) and Features (X)
    Y_full = df_full['log_Yield'] 
    X_full = df_full.drop(columns=['log_Yield']) 
    
    # Clean data
    X_full.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_full.dropna(inplace=True)
    Y_full = Y_full[X_full.index]

    # --- Retrain the best model (Example: XGBoost) on 100% of the data ---
    final_model = XGBRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, objective='reg:squarederror'
    )
    final_model.fit(X_full, Y_full)
    
    print("Final model (XGBoost) trained on 100% of historical data.")
    
    # Predict on the entire dataset
    df_full['Predicted_log_Yield'] = final_model.predict(X_full)
    
    # Inverse Transform: Predicted Yield = exp(Predicted_log_Yield) - 1
    df_full['Predicted_Yield_ton_ha'] = np.expm1(df_full['Predicted_log_Yield'])

    # Create the output DataFrame
    df_output = df_full[['Predicted_Yield_ton_ha']].copy()
    
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Save the final predictions CSV
    df_output.to_csv(OUTPUT_FILE, index=False)
    
    print("\nSUCCESS: Final Yield predictions saved.")
    print(f"Output file: {OUTPUT_FILE}")

# --- Main Execution ---

if __name__ == '__main__':
    train_final_model_and_predict()