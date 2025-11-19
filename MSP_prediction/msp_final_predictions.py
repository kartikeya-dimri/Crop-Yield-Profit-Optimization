import pandas as pd
import numpy as np
import os
import pickle

# --- Configuration ---
# Get absolute paths to avoid relative path issues
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PREPROCESSING_DIR = os.path.join(PROJECT_ROOT, 'preprocessing')
TRAIN_DATA_PATH = os.path.join(PREPROCESSING_DIR, 'msp_TRAIN.csv')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'best_msp_model.pkl') # Saved by msp_regression_models.py
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'Final_Predictions')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'msp_final_predictions.csv')

# --- 1. Load Data and Model ---

def generate_final_predictions():
    """Loads the best model and full training data to generate predictions."""
    print("--- Generating Final MSP Predictions ---")
    
    try:
        # Load the full processed data
        df_train = pd.read_csv(TRAIN_DATA_PATH)
        
        # Load the best trained model
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
            
    except FileNotFoundError as e:
        print(f"Error: Required file not found: {e}")
        return

    # Prepare features (X) using the same logic as training
    X_full = df_train.drop(columns=['MSP'])
    
    # Predict on the full dataset
    df_train['Predicted_MSP'] = model.predict(X_full)
    
    # Create the output DataFrame
    # Keeping the actual MSP for comparison/model integration, and the predicted value
    df_output = df_train[['Predicted_MSP']].copy()
    
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Save the final predictions CSV
    df_output.to_csv(OUTPUT_FILE, index=False)
    
    print("\nSUCCESS: Final MSP predictions saved.")
    print(f"Output file: {OUTPUT_FILE}")

# --- Main Execution ---

if __name__ == '__main__':
    generate_final_predictions()