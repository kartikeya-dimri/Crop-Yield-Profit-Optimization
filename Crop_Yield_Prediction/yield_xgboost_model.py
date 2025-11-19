import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# --- Configuration ---
# Get absolute paths to avoid relative path issues
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PREPROCESSING_DIR = os.path.join(PROJECT_ROOT, 'preprocessing')
TRAIN_DATA_PATH = os.path.join(PREPROCESSING_DIR, 'crop_yield_TRAIN.csv')

# --- 1. Load and Prepare Data ---
def load_and_split_data(path):
    try:
        print(f"Loading data from: {path}")
        df = pd.read_csv(path)
        Y = df['log_Yield'] 
        X = df.drop(columns=['log_Yield']) 
        
        # Clean data
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.dropna(inplace=True)
        Y = Y[X.index] 
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        return X_train, X_test, Y_train, Y_test
    except FileNotFoundError:
        print(f"Error: Training file not found at {path}")
        return None, None, None, None

# --- 2. Train, Tune, and Evaluate XGBoost ---

def train_and_tune_xgboost(X_train, X_test, Y_train, Y_test):
    """Trains XGBoost with Hyperparameter Tuning using GridSearchCV."""
    print("\n--- Tuning and Training XGBoost Regression Model ---")
    
    # Define the Hyperparameter Grid
    # These are the most important parameters to tune for XGBoost
    param_grid = {
        'n_estimators': [100, 200],      # Number of trees
        'learning_rate': [0.05, 0.1],    # Step size shrinkage
        'max_depth': [3, 5, 7],          # Maximum depth of a tree (controls complexity)
        'subsample': [0.8, 1.0],         # Fraction of samples used per tree
        'colsample_bytree': [0.8, 1.0]   # Fraction of features used per tree
    }

    # Initialize base model
    xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

    # Initialize GridSearchCV
    # n_jobs=-1 uses all available processors to speed up training
    # verbose=1 shows progress updates
    grid_search = GridSearchCV(
        estimator=xgb, 
        param_grid=param_grid, 
        scoring='r2', 
        cv=3, 
        n_jobs=-1, 
        verbose=1
    )

    print("Starting Grid Search (This may take a few minutes)...")
    grid_search.fit(X_train, Y_train)

    # Get the best model found
    best_model = grid_search.best_estimator_
    print(f"\nBest Parameters Found: {grid_search.best_params_}")
    
    # Predict and evaluate using the best model
    Y_pred = best_model.predict(X_test)
    
    r2 = r2_score(Y_test, Y_pred)
    mse = mean_squared_error(Y_test, Y_pred)
    rmse = np.sqrt(mse)
    
    print(f"Best XGBoost R-squared (Log Scale): {r2:.4f}")
    print(f"Best XGBoost RMSE (Log Scale): {rmse:.4f}")
    
    # Save the best model
    model_path = os.path.join(SCRIPT_DIR, 'xgb_yield_model.pkl')
    with open(model_path, 'wb') as file:
        pickle.dump(best_model, file)
    print(f"Best model saved to: {model_path}")
    
    return {"XGBoost": {"R2": r2, "RMSE": rmse, "model": best_model}}

# --- Main Execution ---

if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = load_and_split_data(TRAIN_DATA_PATH)
    
    if X_train is not None:
        train_and_tune_xgboost(X_train, X_test, Y_train, Y_test)