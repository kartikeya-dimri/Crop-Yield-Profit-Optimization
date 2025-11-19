import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
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
    """Loads the preprocessed yield data and splits it into training/testing sets."""
    try:
        print(f"Loading data from: {path}")
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: Training file not found at {path}")
        return None, None, None, None

    # Define Target (y) and Features (X)
    Y = df['log_Yield'] 
    X = df.drop(columns=['log_Yield']) 
    
    # Clean data
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.dropna(inplace=True)
    Y = Y[X.index] # Align Y index after dropping NaNs
    
    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    return X_train, X_test, Y_train, Y_test

# --- 2. Train and Evaluate Models (Optimized) ---

def train_and_evaluate(X_train, X_test, Y_train, Y_test):
    
    print("--- Training and Evaluation of Advanced Yield Regression Models ---")
    print("(Using optimized Cross-Validation estimators for speed)")

    # Define alphas to test
    alphas_to_test = np.logspace(-3, 1, 20)

    # 1. Ridge Regression (RidgeCV)
    # RidgeCV is extremely fast because it uses efficient linear algebra 
    print("\nTraining Ridge (Optimized)...")
    ridge_model = RidgeCV(alphas=alphas_to_test, scoring='r2', cv=5)
    ridge_model.fit(X_train, Y_train)
    
    y_pred_ridge = ridge_model.predict(X_test)
    r2_ridge = r2_score(Y_test, y_pred_ridge)
    rmse_ridge = np.sqrt(mean_squared_error(Y_test, y_pred_ridge))
    
    print(f"Ridge R-squared (Log Scale): {r2_ridge:.4f}")
    print(f"Ridge RMSE (Log Scale): {rmse_ridge:.4f}")
    print(f"Best Alpha: {ridge_model.alpha_:.4f}")

    # 2. Lasso Regression (LassoCV)
    # LassoCV uses coordinate descent on the regularization path, much faster than GridSearch
    print("\nTraining Lasso (Optimized)...")
    lasso_model = LassoCV(alphas=alphas_to_test, cv=5, random_state=42, n_jobs=-1)
    lasso_model.fit(X_train, Y_train)
    
    y_pred_lasso = lasso_model.predict(X_test)
    r2_lasso = r2_score(Y_test, y_pred_lasso)
    rmse_lasso = np.sqrt(mean_squared_error(Y_test, y_pred_lasso))
    
    print(f"Lasso R-squared (Log Scale): {r2_lasso:.4f}")
    print(f"Lasso RMSE (Log Scale): {rmse_lasso:.4f}")
    print(f"Best Alpha: {lasso_model.alpha_:.4f}")

    # 3. ElasticNet (ElasticNetCV)
    print("\nTraining ElasticNet (Optimized)...")
    # l1_ratio corresponds to the mix between L1 and L2 regularization
    enet_model = ElasticNetCV(alphas=alphas_to_test, l1_ratio=[0.1, 0.5, 0.9], cv=5, random_state=42, n_jobs=-1)
    enet_model.fit(X_train, Y_train)
    
    y_pred_enet = enet_model.predict(X_test)
    r2_enet = r2_score(Y_test, y_pred_enet)
    rmse_enet = np.sqrt(mean_squared_error(Y_test, y_pred_enet))
    
    print(f"ElasticNet R-squared (Log Scale): {r2_enet:.4f}")
    print(f"ElasticNet RMSE (Log Scale): {rmse_enet:.4f}")
    print(f"Best Alpha: {enet_model.alpha_:.4f}")

    return {
        "Ridge": {"R2": r2_ridge, "RMSE": rmse_ridge, "model": ridge_model},
        "Lasso": {"R2": r2_lasso, "RMSE": rmse_lasso, "model": lasso_model},
        "ElasticNet": {"R2": r2_enet, "RMSE": rmse_enet, "model": enet_model}
    }

# --- Main Execution ---

if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = load_and_split_data(TRAIN_DATA_PATH)
    
    if X_train is not None:
        train_and_evaluate(X_train, X_test, Y_train, Y_test)