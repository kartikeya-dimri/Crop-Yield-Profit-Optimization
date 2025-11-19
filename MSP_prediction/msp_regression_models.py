import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# --- Configuration ---
# Get absolute paths to avoid relative path issues
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PREPROCESSING_DIR = os.path.join(PROJECT_ROOT, 'preprocessing')
TRAIN_DATA_PATH = os.path.join(PREPROCESSING_DIR, 'msp_TRAIN.csv')

# --- 1. Load and Prepare Data ---

def load_and_split_data(path):
    """Loads the preprocessed MSP data and splits it into training/testing sets."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: MSP training file not found at {path}")
        return None, None, None, None
    
    # Define Target (y) and Features (X)
    Y = df['MSP'] 
    X = df.drop(columns=['MSP'])
    
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.dropna(inplace=True)
    Y = Y[X.index] # Align Y index after dropping NaNs
    
    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    return X_train, X_test, Y_train, Y_test

# --- 2. Train and Evaluate Models ---

def train_and_evaluate(X_train, X_test, Y_train, Y_test):
    
    models = {
        "OLS": LinearRegression(),
        "Ridge": GridSearchCV(Ridge(), param_grid={'alpha': np.logspace(-3, 1, 100)}, scoring='r2', cv=5),
        "Lasso": GridSearchCV(Lasso(max_iter=5000), param_grid={'alpha': np.logspace(-3, 1, 100)}, scoring='r2', cv=5)
    }

    results = {}
    best_model = None
    best_r2 = -float('inf')
    
    print("--- Training and Evaluation of MSP Regression Models ---")
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        if name in ["Ridge", "Lasso"]:
            model.fit(X_train, Y_train)
            final_model = model.best_estimator_
            print(f"Best Alpha for {name}: {model.best_params_['alpha']:.4f}")
        else:
            model.fit(X_train, Y_train)
            final_model = model
            
        Y_pred = final_model.predict(X_test)
        
        r2 = r2_score(Y_test, Y_pred)
        rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
        
        print(f"{name} R-squared: {r2:.4f}")
        print(f"{name} RMSE: {rmse:.2f}")
        
        results[name] = {"R2": r2, "RMSE": rmse, "model": final_model}
        
        # Track the best model based on R-squared
        if r2 > best_r2:
            best_r2 = r2
            best_model = name
            
    print("\n" + "="*50)
    print(f"BEST PERFORMING MODEL: {best_model} (R2: {best_r2:.4f})")
    print("="*50)
    
    # Save the best model
    model_save_path = os.path.join(SCRIPT_DIR, 'best_msp_model.pkl')
    with open(model_save_path, 'wb') as file:
        pickle.dump(results[best_model]["model"], file)
    print(f"The best model ({best_model}) saved as: {model_save_path}")

# --- Main Execution ---

if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = load_and_split_data(TRAIN_DATA_PATH)
    
    if X_train is not None:
        train_and_evaluate(X_train, X_test, Y_train, Y_test)