import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set file paths (get script directory and navigate to data folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
CROP_YIELD_PATH = os.path.join(DATA_DIR, 'crop_yield.csv')
MSP_PATH = os.path.join(DATA_DIR, 'minimum-support-prices.csv')

# Set output directory for plots
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'eda_outputs')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- SECTION 1: CROP YIELD DATASET EDA ---

def analyze_crop_yield_data():
    """Performs EDA on the Crop Yield dataset."""
    print("="*50)
    print("STARTING EDA for CROP YIELD Dataset")
    print("="*50)
    
    try:
        df = pd.read_csv(CROP_YIELD_PATH)
    except FileNotFoundError:
        print(f"Error: Crop Yield file not found at {CROP_YIELD_PATH}")
        return

    # Descriptive Statistics
    numerical_cols = ['Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield']
    print("\n--- 1. Descriptive Statistics ---")
    print(df[numerical_cols].describe().to_markdown(numalign="left", stralign="left"))

    # Critical Preprocessing Step for Visualization: Handle Skewness
    # Applying log(1+x) transformation for better visualization of skewed data
    df['log_Yield'] = np.log1p(df['Yield'])
    
    # Generate Plots
    plt.figure(figsize=(18, 16))
    plt.suptitle('Exploratory Data Analysis: Crop Yield Dataset', fontsize=20, y=1.02)

    # Plot 1: Distribution of Log-Transformed Target Variable (Log(1+Yield))
    plt.subplot(2, 2, 1)
    sns.histplot(df['log_Yield'], bins=50, kde=True)
    plt.title('Distribution of Log(1 + Yield)', fontsize=14)
    plt.xlabel('Log(1 + Yield)')

    # Plot 2: Yield vs. Season (using log scale for visibility)
    plt.subplot(2, 2, 2)
    sns.boxplot(x='Season', y='log_Yield', data=df)
    plt.title('Log(1 + Yield) vs. Season', fontsize=14)
    plt.ylabel('Log(1 + Yield)')

    # Plot 3: Yield vs. Top 10 States (using log scale for visibility)
    top_10_states = df['State'].value_counts().nlargest(10).index
    df_top_states = df[df['State'].isin(top_10_states)]
    plt.subplot(2, 2, 3)
    sns.boxplot(x='State', y='log_Yield', data=df_top_states, order=top_10_states)
    plt.title('Log(1 + Yield) vs. Top 10 States', fontsize=14)
    plt.ylabel('Log(1 + Yield)')
    plt.xticks(rotation=45, ha='right')

    # Plot 4: Correlation Heatmap for Numerical Features
    plt.subplot(2, 2, 4)
    corr_matrix = df[['Crop_Year'] + numerical_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix of Numerical Features', fontsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(os.path.join(OUTPUT_DIR, 'crop_yield_eda.png'))
    print(f"\n[INFO] Crop Yield EDA image saved to {OUTPUT_DIR}/crop_yield_eda.png")
    print("="*50)


# --- SECTION 2: MINIMUM SUPPORT PRICE (MSP) DATASET EDA ---

def analyze_msp_data():
    """Performs EDA on the Minimum Support Price dataset."""
    print("\n\n" + "="*50)
    print("STARTING EDA for MSP Dataset (Price Prediction)")
    print("="*50)
    
    try:
        df_msp = pd.read_csv(MSP_PATH)
    except FileNotFoundError:
        print(f"Error: MSP file not found at {MSP_PATH}")
        return

    # Rename target column for simplicity
    df_msp = df_msp.rename(columns={'min_support_price': 'MSP'})

    # Descriptive Statistics
    print("\n--- 1. Descriptive Statistics (MSP) ---")
    print(df_msp['MSP'].describe().to_markdown(numalign="left", stralign="left"))
    
    # Generate Plots
    plt.figure(figsize=(18, 5))
    plt.suptitle('Exploratory Data Analysis: Minimum Support Price (MSP) Dataset', fontsize=20, y=1.02)

    # Plot 1: Distribution of MSP (Target Variable)
    plt.subplot(1, 3, 1)
    sns.histplot(df_msp['MSP'], bins=30, kde=True)
    plt.title('Distribution of MSP (Rs/quintal)', fontsize=14)

    # Plot 2: Average MSP by Season
    msp_by_season = df_msp.groupby('season')['MSP'].mean().sort_values(ascending=False).reset_index()
    plt.subplot(1, 3, 2)
    sns.barplot(x='season', y='MSP', data=msp_by_season)
    plt.title('Average MSP by Season', fontsize=14)
    plt.ylabel('Average MSP (Rs/quintal)')

    # Plot 3: Average MSP by Year
    # Clean up year column (e.g., extract first year)
    df_msp['start_year'] = df_msp['year'].str.split('-').str[0].astype(int)
    msp_by_year = df_msp.groupby('start_year')['MSP'].mean().reset_index()
    plt.subplot(1, 3, 3)
    sns.lineplot(x='start_year', y='MSP', data=msp_by_year, marker='o')
    plt.title('Average MSP Trend Over Time', fontsize=14)
    plt.xlabel('Start Year of Agricultural Cycle')
    plt.ylabel('Average MSP (Rs/quintal)')
    plt.xticks(rotation=45)

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(os.path.join(OUTPUT_DIR, 'msp_eda.png'))
    print(f"\n[INFO] MSP EDA image saved to {OUTPUT_DIR}/msp_eda.png")
    print("="*50)


if __name__ == '__main__':
    analyze_crop_yield_data()
    analyze_msp_data()