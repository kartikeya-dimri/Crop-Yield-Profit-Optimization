import pandas as pd
import numpy as np
import os
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpStatus

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
PRED_DIR = os.path.join(PROJECT_ROOT, 'Final_Predictions')

ORIGINAL_YIELD_FILE = os.path.join(DATA_DIR, 'crop_yield.csv')
PRED_YIELD_FILE = os.path.join(PRED_DIR, 'crop_yield_final_predictions.csv')
PRED_MSP_FILE = os.path.join(PRED_DIR, 'msp_final_predictions.csv')
ORIGINAL_MSP_FILE = os.path.join(DATA_DIR, 'minimum-support-prices.csv')

# --- Simulation Settings ---
FARMER_STATE = 'Uttarakhand'  # Changed to your scenario
FARMER_SEASON = 'Kharif'
TOTAL_LAND_AVAILABLE_HA = 0.5
BUDGET_AVAILABLE = 200000    

# --- Economic Assumptions ---
# Base cost of cultivation per hectare (Seeds, labor, basic fertilizers)
BASE_COST_PER_HA = 40000.0 
# Default market price if MSP is not found (Conservative estimate)
DEFAULT_PRICE_RS_QT = 1500.0 

# --- 1. Data Loading and Merging ---

def load_and_prepare_data():
    print("Loading data for optimization...")
    try:
        df_orig = pd.read_csv(ORIGINAL_YIELD_FILE)
        df_pred_yield = pd.read_csv(PRED_YIELD_FILE)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit()
    
    df_combined = df_orig.copy()
    df_combined['Predicted_Yield'] = df_pred_yield['Predicted_Yield_ton_ha']
    
    # Clean Data
    if 'State' in df_combined.columns:
        df_combined['State'] = df_combined['State'].str.strip()
    if 'Season' in df_combined.columns:
        df_combined['Season'] = df_combined['Season'].str.strip()
        
    try:
        df_msp_orig = pd.read_csv(ORIGINAL_MSP_FILE)
        df_msp_pred = pd.read_csv(PRED_MSP_FILE)
    except FileNotFoundError:
        print("Error: MSP data missing.")
        exit()

    df_msp_combined = df_msp_orig.copy()
    df_msp_combined['Predicted_MSP'] = df_msp_pred['Predicted_MSP']
    
    if 'season' in df_msp_combined.columns:
        df_msp_combined['season'] = df_msp_combined['season'].str.strip()
    if 'crop' in df_msp_combined.columns:
        df_msp_combined['crop'] = df_msp_combined['crop'].str.strip()
    
    return df_combined, df_msp_combined

# --- 2. Optimization Engine ---

def run_optimization(df_combined, df_msp, state, season, total_land, budget):
    print(f"\n--- Running Optimization for {state} ({season}) ---")
    
    candidate_crops_df = df_combined[
        (df_combined['State'] == state) & 
        (df_combined['Season'] == season)
    ].copy()
    
    if candidate_crops_df.empty:
        print(f"No crops found for State: '{state}', Season: '{season}'")
        return

    # Group by Crop
    crop_stats = candidate_crops_df.groupby('Crop').agg({
        'Predicted_Yield': 'mean'
    }).reset_index()
    
    # 2. Get Prices (MSP) & Calculate Financials
    crop_stats['Estimated_Price'] = 0.0
    crop_stats['Source_Price'] = 'Fallback'
    
    for index, row in crop_stats.iterrows():
        crop_name = row['Crop']
        msp_row = df_msp[df_msp['crop'].str.contains(crop_name, case=False, na=False)]
        
        if not msp_row.empty:
            crop_stats.at[index, 'Estimated_Price'] = msp_row['Predicted_MSP'].mean()
            crop_stats.at[index, 'Source_Price'] = 'MSP Model'
        else:
            crop_stats.at[index, 'Estimated_Price'] = DEFAULT_PRICE_RS_QT
            
    # Revenue Calculation
    # Revenue (Rs/ha) = Yield (ton/ha) * 10 (qt/ton) * Price (Rs/qt)
    crop_stats['Revenue_Per_Ha'] = (crop_stats['Predicted_Yield'] * 10) * crop_stats['Estimated_Price']
    
    # Cost Calculation (Refined)
    # We assume cost is at least BASE_COST, OR 50% of revenue (whichever is higher)
    # This prevents "free money" scenarios where high yield = pure profit
    crop_stats['Estimated_Cost'] = crop_stats['Revenue_Per_Ha'].apply(lambda r: max(BASE_COST_PER_HA, r * 0.5))
    
    # Profit Calculation
    crop_stats['Profit_Per_Ha'] = crop_stats['Revenue_Per_Ha'] - crop_stats['Estimated_Cost']
    
    # Filter Profitable Crops
    profitable_crops = crop_stats[crop_stats['Profit_Per_Ha'] > 0].reset_index(drop=True)
    
    if profitable_crops.empty:
        print("No profitable crops found (Costs > Revenue).")
        print("Top 3 revenue generators (for debugging):")
        print(crop_stats.sort_values(by='Revenue_Per_Ha', ascending=False).head(3)[['Crop', 'Predicted_Yield', 'Estimated_Price', 'Revenue_Per_Ha']])
        return

    print(f"\nIdentified {len(profitable_crops)} profitable crops. Top 5 by Profit/ha:")
    print(profitable_crops[['Crop', 'Predicted_Yield', 'Estimated_Price', 'Estimated_Cost', 'Profit_Per_Ha']].sort_values(by='Profit_Per_Ha', ascending=False).head(5).to_string(index=False))

    # --- 3. Linear Programming Model ---
    model = LpProblem(name="Crop_Allocation_Optimization", sense=LpMaximize)
    
    crops = profitable_crops['Crop'].tolist()
    area_vars = LpVariable.dicts("Area", crops, lowBound=0, cat='Continuous')
    
    # Objective: Maximize Profit
    model += lpSum([area_vars[c] * profitable_crops.loc[profitable_crops['Crop'] == c, 'Profit_Per_Ha'].values[0] for c in crops])
    
    # Constraints
    model += lpSum([area_vars[c] for c in crops]) <= total_land, "Total_Land"
    model += lpSum([area_vars[c] * profitable_crops.loc[profitable_crops['Crop'] == c, 'Estimated_Cost'].values[0] for c in crops]) <= budget, "Budget"
    
    model.solve()
    
    # --- 4. Output Results ---
    print("\n" + "="*50)
    print(f"OPTIMIZATION RESULTS ({LpStatus[model.status]})")
    print("="*50)
    
    total_profit = 0
    total_area = 0
    total_cost = 0
    
    for c in crops:
        area = area_vars[c].value()
        if area and area > 0.01:
            row = profitable_crops.loc[profitable_crops['Crop'] == c].iloc[0]
            profit = area * row['Profit_Per_Ha']
            cost = area * row['Estimated_Cost']
            revenue = area * row['Revenue_Per_Ha']
            
            total_profit += profit
            total_area += area
            total_cost += cost
            
            print(f"✅ GROW **{c:<12}**: {area:.2f} ha")
            print(f"   - Yield: {row['Predicted_Yield']:.2f} ton/ha")
            print(f"   - Price: ₹ {row['Estimated_Price']:.2f} /qt ({row['Source_Price']})")
            print(f"   - Econ:  Rev: ₹ {revenue:,.0f} | Cost: ₹ {cost:,.0f} | Profit: ₹ {profit:,.0f}")
            print("-" * 50)
            
    print(f"Total Land: {total_area:.2f}/{total_land} ha")
    print(f"Total Cost: ₹ {total_cost:,.2f}")
    print(f"NET PROFIT: ₹ {total_profit:,.2f}")
    print("="*50)

if __name__ == '__main__':
    try:
        import pulp
    except ImportError:
        print("Error: 'pulp' library missing.")
        exit()

    df_yield, df_msp = load_and_prepare_data()
    
    run_optimization(
        df_yield, df_msp, 
        state=FARMER_STATE, 
        season=FARMER_SEASON, 
        total_land=TOTAL_LAND_AVAILABLE_HA,
        budget=BUDGET_AVAILABLE
    )