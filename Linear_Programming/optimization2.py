import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt  # <--- ADDED THIS
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

# ==========================================
# 🚜 FARMER'S SCENARIO
# ==========================================
FARMER_STATE = 'Karnataka'
FARMER_CROPS = ['Rice', 'Maize', 'Sugarcane', 'Cotton(lint)', 'Arhar/Tur', 'Jute']

# --- 1. RESOURCES ---
TOTAL_LAND_HA = 10.0        
TOTAL_BUDGET_RS = 800000.0  
TOTAL_LABOR_DAYS = 1500.0   
TOTAL_WATER_M3 = 80000.0    

# --- 2. CONSTRAINTS ---
MAX_LIMITS_HA = {
    'Sugarcane': 3.0,      
    'Cotton(lint)': 4.0
}

MIN_LIMITS_HA = {
    'Rice': 1.0,           
    'Arhar/Tur': 0.5       
}

# --- 3. CROP INPUT ESTIMATES ---
CROP_RESOURCES = {
    'Rice':         {'Cost': 550, 'Labor': 150, 'Water': 1200}, 
    'Maize':        {'Cost': 350, 'Labor': 80,  'Water': 500},
    'Sugarcane':    {'Cost': 800, 'Labor': 200, 'Water': 2000}, 
    'Cotton(lint)': {'Cost': 450, 'Labor': 120, 'Water': 700},
    'Arhar/Tur':    {'Cost': 300, 'Labor': 60,  'Water': 400},
    'Jute':         {'Cost': 400, 'Labor': 100, 'Water': 600},
}
DEFAULT_RES = {'Cost': 400, 'Labor': 100, 'Water': 600}
DEFAULT_PRICE = 1500.0

# --- Helper Functions ---
def load_data():
    try:
        df_orig = pd.read_csv(ORIGINAL_YIELD_FILE)
        df_pred = pd.read_csv(PRED_YIELD_FILE)
        df_msp_orig = pd.read_csv(ORIGINAL_MSP_FILE)
        df_msp_pred = pd.read_csv(PRED_MSP_FILE)
    except FileNotFoundError:
        # Dummy data generation for testing if files missing
        print("Warning: Data files not found. Using mock data.")
        return pd.DataFrame(), pd.DataFrame()
    
    df_comb = df_orig.copy()
    df_comb['Predicted_Yield'] = df_pred['Predicted_Yield_ton_ha']
    df_msp_comb = df_msp_orig.copy()
    df_msp_comb['Predicted_MSP'] = df_msp_pred['Predicted_MSP']
    
    for df in [df_comb, df_msp_comb]:
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip()
    return df_comb, df_msp_comb

def prepare_data(df_yield, df_msp):
    stats = []
    print(f"Analyzing crops for {FARMER_STATE}...")
    
    # Fallback if data is empty (Mocking for the sake of the graph if needed)
    if df_yield.empty:
        print("Using hardcoded yield/price values for demonstration.")
        mock_yields = {'Rice': 4.0, 'Maize': 5.0, 'Sugarcane': 80.0, 'Cotton(lint)': 1.5, 'Arhar/Tur': 1.2, 'Jute': 2.5}
        mock_prices = {'Rice': 2000, 'Maize': 1800, 'Sugarcane': 300, 'Cotton(lint)': 6000, 'Arhar/Tur': 6000, 'Jute': 4000}
    
    for crop in FARMER_CROPS:
        if not df_yield.empty:
            row = df_yield[(df_yield['State'] == FARMER_STATE) & (df_yield['Crop'].str.contains(crop, case=False, na=False))]
            if row.empty: continue
            yield_val = row['Predicted_Yield'].mean()
            
            msp_row = df_msp[df_msp['crop'].str.contains(crop, case=False, na=False)]
            price = msp_row['Predicted_MSP'].mean() if not msp_row.empty else DEFAULT_PRICE
        else:
            yield_val = mock_yields.get(crop, 3.0)
            price = mock_prices.get(crop, 3000)

        res = DEFAULT_RES
        for k in CROP_RESOURCES:
            if k.lower() in crop.lower(): res = CROP_RESOURCES[k]
        
        # Adjust yield unit for sugarcane if necessary (usually tons are high)
        revenue = (yield_val * 1.0) * price # Revenue per Ha
        profit = revenue - res['Cost']
        
        stats.append({
            'Crop': crop, 'Profit': profit, 
            'Cost': res['Cost'], 'Labor': res['Labor'], 'Water': res['Water']
        })
    return pd.DataFrame(stats)

# --- VISUALIZATION FUNCTION ---
def visualize_results(results, resource_usage):
    """
    Generates a dashboard showing the Optimal Vertex (Allocation) 
    and the Constraint Saturation.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Optimization Results: {FARMER_STATE}', fontsize=16)

    # --- PLOT 1: THE OPTIMAL VERTEX (Crop Allocation) ---
    crops = [r['Crop'] for r in results if r['Area'] > 0]
    areas = [r['Area'] for r in results if r['Area'] > 0]
    colors = plt.cm.Paired(np.arange(len(crops)))

    ax1.bar(crops, areas, color=colors, edgecolor='black')
    ax1.set_title('Optimal Land Allocation (Vertex)', fontsize=12)
    ax1.set_ylabel('Hectares Allocated')
    ax1.set_ylim(0, max(areas) * 1.2)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add labels on bars
    for i, v in enumerate(areas):
        ax1.text(i, v + 0.1, f"{v:.2f} ha", ha='center', fontweight='bold')

    # --- PLOT 2: THE CONSTRAINTS (Resource Usage) ---
    # Normalize everything to percentage to show "how close to the limit" we are
    res_names = list(resource_usage.keys())
    used = [v['Used'] for v in resource_usage.values()]
    limits = [v['Limit'] for v in resource_usage.values()]
    pct_used = [u/l*100 for u, l in zip(used, limits)]
    
    # Color bar red if it hits the limit (Constraint Binding)
    bar_colors = ['green' if p < 99 else 'red' for p in pct_used]

    y_pos = np.arange(len(res_names))
    ax2.barh(y_pos, pct_used, color=bar_colors, edgecolor='black')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(res_names)
    ax2.set_xlabel('Percentage of Resource Used (%)')
    ax2.set_title('Constraint Analysis (Resource Saturation)', fontsize=12)
    ax2.set_xlim(0, 110)
    ax2.axvline(100, color='red', linestyle='--', linewidth=2, label='Constraint Limit')
    
    # Text labels inside bars
    for i, (u, l) in enumerate(zip(used, limits)):
        label_text = f"{int(u):,} / {int(l):,}"
        ax2.text(5, i, label_text, va='center', color='white', fontweight='bold', fontsize=10)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- SOLVER ---
def solve_optimization(df):
    print(f"\n--- Running Balanced Optimization ---")
    
    relevant_df = df.copy()
    model = LpProblem("Balanced_Farm_Optimization", LpMaximize)
    crops = relevant_df['Crop'].tolist()
    x = LpVariable.dicts("Area", crops, lowBound=0, cat='Continuous')
    
    # Objective
    model += lpSum([x[c] * relevant_df.loc[relevant_df['Crop'] == c, 'Profit'].values[0] for c in crops])
    
    # Resource Constraints
    model += lpSum([x[c] for c in crops]) <= TOTAL_LAND_HA, "Land"
    model += lpSum([x[c] * relevant_df.loc[relevant_df['Crop'] == c, 'Cost'].values[0] for c in crops]) <= TOTAL_BUDGET_RS, "Budget"
    model += lpSum([x[c] * relevant_df.loc[relevant_df['Crop'] == c, 'Labor'].values[0] for c in crops]) <= TOTAL_LABOR_DAYS, "Labor"
    model += lpSum([x[c] * relevant_df.loc[relevant_df['Crop'] == c, 'Water'].values[0] for c in crops]) <= TOTAL_WATER_M3, "Water"

    # Diversification Constraints
    for c in crops:
        if c in MAX_LIMITS_HA:
            model += x[c] <= MAX_LIMITS_HA[c], f"Max_Limit_{c}"
        if c in MIN_LIMITS_HA:
            model += x[c] >= MIN_LIMITS_HA[c], f"Min_Limit_{c}"

    model.solve()
    
    # --- POST-OPTIMIZATION CALCS ---
    final_allocation = []
    total_profit = 0
    
    # Track totals for the constraint plot
    total_land_used = 0
    total_budget_used = 0
    total_labor_used = 0
    total_water_used = 0

    print("\n" + "="*60)
    print(f"OPTIMAL ALLOCATION ({LpStatus[model.status]})")
    print("="*60)
    
    for c in crops:
        area = x[c].value()
        if area is None: area = 0 # Safety
        
        row = relevant_df[relevant_df['Crop'] == c].iloc[0]
        
        if area > 0.01:
            p = area * row['Profit']
            total_profit += p
            
            # Accumulate used resources
            total_land_used += area
            total_budget_used += (area * row['Cost'])
            total_labor_used += (area * row['Labor'])
            total_water_used += (area * row['Water'])

            note = ""
            if c in MIN_LIMITS_HA and abs(area - MIN_LIMITS_HA[c]) < 0.1: note = "(Min Met)"
            elif c in MAX_LIMITS_HA and abs(area - MAX_LIMITS_HA[c]) < 0.1: note = "(Max Hit)"

            print(f"✅ GROW **{c:<15}**: {area:.2f} ha   {note}")
            final_allocation.append({'Crop': c, 'Area': area})

    print("-" * 60)
    print(f"NET PROFIT: ₹ {total_profit:,.2f}")
    print("="*60)
    
    # --- TRIGGER PLOT ---
    resource_data = {
        'Land (ha)': {'Used': total_land_used, 'Limit': TOTAL_LAND_HA},
        'Water (m3)': {'Used': total_water_used, 'Limit': TOTAL_WATER_M3},
        'Labor (days)': {'Used': total_labor_used, 'Limit': TOTAL_LABOR_DAYS},
        'Budget (₹)': {'Used': total_budget_used, 'Limit': TOTAL_BUDGET_RS},
    }
    
    visualize_results(final_allocation, resource_data)

if __name__ == '__main__':
    df_y, df_m = load_data()
    data = prepare_data(df_y, df_m)
    if not data.empty: solve_optimization(data)