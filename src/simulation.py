import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os

def multi_sku_warehouse_simulation(sim_df, weeks=12, shock_params=None):
    """
    Simulate inventory across multiple SKUs and locations with real data & shock events.
    Returns a dictionary with results for each SKU/location pair.
    """
    result_dict = {}
    # Use drop_duplicates to get unique SKU-Location pairs that exist in the data
    for idx, row_info in sim_df[['SKU', 'Location']].drop_duplicates().iterrows():
        sku = row_info['SKU']
        loc = row_info['Location']
        
        # Filter for the specific combination
        filtered = sim_df[(sim_df['SKU'] == sku) & (sim_df['Location'] == loc)]
        if filtered.empty:
            continue
        
        row = filtered.iloc[0]
        history = {
            "week": [], "stock": [], "demand": [],
            "sales": [], "lost_sales": [], "arrived": [], "in_shock": []
        }
        
        # Get parameters from data or fallbacks if missing/NaN
        stock = int(row['Stock levels']) if not np.isnan(row['Stock levels']) else 100
        lead_time = int(row['Lead times']) if not np.isnan(row['Lead times']) else 2
        pipeline = [0] * lead_time
        
        # Distribute demand/production over simulated weeks, with randomness
        base_weekly_demand = max(1, int(row['Number of products sold'] / weeks)) if not np.isnan(row['Number of products sold']) else 20
        shipment_qty = max(1, int(row['Production volumes'] / weeks)) if not np.isnan(row['Production volumes']) else 50

        key = (sku, loc)
        shock_week, shock_len = -1, 0
        if shock_params and key in shock_params:
            shock_week, shock_len = shock_params[key]
        lost_sales = 0

        for week in range(1, weeks + 1):
            in_shock = (shock_week <= week < shock_week + shock_len) if shock_week > 0 else False
            shipment = 0 if in_shock else shipment_qty
            arrived = pipeline.pop(0)
            stock += arrived
            pipeline.append(shipment)
            demand = max(0, int(np.random.normal(base_weekly_demand, 5)))
            actual_sales = min(stock, demand)
            lost = max(0, demand - stock)
            stock -= actual_sales
            lost_sales += lost
            
            # Log history
            history['week'].append(week)
            history['stock'].append(stock)
            history['demand'].append(demand)
            history['sales'].append(actual_sales)
            history['lost_sales'].append(lost)
            history['arrived'].append(arrived)
            history['in_shock'].append(in_shock)
        result_dict[key] = history
    return result_dict

def plot_simulation_results(result_dict, output_dir="plots"):
    """
    Saves simulation results as PNG files for each SKU-location pair.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for key, hist in result_dict.items():
        plt.figure(figsize=(12, 6)) # Create a new figure for each plot
        plt.plot(hist['week'], hist['stock'], label="Stock", marker='o', zorder=5)
        plt.plot(hist['week'], hist['demand'], label="Demand", linestyle='--')
        plt.plot(hist['week'], hist['sales'], label="Sales", linestyle=':')
        plt.plot(hist['week'], hist['lost_sales'], label="Lost Sales", linestyle='-.', color='orange')
        
        # Highlight the disruption period
        if any(hist['in_shock']):
             plt.fill_between(hist['week'], 0, max(hist['stock'])*1.1 if max(hist['stock']) > 0 else 100, 
                              where=hist['in_shock'], color='red', alpha=0.2, label="Disruption")

        plt.title(f"Simulation for SKU: {key[0]}, Location: {key[1]}")
        plt.xlabel("Week")
        plt.ylabel("Units")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        
        # Save the plot to a file and close it to free memory
        plot_filename = os.path.join(output_dir, f"sim_{key[0]}_{key[1]}.png")
        plt.savefig(plot_filename)
        plt.close()
        
def compute_summary_table(results):
    summary_list = []
    for (sku, loc), hist in results.items():
        total_lost_sales = sum(hist['lost_sales'])
        avg_stock = sum(hist['stock']) / len(hist['stock'])
        disruptions = sum(hist['in_shock'])
        summary_list.append({
            'SKU': sku,
            'Location': loc,
            'Total Lost Sales': total_lost_sales,
            'Mean Stock Level': round(avg_stock, 2),
            'Disruption Weeks': disruptions
        })
    return pd.DataFrame(summary_list)        

if __name__ == "__main__":
    # --- Build paths relative to THIS script file for robustness ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, "data", "supply_chain_data.csv")
    plots_dir = os.path.join(script_dir, "plots") # Save plots inside 'src'
    reports_dir = os.path.join(script_dir, "reports") # Define reports directory

    # --- 1. Load data and select a valid subset for the demo ---
    try:
        df = pd.read_csv(data_path)
        df.rename(columns={
            'product': 'SKU',
            'region': 'Location',
            'initial_stock': 'Stock levels',
            'demand_mean': 'Number of products sold',
            'production_capacity': 'Production volumes',
            'lead_time_mean': 'Lead times'
        }, inplace=True)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{data_path}'")
        exit()

    # **THE FIX**: Take the first 5 rows to guarantee we have valid SKU/Location pairs
    simulation_df = df.head(5).copy() 
    print(f"Simulating with the first {len(simulation_df)} rows of data.")

    # --- 2. Set up advanced disruption scenarios ---
    shock_params = {}
    for idx, row in simulation_df.iterrows():
        if random.random() < 0.5: # ~50% chance of a disruption
            start = random.randint(3, 7)
            length = random.randint(2, 4)
            shock_params[(row['SKU'], row['Location'])] = (start, length)

    # --- 3. Run simulation ---
    print("Running simulation...")
    results = multi_sku_warehouse_simulation(simulation_df, weeks=12, shock_params=shock_params)

    # --- 4. Visualize results ---
    if results:
        print("Saving plots...")
        plot_simulation_results(results, output_dir=plots_dir)
        print(f"\n Simulation complete. Plots saved in: {plots_dir}")
        print("Computing summary table...")
        summary_df = compute_summary_table(results)
        
        # Create reports directory if it doesn't exist
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
        summary_path = os.path.join(reports_dir, "summary_stats.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary stats saved to: {summary_path}")
        
        print("\n Simulation complete.")
    else:
        print("\n No results were generated from the simulation. No plots to save.")