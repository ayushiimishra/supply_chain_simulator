import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import networkx as nx

def find_alternate_path(graph, source, dest, disrupted_nodes):
    try:
        permitted = [n for n in graph.nodes if n not in disrupted_nodes]
        SG = graph.subgraph(permitted)
        return nx.shortest_path(SG, source=source, target=dest)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None

def multi_sku_simulation(sim_df, graph, weeks=12, shock_params=None, rerouting_enabled=True):
    """
    Simulate inventory/disruptions with alternate rerouting for multiple SKUs/locations.
    Returns: {(SKU, Location): history dict}
    """
    plants = list(sim_df['Location'].unique())
    customers = list(sim_df['Customer demographics'].unique())
    result_dict = {}

    disrupted_nodes = set()
    # Simulate a disruption scenario per (SKU, Location), but for batch compare, pass none or randomize
    for idx, pair in sim_df[['SKU', 'Location']].drop_duplicates().iterrows():
        sku, loc = pair['SKU'], pair['Location']
        filtered = sim_df[(sim_df['SKU'] == sku) & (sim_df['Location'] == loc)]
        if filtered.empty:
            continue
        row = filtered.iloc[0]
        history = {
            "week": [], "stock": [], "demand": [], "sales": [], "lost_sales": [],
            "arrived": [], "in_shock": [], "rerouted": []
        }
        stock = int(row['Stock levels']) if not np.isnan(row['Stock levels']) else 100
        lead_time = int(row['Lead times']) if not np.isnan(row['Lead times']) else 2
        pipeline = [0] * lead_time
        base_weekly_demand = max(1, int(row['Number of products sold'] / weeks)) if not np.isnan(row['Number of products sold']) else 20
        shipment_qty = max(1, int(row['Production volumes'] / weeks)) if not np.isnan(row['Production volumes']) else 50
        key = (sku, loc)
        (shock_week, shock_len) = shock_params[key] if (shock_params and key in shock_params) else (-1, 0)
        for week in range(1, weeks + 1):
            in_shock = (shock_week <= week < shock_week + shock_len) if shock_week > 0 else False
            if in_shock:
                disrupted_nodes.add(loc)
            else:
                disrupted_nodes.discard(loc)

            shipment = 0 if in_shock else shipment_qty
            arrived = pipeline.pop(0)
            stock += arrived
            pipeline.append(shipment)
            demand = max(0, int(np.random.normal(base_weekly_demand, 5)))
            actual_sales = min(stock, demand)
            rerouted = False

            if in_shock and rerouting_enabled:
                # Try to reroute from other plant not in disruption
                for alt_loc in plants:
                    if alt_loc != loc and alt_loc not in disrupted_nodes:
                        path = find_alternate_path(graph, alt_loc, row['Customer demographics'], disrupted_nodes)
                        if path:
                            rerouted = True
                            # Assume alternate plant can only serve part of demand for demo; adjust as needed
                            actual_sales = min(stock, demand)
                            break
            lost = max(0, demand - actual_sales)
            stock -= actual_sales
            history['week'].append(week)
            history['stock'].append(stock)
            history['demand'].append(demand)
            history['sales'].append(actual_sales)
            history['lost_sales'].append(lost)
            history['arrived'].append(arrived)
            history['in_shock'].append(in_shock)
            history['rerouted'].append(rerouted)
        result_dict[key] = history
    return result_dict

def compute_weekly_kpis(results, weeks=12):
    kpi_records = []
    for week_idx in range(weeks):
        fulfilled = 0
        lost = 0
        rerouted = 0
        plant_stock_list = []
        customer_stock_list = []
        for (sku, loc), hist in results.items():
            fulfilled += hist['sales'][week_idx]
            lost += hist['lost_sales'][week_idx]
            rerouted += int(hist['rerouted'][week_idx])
            plant_stock_list.append(hist['stock'][week_idx])
            customer_stock_list.append(hist['sales'][week_idx])
        kpi_records.append({
            'Week': week_idx + 1,
            'Total Fulfilled Demand': fulfilled,
            'Total Lost Demand': lost,
            'Rerouted Shipments': rerouted,
            'Plant Stock Avg': np.mean(plant_stock_list) if plant_stock_list else 0,
            'Customer Stock Avg': np.mean(customer_stock_list) if customer_stock_list else 0,
            '% Rerouted': rerouted / max(1, (fulfilled + rerouted))
        })
    return pd.DataFrame(kpi_records)

def plot_advanced_dashboard(kpi_df, output_dir="plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.figure(figsize=(18, 14))
    plt.subplot(3, 2, 1)
    plt.plot(kpi_df['Week'], kpi_df['Total Fulfilled Demand'], label='Fulfilled', marker='o')
    plt.plot(kpi_df['Week'], kpi_df['Total Lost Demand'], label='Lost', marker='x')
    plt.title('Fulfilled vs Lost Demand')
    plt.xlabel('Week'); plt.ylabel('Units'); plt.legend(); plt.grid(True)
    plt.subplot(3, 2, 2)
    plt.plot(kpi_df['Week'], kpi_df['Plant Stock Avg'], label='Plant Stock', marker='s', color='green')
    plt.title('Average Plant Stock'); plt.xlabel('Week'); plt.ylabel('Stock')
    plt.legend(); plt.grid(True)
    plt.subplot(3, 2, 3)
    plt.plot(kpi_df['Week'], kpi_df['Customer Stock Avg'], label='Customer Stock', marker='d', color='purple')
    plt.title('Average Customer Stock'); plt.xlabel('Week'); plt.ylabel('Stock')
    plt.legend(); plt.grid(True)
    plt.subplot(3, 2, 4)
    plt.plot(kpi_df['Week'], np.array(kpi_df['% Rerouted']) * 100, label='% Rerouted', marker='s', color='orange')
    plt.title('Percentage Rerouted Shipments'); plt.xlabel('Week'); plt.ylabel('% Rerouted')
    plt.ylim(0, 100); plt.legend(); plt.grid(True)
    plt.subplot(3, 2, 5)
    plt.plot(kpi_df['Week'], np.cumsum(kpi_df['Total Lost Demand']), label='Cumulative Lost', color='red')
    plt.title('Cumulative Lost Demand'); plt.xlabel('Week'); plt.ylabel('Cumulative Loss')
    plt.legend(); plt.grid(True)
    plt.subplot(3, 2, 6)
    plt.bar(kpi_df['Week'], kpi_df['Total Lost Demand'], label='Lost', color='coral')
    plt.bar(kpi_df['Week'], kpi_df['Total Fulfilled Demand'],
            bottom=kpi_df['Total Lost Demand'], label='Fulfilled', color='steelblue')
    plt.title('Stacked Weekly Demand'); plt.xlabel('Week'); plt.ylabel('Units')
    plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.suptitle('Supply Chain Resilience Simulation – KPI Dashboard', size=18, y=1.02)
    plt.subplots_adjust(top=0.92)
    dashboard_path = os.path.join(output_dir, "advanced_kpi_dashboard.png")
    plt.savefig(dashboard_path); plt.close()
    print(f"Advanced dashboard saved at {dashboard_path}")

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

def build_supply_chain_graph(sim_df):
    # Simple DiGraph using SKU->Location->Customer
    G = nx.DiGraph()
    for idx, row in sim_df.iterrows():
        supplier, plant, customer = row['Supplier name'], row['Location'], row['Customer demographics']
        G.add_node(supplier, node_type='supplier')
        G.add_node(plant, node_type='plant')
        G.add_node(customer, node_type='customer')
        G.add_edge(supplier, plant)
        G.add_edge(plant, customer)
    return G

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, "data", "supply_chain_data.csv")
    plots_dir = os.path.join(script_dir, "plots")
    reports_dir = os.path.join(script_dir, "reports")

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{data_path}'")
        exit()

    # DEMO: Select few rows for speed or test entire df for production
    simulation_df = df.head(10).copy()
    graph = build_supply_chain_graph(simulation_df)
    print(f"Simulating with first {len(simulation_df)} rows of data.")

    # Batch/Scenario Comparison:
    disruption_probs = [0.1, 0.3, 0.5, 0.7, 0.9]
    batch_results = []
    for prob in disruption_probs:
        # Assign disruptions for each row based on prob
        shock_params = {}
        for idx, row in simulation_df.iterrows():
            if random.random() < prob:
                start = random.randint(3, 7)
                length = random.randint(2, 4)
                shock_params[(row['SKU'], row['Location'])] = (start, length)
        # Simulate with rerouting enabled
        results = multi_sku_simulation(
            simulation_df, graph, weeks=12, shock_params=shock_params, rerouting_enabled=True
        )
        kpi_df = compute_weekly_kpis(results, weeks=12)
        batch_results.append({"prob": prob, "kpi": kpi_df})
        # Plot dashboard for each scenario
        plot_advanced_dashboard(kpi_df, output_dir=os.path.join(plots_dir, f"dash_prob_{str(prob).replace('.','')}"))
        print(f"Scenario {prob} complete.")

    # Comparison: plot average lost demand and % rerouted
    avg_lost = [k["kpi"]['Total Lost Demand'].mean() for k in batch_results]
    avg_rerouted = [k["kpi"]['% Rerouted'].mean()*100 for k in batch_results]

    plt.figure(figsize=(10, 6))
    plt.plot(disruption_probs, avg_lost, marker='o', label='Avg Lost Demand')
    plt.plot(disruption_probs, avg_rerouted, marker='s', label='Avg % Rerouted')
    plt.xlabel('Disruption Probability'); plt.ylabel('Units/%'); plt.legend()
    plt.title('Batch Scenario Comparison: Loss & Rerouting vs Disruption Rate')
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "batch_scenario_comparison.png"))
    plt.close()
    print(f"Batch scenario comparison chart saved to {plots_dir}")

    # For main/last scenario, save summary table
    summary_df = compute_summary_table(results)
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
    summary_path = os.path.join(reports_dir, "summary_stats.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary stats saved to: {summary_path}")

    print("All steps complete.")
