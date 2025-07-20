import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import networkx as nx
import seaborn as sns
from pulp import LpProblem, LpMinimize, LpVariable, lpSum

# --- Shock Types and Severities ---
SHOCK_TYPES = ["geopolitical", "cyberattack", "labor_strike", "natural_disaster"]
SEVERITY_MAP = {"mild": 0.25, "moderate": 0.5, "severe": 1.0}

def make_shock_dict(week, shock_type, target, severity, duration):
    return {
        "week": int(week),
        "type": str(shock_type),
        "target": str(target),
        "severity": float(severity),
        "duration": int(duration)
    }

def generate_shocks(graph, weeks, user_seed=None, custom_shocks=None):
    np.random.seed(user_seed)
    shocks = []
    # Only add well-formed dicts!
    if custom_shocks:
        for entry in custom_shocks:
            if isinstance(entry, dict):
                shocks.append(entry)
    # Random shocks
    for node in graph.nodes:
        if np.random.rand() < 0.3:
            week = np.random.randint(1, weeks - 4)
            shock_type = np.random.choice(SHOCK_TYPES)
            severity = SEVERITY_MAP[np.random.choice(list(SEVERITY_MAP))]
            duration = np.random.randint(1, 5)
            shocks.append(make_shock_dict(week, shock_type, node, severity, duration))
    return shocks

def get_active_shocks(shocks, week):
    status = {}
    for shock in shocks:
        # Always access as dict
        start, end = shock["week"], shock["week"] + shock["duration"]
        if start <= week < end:
            status[shock["target"]] = {
                "severity": shock["severity"],
                "type": shock["type"]
            }
    return status

def find_alternate_path(graph, source, dest, disrupted_nodes):
    try:
        permitted = [n for n in graph.nodes if n not in disrupted_nodes]
        SG = graph.subgraph(permitted)
        return nx.shortest_path(SG, source=source, target=dest)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None

def multi_sku_simulation(
    sim_df, graph, weeks=12, shock_params=None, rerouting_enabled=True, seed=42, buffer_size=0
):
    """
    Simulate inventory/disruptions with alternate rerouting, arbitrary buffer, and robust custom/random shock handling.
    Returns: {(SKU, Location): history dict}
    """
    np.random.seed(seed)
    plants = list(sim_df['Location'].unique())
    result_dict = {}

    # ---- Unified: shocks must always be list of dicts ----
    shocks = []
    if shock_params:
        # Accept any list: (tuples get rejected, only dicts allowed)
        shocks = [s for s in shock_params if isinstance(s, dict)]
    # Optionally: you can uncomment the next line to always add some random shocks if none are supplied
    # else:
    #     shocks = generate_shocks(graph, weeks, user_seed=seed, custom_shocks=[])

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
        # Apply mitigation/buffer at start
        stock += buffer_size if buffer_size else 0

        lead_time = int(row['Lead times']) if not np.isnan(row['Lead times']) else 2
        pipeline = [0] * lead_time
        base_weekly_demand = max(1, int(row['Number of products sold'] / weeks)) if not np.isnan(row['Number of products sold']) else 20
        shipment_qty = max(1, int(row['Production volumes'] / weeks)) if not np.isnan(row['Production volumes']) else 50

        for week in range(1, weeks + 1):
            active_shocks = get_active_shocks(shocks, week)
            in_shock = loc in active_shocks
            severity = active_shocks.get(loc, {}).get('severity', 0)

            # Effective output accounting for shock severity
            effective_shipment = int(shipment_qty * (1 - severity)) if in_shock else shipment_qty
            shipment = effective_shipment
            arrived = pipeline.pop(0)
            stock += arrived
            pipeline.append(shipment)

            demand = max(0, int(np.random.normal(base_weekly_demand, 5)))
            actual_sales = min(stock, demand)
            rerouted = False

            disrupted_nodes = set(active_shocks.keys())
            if in_shock and rerouting_enabled:
                for alt_loc in plants:
                    if alt_loc != loc and alt_loc not in disrupted_nodes:
                        path = find_alternate_path(graph, alt_loc, row['Customer demographics'], disrupted_nodes)
                        if path:
                            rerouted = True
                            # Optionally, record alternate source in output for advanced traceability
                            break
                if not rerouted:
                    actual_sales = 0

            lost = max(0, demand - actual_sales)
            stock -= actual_sales

            history["week"].append(week)
            history["stock"].append(stock)
            history["demand"].append(demand)
            history["sales"].append(actual_sales)
            history["lost_sales"].append(lost)
            history["arrived"].append(arrived)
            history["in_shock"].append(in_shock)
            history["rerouted"].append(rerouted)
        result_dict[(sku, loc)] = history
        total_lost = 0
        for hist in result_dict.values():
            total_lost += sum(hist['lost_sales'])
        avg_lost = total_lost / (weeks if weeks else 1)
        return result_dict, avg_lost

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

def optimize_backup_stock(plants, costs, lost_sales, budget):
    prob = LpProblem("BufferMitigation", LpMinimize)
    buffers = LpVariable.dicts("Buffer", plants, lowBound=0, cat='Continuous')
    total_cost = lpSum([costs[p] * buffers[p] for p in plants])
    total_loss = lpSum([max(0, lost_sales[p] - buffers[p]) for p in plants])
    prob += total_loss + 0.1 * total_cost
    prob += total_cost <= budget
    prob.solve()
    return {p: buffers[p].varValue for p in plants}

def plot_risk_heatmap(node_risk, output_path="risk_heatmap.png"):
    data = pd.DataFrame.from_dict(node_risk, orient="index", columns=["Total Lost"])
    plt.figure(figsize=(8, max(3, len(data) // 2)))
    sns.heatmap(data, annot=True, cmap="Reds", cbar=False)
    plt.title("Node Risk Heatmap: Total Lost Demand / Outages")
    plt.ylabel("Node")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Risk heatmap saved to {output_path}")

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

    # Select subset for demo or expand for full run
    simulation_df = df.head(10).copy()
    graph = build_supply_chain_graph(simulation_df)
    print(f"Simulating with first {len(simulation_df)} rows of data.")

    # Run a sample batch of scenarios and generate risk heatmap
    disruption_probs = [0.1, 0.3, 0.5, 0.7, 0.9]
    batch_results = []
    node_risk = {}
    for prob in disruption_probs:
        shock_params = {}
        for idx, row in simulation_df.iterrows():
            if random.random() < prob:
                start = random.randint(3, 7)
                length = random.randint(2, 4)
                shock_params[(row['SKU'], row['Location'])] = (start, length)
        results = multi_sku_simulation(
            simulation_df, graph, weeks=12, shock_params=shock_params, rerouting_enabled=True)
        kpi_df = compute_weekly_kpis(results, weeks=12)
        batch_results.append({"prob": prob, "kpi": kpi_df})
        plot_advanced_dashboard(kpi_df, output_dir=os.path.join(plots_dir, f"dash_prob_{str(prob).replace('.','')}"))
        # Aggregate risk (sum lost sales per location)
        for (sku, loc), hist in results.items():
            node_risk[loc] = node_risk.get(loc, 0) + sum(hist['lost_sales'])

    # Batch scenario comparison plot
    avg_lost = [k["kpi"]['Total Lost Demand'].mean() for k in batch_results]
    avg_rerouted = [k["kpi"]['% Rerouted'].mean() * 100 for k in batch_results]
    plt.figure(figsize=(10, 6))
    plt.plot(disruption_probs, avg_lost, marker='o', label='Avg Lost Demand')
    plt.plot(disruption_probs, avg_rerouted, marker='s', label='Avg % Rerouted')
    plt.xlabel('Disruption Probability'); plt.ylabel('Units/%'); plt.legend()
    plt.title('Batch Scenario Comparison: Loss & Rerouting vs Disruption Rate')
    plt.grid(True); plt.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, "batch_scenario_comparison.png"))
    plt.close()
    print(f"Batch scenario comparison chart saved to {plots_dir}")

    # Risk heatmap for vulnerable nodes
    plot_risk_heatmap(node_risk, output_path=os.path.join(plots_dir, "risk_heatmap.png"))

    # Mitigation optimization (dummy demo for now)
    plants = list(simulation_df['Location'].unique())
    dummy_costs = {p: 1.0 for p in plants}
    dummy_lost = {p: node_risk.get(p, 0) for p in plants}
    buffer_res = optimize_backup_stock(plants, dummy_costs, dummy_lost, budget=100.0)
    print(f"Optimal buffer allocation (sample only): {buffer_res}")

    # Save summary KPIs
    summary_df = compute_summary_table(results)
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
    summary_path = os.path.join(reports_dir, "summary_stats.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary stats saved to: {summary_path}")
    print("All steps complete.")
