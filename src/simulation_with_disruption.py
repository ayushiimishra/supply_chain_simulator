import pandas as pd
import numpy as np
import networkx as nx
import os

class SupplyChainGraph:
    # (Paste your class from before, or import it for modular code.)

    def __init__(self, graph):
        self.graph = graph
        self.disrupted_nodes = set()

    def disrupt_node(self, node):
        if node in self.graph.nodes:
            self.disrupted_nodes.add(node)

    def get_affected_nodes(self):
        affected = set()
        for disrupted in self.disrupted_nodes:
            affected.update(nx.descendants(self.graph, disrupted))
        return affected

def build_supply_chain_graph(csv_path):
    df = pd.read_csv(csv_path)
    G = nx.DiGraph()
    for _, row in df.iterrows():
        supplier = row['Supplier name']
        plant = row['Location']
        customer = row['Customer demographics']
        G.add_node(supplier, node_type='supplier')
        G.add_node(plant, node_type='plant')
        G.add_node(customer, node_type='customer')
        G.add_edge(supplier, plant)
        G.add_edge(plant, customer)
    return G

def simulate_one_week(graph, sc_graph, df, week=1):
    """
    Simulates 1 week inventory flow across the network.
    - Blocks shipments from disrupted nodes.
    - Prints inventory and lost sales at each customer.
    """
    # Initialize plant and customer stock
    customer_stock = {}
    plant_stock = {}
    lost_sales = {}

    plants = set(df['Location'].unique())
    customers = set(df['Customer demographics'].unique())

    # Initial stocks (for demo: 100 at plant, 0 at customer)
    for p in plants:
        plant_stock[p] = 100
    for c in customers:
        customer_stock[c] = 0
        lost_sales[c] = 0

    # Simulate shipments: Supplier → Plant
    for _, row in df.iterrows():
        supplier, plant = row['Supplier name'], row['Location']
        prod_volume = row['Production volumes'] if not np.isnan(row['Production volumes']) else 20
        if supplier not in sc_graph.disrupted_nodes:
            plant_stock[plant] += prod_volume  # Arriving stock
        # If disrupted, no arrival!

    # Simulate shipments: Plant → Customer
    demand_mu = 20  # average demand, you can tweak or use your CSV
    for _, row in df.iterrows():
        plant, customer = row['Location'], row['Customer demographics']
        demand = int(np.random.normal(demand_mu, 5))
        if (plant not in sc_graph.disrupted_nodes and customer not in sc_graph.get_affected_nodes()):
            supplied = min(demand, plant_stock[plant])
            customer_stock[customer] += supplied
            plant_stock[plant] -= supplied
            lost_sales[customer] += max(0, demand - supplied)
        else:
            lost_sales[customer] += demand  # all demand lost

    print(f"\n=== WEEK {week} REPORT ===")
    print("-- Customer Stock Levels:")
    for c in customers:
        print(f"{c:20}: {customer_stock[c]}")
    print("-- Lost Sales (unsupplied):")
    for c in customers:
        print(f"{c:20}: {lost_sales[c]}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_path = os.path.join(project_root, "data", "supply_chain_data.csv")

    graph = build_supply_chain_graph(csv_path)
    sc_graph = SupplyChainGraph(graph)
    # Disrupt a supplier node (or pick your own)
    supplier_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'supplier']
    if supplier_nodes:
        sc_graph.disrupt_node(supplier_nodes[0])
        print(f"Disrupted Node: {supplier_nodes[0]}")

    df = pd.read_csv(csv_path)
    simulate_one_week(graph, sc_graph, df, week=1)
