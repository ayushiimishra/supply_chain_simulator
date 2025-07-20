import pandas as pd
import numpy as np
import networkx as nx
import random

class SupplyChainGraph:
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

    def find_alternate_path(self, source, target):
        disrupted = self.disrupted_nodes
        try:
            if source in disrupted or target in disrupted:
                return None
            return nx.shortest_path(self.graph, source=source, target=target)
        except nx.NetworkXNoPath:
            try:
                subgraph = self.graph.subgraph([n for n in self.graph.nodes if n not in disrupted])
                return nx.shortest_path(subgraph, source=source, target=target)
            except nx.NetworkXNoPath:
                return None

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

def simulate_multiweek_with_rerouting(graph, df, weeks=12, disruption_chance=0.2):
    sc_graph = SupplyChainGraph(graph)
    plants = set(df['Location'].unique())
    customers = set(df['Customer demographics'].unique())
    plant_stock = {plant: 100 for plant in plants}
    customer_stock = {cust: 0 for cust in customers}
    lost_sales = {cust: 0 for cust in customers}
    demand_mu = 20

    for week in range(1, weeks + 1):
        print(f"\n--- Week {week} Simulation ---")
        sc_graph.disrupted_nodes.clear()
        possible_disruptions = [n for n in graph.nodes() if graph.nodes[n]['node_type'] in ['supplier', 'plant']]
        for node in possible_disruptions:
            if random.random() < disruption_chance:
                sc_graph.disrupt_node(node)
        print(f"Disrupted Nodes this week: {sc_graph.disrupted_nodes}")

        # Supplier -> Plant shipments
        for _, row in df.iterrows():
            supplier = row['Supplier name']
            plant = row['Location']
            prod_volume = row['Production volumes'] if not np.isnan(row['Production volumes']) else 20
            if supplier in sc_graph.disrupted_nodes:
                alt_path = sc_graph.find_alternate_path(supplier, plant)
                if alt_path and len(alt_path) > 1:
                    print(f"Alternate supply path found for plant {plant} from {supplier}: {alt_path}")
                    plant_stock[plant] += prod_volume
                else:
                    print(f"No supply to plant {plant} (supplier {supplier} disrupted and no alternate path)")
            else:
                plant_stock[plant] += prod_volume

        # Plant -> Customer shipments
        for _, row in df.iterrows():
            plant = row['Location']
            customer = row['Customer demographics']
            demand = max(0, int(np.random.normal(demand_mu, 5)))
            supply_plant = plant
            rerouted = False

            if plant in sc_graph.disrupted_nodes or customer in sc_graph.get_affected_nodes():
                # Try alternate plant
                for alt_plant in plants:
                    if alt_plant != plant and alt_plant not in sc_graph.disrupted_nodes:
                        if sc_graph.find_alternate_path(alt_plant, customer):
                            supply_plant = alt_plant
                            rerouted = True
                            print(f"Rerouted customer {customer} demand from plant {plant} to plant {alt_plant}")
                            break
                if supply_plant == plant:
                    lost_sales[customer] += demand
                    print(f"Customer {customer} lost demand {demand} (plant disrupted or path blocked)")
                    continue

            supply = min(plant_stock.get(supply_plant, 0), demand)
            plant_stock[supply_plant] -= supply
            customer_stock[customer] += supply
            if (demand - supply) > 0:
                lost_sales[customer] += demand - supply
                print(f"Customer {customer} partial supply: {supply}, lost sales {demand - supply}")
            else:
                print(f"Customer {customer} fully supplied: {supply}")

        print("Inventory levels at end of week:")
        for p in plant_stock:
            print(f"Plant {p}: {plant_stock[p]} units")
        for c in customer_stock:
            print(f"Customer {c}: {customer_stock[c]} units, lost sales {lost_sales[c]}")

if __name__ == "__main__":
    csv_path = "data/supply_chain_data.csv"  # Update path if needed
    graph = build_supply_chain_graph(csv_path)
    df = pd.read_csv(csv_path)
    simulate_multiweek_with_rerouting(graph, df, weeks=12, disruption_chance=0.3)
