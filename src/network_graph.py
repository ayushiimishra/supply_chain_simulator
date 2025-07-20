import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

class SupplyChainGraph:
    def __init__(self, graph):
        self.graph = graph
        self.disrupted_nodes = set()

    def disrupt_node(self, node):
        """Mark a node as disrupted."""
        if node in self.graph.nodes:
            self.disrupted_nodes.add(node)
        else:
            print(f"Node '{node}' not found in graph.")

    def get_affected_nodes(self):
        """Get all downstream nodes affected by disruptions."""
        affected = set()
        for disrupted in self.disrupted_nodes:
            affected.update(nx.descendants(self.graph, disrupted))
        return affected

    def visualize(self, output_path="disruption_impact.png"):
        """Visualizes the network and saves the plot to a file."""
        pos = nx.spring_layout(self.graph, k=0.8, iterations=50)
        node_types = nx.get_node_attributes(self.graph, 'node_type')
        colors = {'supplier': 'skyblue', 'plant': 'orange', 'customer': 'red'}

        # Calculate affected nodes once for efficiency
        affected_nodes = self.get_affected_nodes()

        node_color = []
        for n in self.graph.nodes:
            if n in self.disrupted_nodes:
                node_color.append('black')
            elif n in affected_nodes:
                node_color.append('lightcoral')
            else:
                node_color.append(colors.get(node_types.get(n), 'gray'))

        plt.figure(figsize=(14, 10))
        nx.draw(self.graph, pos, with_labels=True, node_color=node_color,
                edge_color='gray', node_size=1500, font_size=9, font_weight='bold', arrowsize=20)
        plt.title("Supply Chain Network – Disruptions and Impact", size=16)

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Disruption visualization saved to {output_path}")

def build_supply_chain_graph(csv_path):
    """Builds a directed graph from the supply chain data."""
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

if __name__ == "__main__":
    # Set up a robust path to your CSV file
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        csv_path = os.path.join(project_root, "data", "supply_chain_data.csv")

        graph = build_supply_chain_graph(csv_path)
        sc_graph = SupplyChainGraph(graph)

        # Automatically disrupt the first supplier node for demonstration
        supplier_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'supplier']
        if supplier_nodes:
            sc_graph.disrupt_node(supplier_nodes[0])
            print(f"Disrupted Node: {supplier_nodes[0]}")
        else:
            print("No supplier node found to disrupt.")

        # Save and display the visualization
        output_file = os.path.join(script_dir, "disruption_impact.png")
        sc_graph.visualize(output_path=output_file)

    except FileNotFoundError:
        print(f"Error: Could not find data file at {csv_path}")
    except IndexError:
        print("Could not find a supplier node to disrupt in the sample data.")
    except Exception as e:
        print(f"An error occurred: {e}")
