import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

def build_supply_chain_graph(csv_path):
    """
    Reads the supply chain data and builds a directed graph.
    """
    df = pd.read_csv(csv_path)
    
    G = nx.DiGraph()
    
    for idx, row in df.iterrows():
        # These variable names match your CSV file
        supplier = row['Supplier name']
        plant = row['Location'] 
        warehouse = row.get('Warehouse', None) # Safely get 'Warehouse', will be None if not found
        customer = row['Customer demographics']
        sku = row['SKU']
        
        # Add nodes
        G.add_node(supplier, node_type='supplier')
        G.add_node(plant, node_type='plant')
        if warehouse:
            G.add_node(warehouse, node_type='warehouse')
        G.add_node(customer, node_type='customer')
        
        # Add edges representing the flow
        G.add_edge(supplier, plant, sku=sku)
        if warehouse:
            G.add_edge(plant, warehouse, sku=sku)
            G.add_edge(warehouse, customer, sku=sku)
        else:
            # If no warehouse, flow is directly from plant to customer
            G.add_edge(plant, customer, sku=sku)
    
    return G

def visualize_supply_chain_graph(G, output_path="supply_chain_network.png"):
    """
    Visualizes the supply chain network and saves it to a file.
    """
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G, k=0.9, iterations=50) # Increased spacing
    
    node_types = nx.get_node_attributes(G, 'node_type')
    colors = {'supplier': 'skyblue', 'plant': 'orange', 'warehouse': 'green', 'customer': 'red'}
    node_color = [colors.get(node_types.get(n), 'gray') for n in G.nodes()]
    
    nx.draw(G, pos, with_labels=True, node_color=node_color, edge_color='gray', 
            node_size=2000, font_size=10, font_weight='bold', arrowsize=20)
            
    plt.title("Supply Chain Network Visualization", size=20)
    
    # Save the figure instead of showing it
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close() # Close the plot to free memory
    print(f"Graph saved to {output_path}")

if __name__ == "__main__":
    # --- Build a reliable path to the data file ---
    # This assumes your script is in a 'src' folder and 'data' is a sibling folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_path = os.path.join(project_root, "data", "supply_chain_data.csv")

    try:
        # --- 1. Build the graph ---
        G = build_supply_chain_graph(csv_path)
        print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        
        # --- 2. Visualize the graph ---
        # The output image will be saved in the same folder as your script
        output_image_path = os.path.join(script_dir, "supply_chain_network.png")
        visualize_supply_chain_graph(G, output_path=output_image_path)
        
    except FileNotFoundError:
        print(f"Error: Could not find the data file at the path: {csv_path}")
    except Exception as e:
        print(f"An error occurred: {e}")