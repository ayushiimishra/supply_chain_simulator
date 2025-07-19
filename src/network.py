import pandas as pd

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully!")
        print("Columns:", df.columns.tolist())
        print("First few rows:\n", df.head())
        return df
    except FileNotFoundError:
        print(f"Error: The file was not found at the path: {file_path}")
        print("Please check your file path and folder structure.")
        return None

def extract_nodes(df):
    suppliers = df['Supplier name'].unique()
    warehouses = df['Location'].unique()   
    customers = df['Customer demographics'].unique()
    print("\nSuppliers:", suppliers)
    print("\nWarehouses/Locations:", warehouses)
    print("\nCustomers:", customers)
    return suppliers, warehouses, customers

def extract_edges(df):
    flows = df[['Supplier name', 'Location', 'Customer demographics']]
    edges = []
    for idx, row in flows.iterrows():
        edges.append((row['Supplier name'], row['Location']))
        edges.append((row['Location'], row['Customer demographics']))
    edges = list(set(edges))
    print("\nSample Connections (edges):", edges[:10])
    return edges

if __name__ == "__main__":
    relative_path = "data/supply_chain_data.csv"
    df = load_data(relative_path)
    extract_nodes(df)
    extract_edges(df)

    if df is not None:
        print("\nDataFrame is ready for analysis.")