import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.express as px
import altair as alt
import os
from pyvis.network import Network
import streamlit.components.v1 as components
from simulation import multi_sku_simulation, compute_weekly_kpis, build_supply_chain_graph

st.set_page_config(page_title="Supply Chain Simulation - Advanced", layout="wide")
st.title("📊 Supply Chain Resilience Dashboard")

# --- File Upload with Fallback Demo ---
uploaded_file = st.file_uploader("Upload your `supply_chain_data.csv`", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("CSV uploaded successfully.")
else:
    demo_path = os.path.join("..", "data", "supply_chain_data.csv")
    if os.path.exists(demo_path):
        df = pd.read_csv(demo_path)
        st.info("Demo data loaded from 'data/supply_chain_data.csv'.")
    else:
        st.error("No CSV provided and demo file not found. Please upload a file above to proceed.")
        st.stop()

weeks = st.sidebar.slider("Number of Weeks", 4, 52, 12, 2)
disruption_prob = st.sidebar.slider("Disruption Probability", 0.0, 1.0, 0.3, 0.05)
rerouting_enabled = st.sidebar.checkbox("Enable rerouting", True)

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📈 KPI Dashboard", "🔗 Network Graph", "🧮 Scenario Comparison"])

# --- Tab 1: KPI Dashboard ---
with tab1:
    st.subheader("Supply Chain KPIs")
    if st.button("Run Simulation"):
        sim_df = df.head(20).copy()
        plants = sim_df["Location"].unique()
        shock_params = {}
        for idx, row in sim_df.iterrows():
            if np.random.rand() < disruption_prob:
                start = np.random.randint(3, weeks-2)
                length = np.random.randint(2, min(5, weeks-start))
                shock_params[(row["SKU"], row["Location"])] = (start, length)
        G = build_supply_chain_graph(sim_df)
        results = multi_sku_simulation(sim_df, G, weeks=weeks, shock_params=shock_params, rerouting_enabled=rerouting_enabled)
        kpi_df = compute_weekly_kpis(results, weeks=weeks)

        # Plotly line: Fulfilled vs Lost Demand
        fig1 = px.line(kpi_df, x="Week", y=["Total Fulfilled Demand", "Total Lost Demand"],
                       title="Fulfilled vs Lost Demand", markers=True)
        st.plotly_chart(fig1, use_container_width=True)

        # Altair stacked bar
        alt_chart = (alt.Chart(kpi_df).transform_fold(
            ["Total Fulfilled Demand", "Total Lost Demand"], as_=["Type", "Value"]
        ).mark_bar().encode(
            x="Week:O",
            y="Value:Q",
            color="Type:N",
            tooltip=["Type:N", "Value:Q"]
        ).properties(title="Weekly Demand: Stacked (Altair)", width=600).interactive())
        st.altair_chart(alt_chart, use_container_width=True)

        # Plant and customer stock
        fig2 = px.area(kpi_df, x="Week", y=["Plant Stock Avg", "Customer Stock Avg"],
                       title="Inventory Levels Over Time")
        st.plotly_chart(fig2, use_container_width=True)

        # Percentage rerouted
        fig3 = px.bar(kpi_df, x="Week", y="% Rerouted", title="Percentage Rerouted Shipments per Week")
        st.plotly_chart(fig3, use_container_width=True)

        # Data table
        st.markdown("### KPI Data Table")
        st.dataframe(kpi_df, use_container_width=True)

        st.markdown(
            f"""
            <div style="padding:10px;background#2C3E50;border-radius:10px;">
            <b>Summary:</b>
            <ul>
                <li>Avg. Lost Demand: <span style="color:#ED4753;">{kpi_df['Total Lost Demand'].mean():.1f}</span></li>
                <li>Avg. Rerouted: <span style="color:#E27602;">{kpi_df['% Rerouted'].mean()*100:.1f}%</span></li>
                <li>Final Week Plant Stock Avg: <span style="color:#37A929;">{kpi_df['Plant Stock Avg'].iloc[-1]:.1f}</span></li>
                <li>Final Week Customer Stock Avg: <span style="color:#2155CD;">{kpi_df['Customer Stock Avg'].iloc[-1]:.1f}</span></li>
            </ul>
            </div>
            """, unsafe_allow_html=True
        )

    else:
        st.info("Click 'Run Simulation' to view interactive KPIs.")

# --- Tab 2: Network Graph ---
with tab2:
    st.subheader("Network Structure")
    G = build_supply_chain_graph(df)
    net = Network(notebook=False, height='500px', width='100%', bgcolor='#222222', font_color='white')
    color_map = {
        'supplier': '#7DC5EA', 
        'plant': '#FFB347', 
        'customer': '#FF8BA7'
    }
    node_types = nx.get_node_attributes(G, 'node_type')
    for node in G.nodes:
        net.add_node(node, label=node, color=color_map.get(node_types.get(node, ''), '#bebebe'))
    for edge in G.edges:
        net.add_edge(edge[0], edge[1])
    net.save_graph('network.html')
    HtmlFile = open('network.html', 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    components.html(source_code, height=550, width=900)
    # Optionally: add streamlit-agraph or networkx visual in production

# --- Tab 3: Scenario Comparison ---
with tab3:
    st.subheader("Automated Scenario Grid")
    disruption_values = st.multiselect(
        "Select disruption rates for comparison:",
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9], default=[0.1, 0.5, 0.9]
    )
    if st.button("Run Scenario Grid"):
        summary_list = []
        for prob in disruption_values:
            sim_df = df.head(15)
            shock_params = {}
            for idx, row in sim_df.iterrows():
                if np.random.rand() < prob:
                    start = np.random.randint(2, weeks-3)
                    length = np.random.randint(2, min(5, weeks-start))
                    shock_params[(row["SKU"], row["Location"])] = (start, length)
            G = build_supply_chain_graph(sim_df)
            results = multi_sku_simulation(sim_df, G, weeks=weeks, shock_params=shock_params, rerouting_enabled=rerouting_enabled)
            kpi_df = compute_weekly_kpis(results, weeks=weeks)
            kpi_df["Disruption Prob"] = prob
            summary_list.append(kpi_df)
        all_kpis = pd.concat(summary_list, ignore_index=True)

        # Side-by-side lost/fulfilled
        fig = px.line(
            all_kpis, x="Week", y="Total Lost Demand", color="Disruption Prob",
            title="Total Lost Demand: Scenario Grid"
        )
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.line(
            all_kpis, x="Week", y="Total Fulfilled Demand", color="Disruption Prob",
            title="Total Fulfilled Demand: Scenario Grid"
        )
        st.plotly_chart(fig2, use_container_width=True)

st.caption("Supply Chain Dashboard · Powered by Streamlit, Plotly, Altair, networkx, pandas")
