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

# --- DEMO/DEFAULT FILE PATH ---
DEMO_FILE_PATH = os.path.join("data", "demo_supply_chain.csv")  # Place your demo CSV here

st.set_page_config(page_title="Supply Chain Simulation - Advanced", layout="wide")
st.title("📊 Supply Chain Resilience Dashboard")
st.markdown(
    '''
Upload your own **supply_chain_data.csv** or use the **demo** dataset below.  
The dashboard works instantly out of the box for demos and learning.
''',
    unsafe_allow_html=True
)

# ---- Robust Demo + Upload Logic (Unambiguous/Non-duplicated) ----
uploaded_file = st.file_uploader("Upload your supply_chain_data.csv", type=["csv"])
df = None
src_file = ""
use_demo = False

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(":open_file_folder: Using uploaded file.")
    src_file = "Custom uploaded CSV"
elif os.path.exists(DEMO_FILE_PATH):
    use_demo = st.checkbox("Or use the DEMO file below [default: checked]", value=True)
    if use_demo:
        df = pd.read_csv(DEMO_FILE_PATH)
        st.info(f":information_source: Demo file loaded from `{DEMO_FILE_PATH}`.")
        src_file = "Demo file"
    else:
        st.warning("No file provided. Please upload a CSV above or check the demo option.")
        st.stop()
else:
    st.error("No demo file found and nothing uploaded. Please upload a file to proceed.")
    st.stop()

st.write(f"**Using data source:** {src_file}")
st.dataframe(df.head(12), use_container_width=True)

# --- SHOCK TYPES & CONTROL PARAMS ---
SHOCK_TYPES = ["geopolitical", "cyberattack", "labor_strike", "natural_disaster"]
SEVERITY_MAP = {"mild": 0.25, "moderate": 0.5, "severe": 1.0}

# --- Sidebar: Simulation Controls and Custom Shock Controls ---
weeks = st.sidebar.slider("Number of Weeks", 4, 52, 12, 2)
disruption_prob = st.sidebar.slider("Disruption Probability", 0.0, 1.0, 0.3, 0.05)
rerouting_enabled = st.sidebar.checkbox("Enable rerouting", True)
st.sidebar.markdown("---")
st.sidebar.markdown("**🔔 Custom Shock Controls**")
shock_type = st.sidebar.selectbox("Shock Type", SHOCK_TYPES)
severity_level = st.sidebar.selectbox("Severity", list(SEVERITY_MAP))
custom_duration = st.sidebar.slider("Shock Duration (weeks)", 1, 8, 2)
custom_week = st.sidebar.number_input("Shock starts at week (1-indexed)", 1, weeks, 2)
apply_custom_shock = st.sidebar.checkbox("Simulate with custom shock on a plant", False)
user_seed = st.sidebar.number_input("Random Seed", min_value=0, value=42, step=1)

st.markdown("---")
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 KPI Dashboard", "🔗 Network Graph", "🧮 Scenario Comparison", "📊 Cost vs Resilience"
])

def make_shock_dict(week, shock_type, target, severity, duration):
    return {
        "week": int(week),
        "type": str(shock_type),
        "target": str(target),
        "severity": float(severity),
        "duration": int(duration)
    }

# --- Tab 1: KPI Dashboard ---
with tab1:
    st.subheader("Supply Chain KPIs")
    if st.button("Run Simulation", key="runsimtab1"):
        sim_df = df.head(20).copy()
        G = build_supply_chain_graph(sim_df)
        # Custom shocks always as dicts
        custom_shocks = []
        if apply_custom_shock and len(sim_df["Location"].unique()) > 0:
            plant_target = st.sidebar.selectbox(
                "Plant for custom shock", list(sim_df["Location"].unique()), key="target_plant"
            )
            custom_shocks.append(make_shock_dict(
                custom_week, shock_type, plant_target, SEVERITY_MAP[severity_level], custom_duration
            ))

        # Standard random shocks (always as dicts)
        random_shocks = []
        for idx, row in sim_df.iterrows():
            if np.random.rand() < disruption_prob:
                start = int(np.random.randint(3, weeks-2))
                length = int(np.random.randint(2, min(5, weeks-start)))
                random_shocks.append(make_shock_dict(
                    week=start,
                    shock_type="random",
                    target=row["Location"],
                    severity=1.0,
                    duration=length
                ))

        # Unified shock logic
        active_shocks = custom_shocks if apply_custom_shock else random_shocks

        results = multi_sku_simulation(
            sim_df, G, weeks=weeks,
            shock_params=active_shocks,
            rerouting_enabled=rerouting_enabled, seed=int(user_seed)
        )
        kpi_df = compute_weekly_kpis(results, weeks=weeks)

        # KPI Charts
        fig1 = px.line(kpi_df, x="Week", y=["Total Fulfilled Demand", "Total Lost Demand"], title="Fulfilled vs Lost Demand", markers=True)
        st.plotly_chart(fig1, use_container_width=True)

        alt_chart = (alt.Chart(kpi_df).transform_fold(
            ["Total Fulfilled Demand", "Total Lost Demand"], as_=["Type", "Value"]
        ).mark_bar().encode(
            x="Week:O", y="Value:Q",
            color="Type:N",
            tooltip=["Type:N", "Value:Q"]
        ).properties(title="Weekly Demand: Stacked (Altair)", width=600).interactive())
        st.altair_chart(alt_chart, use_container_width=True)

        fig2 = px.area(kpi_df, x="Week", y=["Plant Stock Avg", "Customer Stock Avg"], title="Inventory Levels Over Time")
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.bar(kpi_df, x="Week", y="% Rerouted", title="Percentage Rerouted Shipments per Week")
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("### KPI Data Table")
        st.dataframe(kpi_df, use_container_width=True)

        st.markdown(
            f"""
            <div style="padding:10px;background:#EDF3FC;border-radius:10px;">
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
    st.subheader("Network Structure (Interactive)")
    G = build_supply_chain_graph(df)
    color_map = {'supplier': '#7DC5EA', 'plant': '#FFB347', 'customer': '#FF8BA7'}
    node_types = nx.get_node_attributes(G, 'node_type')
    net = Network(notebook=False, height='500px', width='100%', bgcolor='#222222', font_color='white')

    for node in G.nodes:
        net.add_node(node, label=node, color=color_map.get(node_types.get(node, ''), '#bebebe'))
    for edge in G.edges:
        net.add_edge(edge[0], edge[1])
    net.save_graph('network.html')
    with open('network.html', 'r', encoding='utf-8') as HtmlFile:
        source_code = HtmlFile.read()
        components.html(source_code, height=550, width=900)

# --- Tab 3: Scenario Comparison ---
with tab3:
    st.subheader("Automated Scenario Grid")
    disruption_values = st.multiselect(
        "Select disruption rates for comparison:",
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9], default=[0.1, 0.5, 0.9]
    )
    scenario_seed = st.number_input("Random Seed for All Scenarios", min_value=0, value=42, step=1)
    if st.button("Run Scenario Grid", key="runscenarios"):
        summary_list = []
        for prob in disruption_values:
            sim_df = df.head(15)
            random_shocks = []
            for idx, row in sim_df.iterrows():
                if np.random.rand() < prob:
                    start = int(np.random.randint(2, weeks-3))
                    length = int(np.random.randint(2, min(5, weeks-start)))
                    random_shocks.append(make_shock_dict(
                        week=start, shock_type="random", target=row["Location"], severity=1.0, duration=length
                    ))
            G = build_supply_chain_graph(sim_df)
            results = multi_sku_simulation(
                sim_df, G, weeks=weeks, shock_params=random_shocks,
                rerouting_enabled=rerouting_enabled, seed=int(scenario_seed)
            )
            kpi_df = compute_weekly_kpis(results, weeks=weeks)
            kpi_df["Disruption Prob"] = prob
            summary_list.append(kpi_df)
        all_kpis = pd.concat(summary_list, ignore_index=True)

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

# --- Tab 4: Cost vs Resilience Tradeoff Plot ---
with tab4:
    st.subheader("Mitigation Cost vs Resilience (What-if Tradeoff)")
    buffer_range = st.slider("Select buffer/backup range (max units/dollars)", 0, 200, (0, 150), 10)
    mitigation_levels = list(range(buffer_range[0], buffer_range[1]+1, 25))
    scenario_results = []
    sim_df = df.head(15)  # or as needed for speed

    # Real scenario sweep!
    G = build_supply_chain_graph(sim_df)
    for buffer in mitigation_levels:
        # Run simulation for this buffer size
        results = multi_sku_simulation(
            sim_df, G, weeks=weeks, shock_params=[], 
            rerouting_enabled=rerouting_enabled, seed=int(user_seed), buffer_size=buffer
        )
        kpi_df = compute_weekly_kpis(results, weeks=weeks)
        avg_lost = kpi_df['Total Lost Demand'].mean()
        scenario_results.append({"Mitigation$": buffer, "AvgLostDemand": avg_lost / weeks})

    results_df = pd.DataFrame(scenario_results)
    fig = px.line(
        results_df, x="Mitigation$", y="AvgLostDemand",
        title="Cost vs Resilience: Mitigation Trade-off",
        markers=True
    )
    fig.update_traces(line_color="#395B64")
    fig.update_layout(
        xaxis_title="Mitigation Investment ($ or Units)",
        yaxis_title="Average Lost Demand",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "This chart is based on actual simulation or optimizer output: as you increase mitigation investment (e.g., buffer/backup spend), average lost demand typically decreases, then levels off."
    )

st.caption("Supply Chain Dashboard · Powered by Streamlit, Plotly, Altair, NetworkX, Pyvis, and your real simulation code.")
