# Supply Chain Resilience Simulation Dashboard

A Python/Streamlit tool for modeling, analyzing, and visualizing supply chain disruptions and mitigation strategies.

Explore the full dashboard here: https://ayushi-risksim.streamlit.app/

## Overview

This project provides an interactive dashboard to simulate supply chain disruptions (such as geopolitical events, labor strikes, cyberattacks, or natural disasters), analyze lost demand and buffer impacts, and visualize network vulnerabilities. The app is ideal for practitioners, students, and analysts to explore “what-if” scenarios and resilience strategies.

**Features:**
- Supply chain network modeling from CSV data
- Simulation of custom and random multi-week disruptions (type, severity, duration)
- Inventory, demand, sales, lost sales, and rerouting logic
- Scenario sweeps and cost vs. resilience trade-off analysis
- Interactive dashboards for KPI trends, risk heatmaps, live network map
- Cloud-ready; runs fully in the cloud after deployment

## Live Demo

Try the dashboard online:  
`https://your-deployed-app-link.streamlit.app`

## Setup & Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/supply-chain-resilience-simulator.git
   cd supply-chain-resilience-simulator
   ```

2. Install dependencies (recommended: use a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```

3. Run the dashboard:
   ```bash
   streamlit run src/streamlit_dashboard.py
   ```

4. Open in browser:  
   Go to http://localhost:8501

## How It Works

- **Load Data:** Use the provided demo CSV or upload your own supply chain data.
- **Set Simulation Parameters:** Choose disruption type, duration, severity, buffer inventory, rerouting, and other settings.
- **Run Simulation:** Advance the network week-by-week, apply disruptions, compute demand and fulfillment flows, and log KPI results.
- **Analyze Results:** Use dashboards for KPI trends, network structure, scenario grid, and the cost vs. resilience curve. Export results and explore optimal resilience investments.

## Project Structure

```
├── data/
│   └── demo_supply_chain.csv               # Example data file
├── src/
│   ├── simulation.py                       # Simulation engine
│   ├── streamlit_dashboard.py              # Interactive dashboard
├── requirements.txt                        # Package dependencies
├── README.md                               # This file
```

## Sample Data Schema

The dashboard accepts a CSV file with columns:
- Supplier name
- Location
- Customer demographics
- SKU
- Production volumes
- Number of products sold
- Lead times
- Stock levels

For an example, see `/data/demo_supply_chain.csv`.

## Deployment

Deploy on Streamlit Cloud:
- Push this repo to GitHub.
- Go to Streamlit Cloud, select the repository and app script (`src/streamlit_dashboard.py`).
- Deploy and share your link.

## Screenshots

<img width="1918" height="932" alt="image" src="https://github.com/user-attachments/assets/327996be-dff0-428a-bbf9-a7b3a6569b11" />
<img width="1523" height="572" alt="image" src="https://github.com/user-attachments/assets/8e832236-f50d-4a3d-b4f4-f59294844a0e" />
<img width="1518" height="627" alt="image" src="https://github.com/user-attachments/assets/b78725f2-48dd-425c-88db-47fb6b5cffdb" />
<img width="1487" height="601" alt="image" src="https://github.com/user-attachments/assets/6dbea83e-e22e-4bfe-87c1-c209bfbe5c51" />
<img width="1907" height="903" alt="image" src="https://github.com/user-attachments/assets/85eccf63-34bb-4f03-b5e5-3fc3a726f116" />
<img width="1432" height="803" alt="image" src="https://github.com/user-attachments/assets/adad25b3-0f97-4b67-8e20-adc564e94461" />
<img width="1457" height="813" alt="image" src="https://github.com/user-attachments/assets/460ae2b0-8893-47b2-9bb2-6809a4499be9" />
<img width="1493" height="797" alt="image" src="https://github.com/user-attachments/assets/1a8aa924-2ece-4964-ad88-6948607682d9" />
<img width="1482" height="871" alt="image" src="https://github.com/user-attachments/assets/7a10d3d8-c894-4f3b-95ff-9d933e5a93a9" />


## Key Features

- Flexible network and scenario modeling
- Built-in disruption types: geopolitical, cyber, strike, disaster
- User-configurable mitigation and reroute controls
- Batch scenario/grid runner for risk assessment
- Visualization of demand, stock, lost sales, node risk, rerouting percentage
- Cloud- and recruiter-ready: deploys instantly via Streamlit sharing

## Contributing

Pull requests, bug reports, and feature suggestions are welcome.  
Please open an issue or submit a pull request for review.

## License

MIT License. See the LICENSE file for details.

## Author & Contact

Ayushi Mishra
Email: ayushixmishra@email.com  

### Inspiration & References

- FTOT-Resilience-Supply_Chain (Volpe USDOT)
- MIT SCREAM Game
- Streamlit
