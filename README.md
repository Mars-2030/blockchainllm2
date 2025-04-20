# 🦠 Pandemic Supply Chain Simulation (using OpenAI LLM Agents) 🦠

This project simulates a multi-echelon (Manufacturer -> Distributor -> Hospital) supply chain for essential drugs during a pandemic scenario. It utilizes Large Language Models (LLMs) via the OpenAI API, combined with rule-based heuristics and forecasting tools, to power the decision-making of agents at each level of the supply chain.

## Overview

The simulation models:
*   Multiple regions with varying populations and healthcare capacities.
*   Multiple drugs with different criticalities, demand factors, and production complexities.
*   Epidemic spread using a multi-wave SIR (Susceptible-Infected-Recovered) model.
*   Drug demand derived from active infection cases.
*   Potential supply chain disruptions (manufacturing, transportation).
*   Inventory management and order fulfillment logic across the supply chain.
*   Agent decision-making using OpenAI LLMs guided by specific prompts, historical data, forecasts, and rule-based overrides.
*   Performance metrics tracking (service level, stockouts, patient impact).
*   Visualization of scenario parameters and simulation results.
*   (Optional) Blockchain integration for transparent allocation tracking.

## Features

*   **Configurable Scenarios:** Adjust number of regions, drugs, simulation length, pandemic severity, disruption probability, etc. via command-line arguments.
*   **LLM-Powered Agents:** Manufacturer, Distributor, and Hospital agents use OpenAI models (configurable, e.g., GPT-3.5-Turbo, GPT-4o) for production, ordering, and allocation decisions.
*   **Hybrid Approach:** Combines LLM reasoning with rule-based fallbacks and adjustments for robustness and proactive behavior.
*   **Integrated Tools:** Agents utilize tools for epidemic forecasting, disruption prediction, optimal ordering, and criticality assessment.
*   **SIR-Based Demand:** Realistic demand generation based on simulated pandemic waves.
*   **Disruption Modeling:** Simulates random manufacturing and transportation disruptions.
*   **Detailed Metrics:** Tracks key performance indicators like service level, stockouts, unfulfilled demand, and patient impact.
*   **Comprehensive Visualization:** Generates plots for epidemic curves, demand, disruptions, inventory levels, performance metrics, etc. using Matplotlib and Seaborn.
*   **Rich Console Output:** Uses the `rich` library for enhanced, readable console logging during simulation.
*   **HTML Reporting:** Saves the detailed console log, including agent reasoning and decisions, to an HTML file.
*   **Blockchain Integration (Optional):** Can connect to an Ethereum node (like Ganache) to record key transactions (allocations, case data) on a smart contract for transparency analysis (requires separate contract deployment).

## Project Structure

```
pandemic_supply_chain/
├── README.md
├── requirements.txt
├── main.py                 # Main simulation script, argument parsing
├── config.py               # Configuration settings, colors, helper functions
├── .env.example            # Example environment file (!!! DO NOT COMMIT ACTUAL .env !!!)
├── .gitignore              # Git ignore file
├── src/
│   ├── __init__.py
│   ├── scenario/           # Pandemic scenario generation and visualization
│   │   ├── __init__.py
│   │   ├── generator.py    # Generates regions, drugs, SIR curves, disruptions
│   │   └── visualizer.py   # Creates plots for scenario data
│   ├── environment/        # Simulation environment logic
│   │   ├── __init__.py
│   │   ├── supply_chain.py # Core simulation environment class, step logic
│   │   └── metrics.py      # Calculates and visualizes performance metrics
│   ├── agents/             # Agent implementations
│   │   ├── __init__.py
│   │   ├── base.py         # Base class for LLM agents
│   │   ├── manufacturer.py # Manufacturer agent logic
│   │   ├── distributor.py  # Distributor agent logic
│   │   └── hospital.py     # Hospital agent logic
│   ├── tools/              # Decision-support tools for agents
│   │   ├── __init__.py
│   │   ├── forecasting.py  # Epidemic and disruption forecasting
│   │   ├── allocation.py   # Allocation and order quantity optimization
│   │   └── assessment.py   # Criticality assessment logic
│   ├── llm/                # Language model integration
│   │   ├── __init__.py
│   │   └── openai_integration.py # Handles OpenAI API calls
│   └── blockchain/         # (Optional) Blockchain interface
│       ├── __init__.py
│       └── interface.py    # Connects to Ethereum node and contract
└── output/                 # Default folder for simulation results (plots, HTML report)
    └── .gitkeep            # Placeholder to keep directory in Git
```

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd pandemic_supply_chain
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up OpenAI API Key:**
    *   Create a file named `.env` in the project root directory (`pandemic_supply_chain/`).
    *   Add your OpenAI API key to the `.env` file:
        ```dotenv
        OPENAI_API_KEY='sk-...' # Replace with your actual OpenAI API key
        ```
    *   **IMPORTANT:** Ensure the `.env` file is listed in your `.gitignore` file to avoid committing your secret key. Add `.env` to `.gitignore` if it's not already there.

5.  **(Optional) Blockchain Setup:**
    *   If you intend to use the `--use-blockchain` flag, you need:
        *   A running Ethereum node (e.g., Ganache for local testing, or an Infura/Alchemy endpoint).
        *   The simulation's smart contract deployed to that network.
        *   Update the `CONTRACT_ADDRESS` and potentially `CONTRACT_ABI_PATH` in `src/blockchain/interface.py` if they differ from the defaults.
        *   Ensure the node URL provided via `--node-url` (or the default in `interface.py`) is correct.

## Running the Simulation

Execute the main script from the project root directory:

```bash
python main.py [OPTIONS]
```

**Common Options:**

*   `--help`: Show all available command-line options.
*   `--days DAYS`: Set the number of simulation days (e.g., `--days 90`).
*   `--regions REGIONS`: Set the number of regions (e.g., `--regions 5`).
*   `--drugs DRUGS`: Set the number of drugs (e.g., `--drugs 3`).
*   `--model MODEL_NAME`: Specify the OpenAI model (e.g., `--model gpt-4o`). Default is `gpt-3.5-turbo`.
*   `--severity SEVERITY`: Set pandemic severity (0-1, can exceed 1 for extreme tests, e.g., `--severity 0.9`).
*   `--disrupt-prob PROB`: Set base disruption probability factor (e.g., `--disrupt-prob 0.3`).
*   `--warehouse-delay DAYS`: Set warehouse release delay (e.g., `--warehouse-delay 3`).
*   `--allocation-batch DAYS`: Set manufacturer allocation batch frequency (1 for daily, e.g., `--allocation-batch 7` for weekly).
*   `--quiet`: Reduce console output verbosity.
*   `--no-viz`: Disable generating visualization plots.
*   `--use-blockchain`: Enable blockchain integration (requires node and contract setup).
*   `--node-url URL`: Specify the Ethereum node URL if using blockchain.

**Example:**

```bash
# Run a 120-day simulation with 5 regions, 3 drugs, using GPT-4o
python main.py --days 120 --regions 5 --drugs 3 --model gpt-4o

# Run a shorter, quieter simulation with weekly allocations and higher disruption chance
python main.py --days 60 --regions 3 --drugs 2 --quiet --allocation-batch 7 --disrupt-prob 0.4
```

## Output

The simulation will:

1.  Print configuration and progress to the console (using `rich` formatting).
2.  Generate various plots (epidemic curves, demand, inventory levels, performance metrics, etc.) and save them as PNG files in a timestamped subfolder within the `output/` directory (e.g., `output/output_YYYYMMDD_HHMMSS.../`).
3.  Save a detailed HTML report of the console output (including agent decisions and reasoning) to the same output subfolder (`simulation_report_openai.html`).

## Configuration

*   Core simulation parameters can be adjusted via command-line arguments (see above).
*   Internal constants, color themes, and the OpenAI API key loading logic are in `config.py`.
*   **API Key MUST be set in the `.env` file.**