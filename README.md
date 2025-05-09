# Pandemic Supply Chain Simulation with OpenAI and Blockchain

## Overview

This project simulates a multi-echelon (Manufacturer -> Distributor -> Hospital) supply chain for essential drugs during a synthetic pandemic scenario. It utilizes Large Language Models (LLMs) via the OpenAI API, combined with rule-based heuristics and forecasting tools, to power the decision-making of agents at each level of the supply chain.

A key feature is the integration of a blockchain (simulated locally using Hardhat) to enhance trust and automate critical processes:

1.  **Trusted Data Source:** Regional case counts, derived from the simulation's internal SIR model, are written to the blockchain daily. Agents requiring this trusted data (currently the Manufacturer) query the blockchain via a dedicated tool, rather than relying solely on potentially less trustworthy summarized data passed directly between agents.
2.  **Automated Critical Actions:** The logic for fair allocation of drugs from the manufacturer to distributors, based on requests, drug criticality, and trusted on-chain case counts, is implemented as a smart contract function (`executeFairAllocation`). The simulation triggers this function when blockchain integration is enabled.

## Features

*   **Multi-Echelon Simulation:** Models Manufacturer, regional Distributors, and regional Hospitals.
*   **Pandemic Dynamics:** Uses an SIR (Susceptible-Infected-Recovered) model to generate epidemic curves per region, influencing drug demand.
*   **LLM-Powered Agents:** Agents use OpenAI (GPT-3.5-turbo, GPT-4, etc.) for core decision-making (production, allocation, ordering).
*   **Rule-Based Logic & Tools:** Agents employ fallback rules and helper tools (forecasting, optimal order quantity, criticality assessment) to augment LLM decisions.
*   **Blockchain Integration (Hardhat/Solidity):**
    *   **Trusted Case Data:** Manufacturer agent queries the blockchain for verified regional case counts via a tool (`get_blockchain_regional_cases_tool`).
    *   **On-Chain Fair Allocation:** Manufacturer allocation logic is executed via the `executeFairAllocation` smart contract function, leveraging on-chain data (cases, criticality).
    *   Daily case counts are written to the blockchain by the environment.
    *   Initial drug criticalities are set on-chain during deployment.
*   **Scenario Generation:** Configurable parameters for regions, drugs, pandemic severity, and disruptions.
*   **Metrics & Visualization:** Tracks key performance indicators (stockouts, service level, patient impact) and generates plots using Matplotlib/Seaborn.
*   **Configuration:** Uses a `.env` file for API keys and blockchain settings.
*   **Console Output:** Rich console output for detailed logging and reporting (savable as HTML).

## Architecture & Blockchain Interaction

1.  **Environment:** Simulates the supply chain, pandemic spread (SIR model), and processes actions.
2.  **Agents (Python):**
    *   Receive observations from the environment (excluding sensitive data like direct case counts).
    *   Use tools (including blockchain query tools).
    *   Generate reasoning and decisions using OpenAI API.
    *   Employ fallback logic and rules.
3.  **Blockchain (Hardhat Node + `SupplyChainData.sol`):**
    *   **Initial Setup:** The deployment script (`deploy.js`) calls `setDrugCriticality` for each drug configured in the script.
    *   **Daily Case Update:** The environment calculates simulated cases and calls `updateRegionalCaseCount` on the contract.
    *   **Manufacturer Case Query:** The Manufacturer agent calls `get_blockchain_regional_cases_tool`, which uses the `BlockchainInterface` to call `getRegionalCaseCount` on the contract.
    *   **Fair Allocation Execution:** The environment calls `_calculate_fair_allocation`. If blockchain is enabled, this triggers the `BlockchainInterface` to call `executeFairAllocation` on the contract, passing manufacturer requests and available inventory. The contract performs the calculation using on-chain criticalities and case counts. It **reverts** if criticality is not set for a requested drug.
4.  **Tools (Python):** Provide functions for forecasting, assessment, order quantities, and blockchain queries.
5.  **OpenAI Integration:** Handles API calls to the chosen LLM.

## Prerequisites

*   **Python:** 3.8 or higher recommended.
*   **Node.js & npm:** LTS version recommended (includes npm). [Download Node.js](https://nodejs.org/)
*   **Git:** For cloning the repository.
*   **OpenAI API Key:** Obtain from [OpenAI](https://platform.openai.com/signup/).

## Setup Instructions (Local Hardhat)

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-project-directory>
    ```

2.  **Set up Python Environment:**
    ```bash
    # Create virtual environment
    python -m venv venv

    # Activate (macOS/Linux)
    source venv/bin/activate
    # OR Activate (Windows CMD/PowerShell)
    # venv\Scripts\activate

    # Install dependencies
    pip install -r requirements.txt
    ```

3.  **Install Node.js Dependencies:**
    ```bash
    npm install
    ```

4.  **Configure Environment Variables (`.env`):**
    *   Create a file named `.env` in the project root.
    *   Add the following content, replacing placeholders:

        ```dotenv
        # Required: Your OpenAI API Key
        OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

        # Required: Private key for LOCAL Hardhat node interaction.
        # Get this from the output of `npx hardhat node` (see Step 6).
        # Example (Hardhat default account #0):
        PRIVATE_KEY="0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

        # Recommended: Local node URL (Hardhat default)
        NODE_URL="http://127.0.0.1:8545"

        # Auto-filled by deploy script - leave blank or commented initially
        # CONTRACT_ADDRESS=""
        ```
    *   **IMPORTANT:** Use a private key provided by the `npx hardhat node` command for `PRIVATE_KEY`. **Do not use real private keys.**

5.  **Compile Smart Contract:**
    ```bash
    npx hardhat compile
    ```
    This generates necessary artifacts (ABI, bytecode) in the `artifacts/` directory.

6.  **(Optional but Recommended) Adjust Drug Criticality Setup:**
    *   The smart contract requires drug criticalities to be set for its `executeFairAllocation` function.
    *   By default, `scripts/deploy.js` sets criticalities for Drugs 0, 1, and 2.
    *   **If you plan to run the Python simulation with `--drugs N` where N > 3:** You **must** modify `scripts/deploy.js` to set criticalities for drugs up to ID `N-1`.
    *   **To modify `deploy.js`:**
        *   Open `scripts/deploy.js`.
        *   Find the `// --- MODIFICATION START ---` and `// --- MODIFICATION END ---` comments within the `main` function.
        *   Change `const numDrugsToConfigure = 3;` to the number of drugs you intend to use (e.g., `5`).
        *   Adjust the `defaultCriticalities` array or the logic to assign appropriate criticality values (1=Low, 2=Medium, 3=High, 4=Critical) for all drugs you will simulate.
        *   Save the `deploy.js` file.
    *   **Failure to do this** will cause blockchain allocations for drugs without set criticalities to fail and revert to the local Python fallback logic.

## Running the Simulation (Local Hardhat)

1.  **Start Local Blockchain Node (Terminal 1):**
    *   Open a terminal, navigate to the project directory.
    *   Run:
        ```bash
        npx hardhat node
        ```
    *   This terminal will start printing blockchain activity. **Leave it running.**
    *   **Copy one of the printed `Private Key:` values** (e.g., for Account #0) and paste it into your `.env` file for the `PRIVATE_KEY` variable if you haven't already.

2.  **Deploy Smart Contract (Terminal 2):**
    *   Open a *new* terminal, navigate to the project directory.
    *   Run (ensure Node.js is accessible, no Python venv active here):
        ```bash
        npm run deploy:localhost
        ```
    *   This script compiles (if needed), deploys the `SupplyChainData` contract to the running Hardhat node, **sets initial drug criticalities (as configured in `deploy.js`)**, saves the ABI to `src/blockchain/`, and **updates the `CONTRACT_ADDRESS` in your `.env` file.**
    *   Verify the output shows success, the correct number of drug criticalities being set, and that your `.env` file now contains the `CONTRACT_ADDRESS`.

3.  **Run Python Simulation (Terminal 2 or 3):**
    *   In the same terminal as deployment (or a new one), **activate your Python virtual environment**:
        ```bash
        # Activate (macOS/Linux)
        source venv/bin/activate
        # OR Activate (Windows CMD/PowerShell)
        # venv\Scripts\activate
        ```
    *   Run the main script, enabling blockchain integration. **Ensure the `--drugs` argument matches the number configured in `deploy.js` (Step 6 above)!**
        ```bash
        # Example: 3 regions, 3 drugs (matching default deploy.js), 10 days, using llm, verbose
        python main.py --regions 3 --drugs 3 --days 10 --use-blockchain --use-llm

        # Example: 5 regions, 5 drugs (requires deploy.js modification!), 50 days, rule based
        # python main.py --regions 5 --drugs 5 --days 50 --use-blockchain
        ```
    *   The `--use-blockchain` flag is crucial to enable interaction with the Hardhat node and smart contract.
    *   The `--use-llm` flag is crucial to enable llm based decision making.
    

## Using MetaMask with Local Hardhat (Optional)

MetaMask is primarily for interacting with public testnets or mainnet, or for development tools like Remix. You **do not need it** just to run the simulation locally as described above. However, you can connect it to your local Hardhat node to view accounts and potentially interact manually:

1.  **Install MetaMask:** Get the browser extension from the official [MetaMask website](https://metamask.io/download/).
2.  **Add Local Network:**
    *   Open MetaMask, click the network dropdown (usually says "Ethereum Mainnet").
    *   Click "Add network".
    *   Click "Add a network manually".
    *   Fill in the details:
        *   **Network Name:** `Hardhat Local` (or any name you prefer)
        *   **New RPC URL:** `http://127.0.0.1:8545`
        *   **Chain ID:** `31337`
        *   **Currency Symbol:** `ETH`
    *   Click "Save".
3.  **Import Hardhat Account:**
    *   In MetaMask, click the circle icon (top right) -> "Import account".
    *   Go back to the terminal where `npx hardhat node` is running.
    *   Copy the `Private Key` for one of the accounts (e.g., Account #0).
    *   Paste the private key into MetaMask and click "Import".
    *   **Warning:** Only import these specific Hardhat testing keys. Never import keys holding real value this way.
4.  **View:** You can now select the "Hardhat Local" network in MetaMask and see the balance (e.g., 10000 ETH) of the imported account. Advanced users could use tools like Remix connected via MetaMask to manually call functions on the contract deployed locally.

## Checking Results

1.  **Python Console Output (`main.py` run):**
    *   Look for successful blockchain initialization and transaction confirmations (`✓`).
    *   Check the daily "Epidemic State" table: `Cases (BC)` should reflect the values written on the previous day.
    *   Observe agent decisions and `[TOOL]Blockchain Regional Cases Output` logs.
    *   Note any warnings about allocation falling back to local logic (indicates a missing drug criticality on-chain).
    *   Review the final results summary (stockouts, impact, service level).
    *   Check the final "Querying Final Blockchain State" section for consistency (cases and criticalities).
2.  **Hardhat Node Console Output (Terminal 1):**
    *   Observe `eth_` JSON-RPC calls corresponding to Python interactions.
    *   Look for transaction mining confirmations or error messages if transactions revert (especially `executeFairAllocation` if criticalities are missing).
3.  **Output Folder (`output_<timestamp>_..._blockchain/`):**
    *   **`simulation_report_openai_blockchain.html`:** Open in a browser for a complete, colorized log of the simulation run.
    *   **`.png` files:** Analyze the generated plots for supply chain performance, inventory levels, epidemic curves, etc.

## Project Structure
```
.
├── .env                   # Environment variables (API Keys, Private Key - Gitignored)
├── .gitignore             # Git ignore file
├── README.md              # Project documentation (This file)
├── config.py              # Simulation configuration, console setup, colors
├── contracts/
│   └── SupplyChainData.sol # Solidity smart contract
├── hardhat.config.js      # Hardhat configuration file
├── ignition/              # (Optional) Hardhat Ignition deployment files
├── main.py                # Main simulation script entry point
├── node_modules/          # Node.js dependencies (created by npm install - Gitignored)
├── output*/               # Default folder for simulation results (HTML, plots - Gitignored)
├── package-lock.json      # Exact Node.js dependency versions
├── package.json           # Node.js project metadata and dependencies
├── requirements.txt       # Python dependencies
├── scripts/
│   └── deploy.js          # Hardhat script to deploy the smart contract
├── src/
│   ├── __init__.py        # Makes src a Python package
│   ├── agents/            # Agent logic module
│   │   ├── __init__.py    # Initializes agents module
│   │   ├── base.py        # Base class for OpenAI agents
│   │   ├── distributor.py # Distributor agent implementation
│   │   ├── hospital.py    # Hospital agent implementation
│   │   └── manufacturer.py# Manufacturer agent implementation
│   ├── blockchain/        # Blockchain interaction module
│   │   ├── __init__.py    # Initializes blockchain module
│   │   ├── interface.py   # Web3.py interface for the smart contract
│   │   └── SupplyChainData.abi.json # Contract ABI (copied by deploy script)
│   ├── environment/       # Simulation environment module
│   │   ├── __init__.py    # Initializes environment module
│   │   ├── metrics.py     # Metrics calculation functions
│   │   └── supply_chain.py# Core simulation environment class
│   ├── llm/               # Large Language Model integration module
│   │   ├── __init__.py    # Initializes llm module
│   │   └── openai_integration.py # Handles calls to OpenAI API
│   ├── scenario/          # Pandemic scenario module
│   │   ├── __init__.py    # Initializes scenario module
│   │   ├── generator.py   # Generates pandemic scenarios (SIR model)
│   │   └── visualizer.py  # Creates plots for scenario data
│   └── tools/             # Agent helper tools module
│       ├── __init__.py    # Initializes tools module, defines Tools class
│       ├── allocation.py  # Allocation and order quantity tools
│       ├── assessment.py  # Criticality assessment tool
│       └── forecasting.py # Epidemic and disruption forecasting tools
├── test/                  # (Optional) Hardhat tests for the smart contract
└── venv/                  # Python virtual environment (created by user - Gitignored)
```

##  Improvements
*   Implement more complex smart contracts for other supply chain actions (e.g., tracking shipments, verifying authenticity). (May be)
*   Explore different LLMs.
*   Develop a web-based UI for running simulations and visualizing results.
*   Integrate with public testnets (like Sepolia) for more realistic deployment scenarios (requires test ETH and adjusting `hardhat.config.js`).
*   review metrics and calculate gas consumption
*   create a baseline for comparison, remove llms and use rule based allocation 

