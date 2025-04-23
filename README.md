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
    *   **Daily Case Update:** The environment calculates simulated cases and calls `updateRegionalCaseCount` on the contract.
    *   **Manufacturer Case Query:** The Manufacturer agent calls `get_blockchain_regional_cases_tool`, which uses the `BlockchainInterface` to call `getRegionalCaseCount` on the contract.
    *   **Fair Allocation Execution:** The environment calls `_calculate_fair_allocation`. If blockchain is enabled, this triggers the `BlockchainInterface` to call `executeFairAllocation` on the contract, passing manufacturer requests and available inventory. The contract performs the calculation using on-chain criticalities and case counts.
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
    *   This script compiles (if needed), deploys the `SupplyChainData` contract to the running Hardhat node, sets initial drug criticalities, saves the ABI to `src/blockchain/`, and **updates the `CONTRACT_ADDRESS` in your `.env` file.**
    *   Verify the output shows success and that your `.env` file now contains the `CONTRACT_ADDRESS`.

3.  **Run Python Simulation (Terminal 2 or 3):**
    *   In the same terminal as deployment (or a new one), **activate your Python virtual environment**:
        ```bash
        # Activate (macOS/Linux)
        source venv/bin/activate
        # OR Activate (Windows CMD/PowerShell)
        # venv\Scripts\activate
        ```
    *   Run the main script, enabling blockchain integration:
        ```bash
        # Example: 3 regions, 3 drugs, 10 days, verbose output
        python main.py --regions 3 --drugs 3 --days 10 --use-blockchain --verbose

        # Example: Longer run
        # python main.py --regions 5 --drugs 3 --days 50 --use-blockchain --verbose
        ```
    *   The `--use-blockchain` flag is crucial to enable interaction with the Hardhat node and smart contract.
    *   Use `--verbose` to see detailed agent decisions and blockchain logs.

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
    *   Review the final results summary (stockouts, impact, service level).
    *   Check the final "Querying Final Blockchain State" section for consistency.
2.  **Hardhat Node Console Output (Terminal 1):**
    *   Observe `eth_` JSON-RPC calls corresponding to Python interactions.
    *   Look for transaction mining confirmations or error messages if transactions revert.
3.  **Output Folder (`output_<timestamp>_..._blockchain/`):**
    *   **`simulation_report_openai_blockchain.html`:** Open in a browser for a complete, colorized log of the simulation run.
    *   **`.png` files:** Analyze the generated plots for supply chain performance, inventory levels, epidemic curves, etc.

## Configuration (`.env`)

Key variables in the `.env` file:

*   `OPENAI_API_KEY`: Your secret key for OpenAI API access.
*   `PRIVATE_KEY`: The private key of the account used to send transactions on the *local* Hardhat network (obtained from `npx hardhat node`).
*   `NODE_URL`: The RPC URL of the blockchain node (defaults to `http://127.0.0.1:8545` for local Hardhat).
*   `CONTRACT_ADDRESS`: The address of the deployed `SupplyChainData` contract (automatically updated by `npm run deploy:localhost`).

## Project Structure
├── contracts/
│ └── SupplyChainData.sol # Solidity smart contract
├── scripts/
│ └── deploy.js # Hardhat script to deploy contract
├── src/
│ ├── agents/ # LLM Agent implementations (base, manufacturer, etc.)
│ ├── blockchain/ # Blockchain interface (interface.py)
│ ├── environment/ # Simulation environment & metrics
│ ├── llm/ # OpenAI API integration
│ ├── scenario/ # Scenario generation & visualization
│ └── tools/ # Forecasting, allocation, assessment tools
├── test/ # (Optional) Hardhat tests for the contract
├── config.py # Simulation configuration, console setup
├── main.py # Main simulation script entry point
├── hardhat.config.js # Hardhat configuration
├── package.json # Node.js dependencies
├── requirements.txt # Python dependencies
├── .env # Environment variables (API Keys, Private Key - Gitignored)
├── .gitignore # Git ignore file
└── README.md # This file


## Future Work / Improvements

*   Implement more complex smart contracts for other supply chain actions (e.g., tracking shipments, verifying authenticity).
*   Allow agents to learn or adapt strategies over time.
*   Explore different LLMs or fine-tuning.
*   Develop a web-based UI for running simulations and visualizing results.
*   Integrate with public testnets (like Sepolia) for more realistic deployment scenarios (requires test ETH and adjusting `hardhat.config.js`).

