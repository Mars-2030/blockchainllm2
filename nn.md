Okay, let's modify the codebase to fully implement Strategy 1: **Blockchain as the Trusted Source of Truth for Regional Case Counts**.

Here's the plan:

1.  **Modify `BlockchainInterface`**: Ensure `get_regional_case_count` handles potential contract call errors gracefully. (Minor change, likely already okay).
2.  **Modify `PandemicSupplyChainTools` & Add Tool**: Add a new tool method specifically for querying blockchain case data.
3.  **Modify `OpenAIPandemicLLMAgent` (Base Agent)**:
    *   Store the `BlockchainInterface` instance.
    *   Add a method to run the new blockchain query tool.
4.  **Modify Agent Initialization (`main.py`, `manufacturer.py`, etc.)**: Pass the `BlockchainInterface` instance to agents when created.
5.  **Modify `PandemicSupplyChainEnvironment`**: Remove `current_cases` and `case_trend` from the `epidemiological_data` section of agent observations (especially manufacturer). Keep `projected_demand`.
6.  **Modify `ManufacturerAgent`**:
    *   Call the new tool in `decide`.
    *   Update LLM prompts (implicitly, by changing observation content).
    *   Update fallback logic and rule-based adjustments (`_make_production_decisions`, `_make_allocation_decisions`) to use the case counts retrieved from the blockchain tool instead of the (now removed) observation data.

---

**Modified Files:**

```python
# --- START OF FILE config.py ---
# No changes needed in config.py, assuming it's already correct for blockchain setup.
# Content omitted for brevity...
# --- END OF FILE config.py ---

# --- START OF FILE hardhat.config.js ---
# No changes needed.
# Content omitted for brevity...
# --- END OF FILE hardhat.config.js ---

# --- START OF FILE src/blockchain/interface.py ---

"""
Blockchain interface using Web3.py to interact with the SupplyChainData smart contract.
"""

import json
import time
import math
from web3 import Web3
from web3.middleware import geth_poa_middleware # For PoA networks like Sepolia, Goerli, maybe Ganache/Hardhat sometimes
from typing import Dict, List, Optional, Any
from rich.console import Console

# Use a shared console or create one if needed
console = Console()

class BlockchainInterface:
    """Handles communication with the SupplyChainData Ethereum smart contract."""

    def __init__(self, node_url: str, contract_address: str, contract_abi_path: str, private_key: Optional[str] = None):
        """
        Initializes the connection to the blockchain and loads the contract.

        Args:
            node_url: URL of the Ethereum node (e.g., "http://127.0.0.1:8545").
            contract_address: Deployed address of the SupplyChainData contract.
            contract_abi_path: Path to the JSON ABI file of the contract.
            private_key: Private key of the account used to send transactions (e.g., updating cases). Optional for read-only interactions.
        """
        self.node_url = node_url
        self.contract_address = contract_address
        self.private_key = private_key
        self.w3 = None
        self.contract = None
        self.account = None
        self.gas_limit = 3000000  # Default gas limit, adjust as needed
        self.gas_price_strategy = "medium" # or "fast", "slow", or a specific Wei value

        try:
            self.w3 = Web3(Web3.HTTPProvider(node_url))

            # Inject PoA middleware if needed (common for testnets and some private nets)
            # Check connection first before trying middleware
            if not self.w3.is_connected():
                 raise ConnectionError(f"Failed to connect to Ethereum node at {node_url}")

            # Try injecting middleware - might be needed for Hardhat/Ganache/Testnets
            try:
                self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
            except Exception as e:
                console.print(f"[yellow]Could not inject PoA middleware (may not be needed): {e}[/]")


            console.print(f"[green]Connected to Ethereum node: {node_url} (Chain ID: {self.w3.eth.chain_id})[/]")

            # Load Contract ABI
            with open(contract_abi_path, 'r') as f:
                contract_abi = json.load(f)

            # Load Contract
            checksum_address = self.w3.to_checksum_address(contract_address)
            self.contract = self.w3.eth.contract(address=checksum_address, abi=contract_abi)
            console.print(f"[green]SupplyChainData contract loaded at address: {contract_address}[/]")

            # Set up account for transactions if private key is provided
            if private_key:
                if not private_key.startswith('0x'):
                    private_key = '0x' + private_key
                self.account = self.w3.eth.account.from_key(private_key)
                self.w3.eth.default_account = self.account.address
                console.print(f"[green]Transaction account set up: {self.account.address}[/]")
            else:
                console.print("[yellow]No private key provided. Only read operations possible.[/]")

            # Test contract connection (optional read call)
            try:
                 owner = self.contract.functions.owner().call()
                 console.print(f"[dim]Contract owner found: {owner}[/dim]")
            except Exception as e:
                 console.print(f"[yellow]Warning: Could not call contract 'owner' function. Contract may not be deployed correctly or ABI mismatch. Error: {e}[/]")


        except FileNotFoundError:
            console.print(f"[bold red]Error: Contract ABI file not found at {contract_abi_path}[/]")
            raise
        except ConnectionError as e:
            console.print(f"[bold red]Error connecting to Ethereum node: {e}[/]")
            raise
        except Exception as e:
            console.print(f"[bold red]Error initializing BlockchainInterface: {e}[/]")
            console.print_exception(show_locals=True)
            raise

    def _get_gas_price(self):
        """Gets gas price based on strategy or network conditions."""
        # For local nodes (Hardhat/Ganache), gas price is often negligible or fixed
        if self.w3.eth.chain_id == 1337 or self.w3.eth.chain_id == 31337: # Common local chain IDs
            return self.w3.to_wei('10', 'gwei') # A reasonable default for local testing
        try:
             # Use eth_gasPrice for simplicity on testnets/mainnet
             return self.w3.eth.gas_price
        except Exception as e:
            console.print(f"[yellow]Could not fetch gas price, using default 10 gwei. Error: {e}[/]")
            return self.w3.to_wei('10', 'gwei')

    def _send_transaction(self, function_call) -> Optional[Dict[str, Any]]:
        """Builds, signs, sends a transaction and waits for the receipt."""
        if not self.account:
            console.print("[bold red]Error: Cannot send transaction. No private key configured.[/]")
            return None
        try:
            nonce = self.w3.eth.get_transaction_count(self.account.address)
            gas_price = self._get_gas_price()

            tx_params = {
                'from': self.account.address,
                'nonce': nonce,
                'gas': self.gas_limit,
                'gasPrice': gas_price,
            }

            # Build transaction
            transaction = function_call.build_transaction(tx_params)

            # Sign transaction
            signed_tx = self.w3.eth.account.sign_transaction(transaction, self.private_key)

            # Send transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            console.print(f"[dim]Transaction sent: {tx_hash.hex()}. Waiting for receipt...[/dim]")

            # Wait for receipt
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180) # 3 min timeout

            if tx_receipt['status'] == 1:
                console.print(f"[green]✓ Transaction successful! Block: {tx_receipt['blockNumber']}, Gas Used: {tx_receipt['gasUsed']}[/]")
                return {'status': 'success', 'receipt': tx_receipt}
            else:
                console.print(f"[bold red]❌ Transaction failed! Receipt: {tx_receipt}[/]")
                return {'status': 'failed', 'receipt': tx_receipt, 'error': 'Transaction reverted'}

        except Exception as e:
            console.print(f"[bold red]Error sending transaction: {e}[/]")
            # console.print_exception(show_locals=False) # Optional: Add more detailed traceback
            return {'status': 'error', 'error': str(e)}


    # --- Write Methods ---

    def update_regional_case_count(self, region_id: int, cases: int) -> Optional[Dict[str, Any]]:
        """Updates the case count for a region via a transaction."""
        try:
            function_call = self.contract.functions.updateRegionalCaseCount(region_id, cases)
            return self._send_transaction(function_call)
        except Exception as e:
            console.print(f"[red]Error preparing updateRegionalCaseCount transaction: {e}[/]")
            return {'status': 'error', 'error': str(e)}

    def set_drug_criticality(self, drug_id: int, criticality_value: int) -> Optional[Dict[str, Any]]:
        """Sets the drug criticality value via a transaction (likely used during setup)."""
        try:
            function_call = self.contract.functions.setDrugCriticality(drug_id, criticality_value)
            return self._send_transaction(function_call)
        except Exception as e:
            console.print(f"[red]Error preparing setDrugCriticality transaction: {e}[/]")
            return {'status': 'error', 'error': str(e)}

    def execute_fair_allocation(self, drug_id: int, region_ids: List[int], requested_amounts: List[float], available_inventory: float) -> Optional[Dict[int, float]]:
        """
        Triggers the fair allocation logic on the smart contract.

        Args:
            drug_id: ID of the drug.
            region_ids: List of requesting region IDs.
            requested_amounts: List of corresponding requested amounts (float).
            available_inventory: Total available inventory (float).

        Returns:
            A dictionary {region_id: allocated_amount (float)} if successful, else None.
            Amounts are converted back from the contract's integer representation.
        """
        try:
            # Convert float amounts to integers for the contract (e.g., scale by 1000 for 3 decimal places)
            # IMPORTANT: Choose a consistent scaling factor for amounts
            SCALE_FACTOR = 1000
            requested_amounts_int = [int(round(r * SCALE_FACTOR)) for r in requested_amounts]
            available_inventory_int = int(round(available_inventory * SCALE_FACTOR))

            if available_inventory_int <= 0:
                 console.print("[yellow]execute_fair_allocation: Available inventory is zero, skipping blockchain call.[/]")
                 return {r_id: 0.0 for r_id in region_ids} # Return zero allocations

            function_call = self.contract.functions.executeFairAllocation(
                drug_id, region_ids, requested_amounts_int, available_inventory_int
            )

            # **Workaround for Simulation:** Use call() to get the return value directly.
            # CAVEAT: This does *not* actually execute the state change or emit events on the chain.
            console.print(f"[dim]Simulating executeFairAllocation call for Drug {drug_id}...[/dim]")
            simulated_from_address = self.account.address if self.account else self.w3.eth.accounts[0] if self.w3.eth.accounts else None
            if simulated_from_address is None:
                console.print("[yellow]Warning: No account available for simulating call, allocation may fail if contract requires sender.[/]")
                allocated_amounts_int = function_call.call() # Try without sender
            else:
                 allocated_amounts_int = function_call.call({'from': simulated_from_address})
            console.print(f"[dim]Contract call simulation returned (int): {allocated_amounts_int}[/dim]")


            # Convert integer amounts back to floats
            allocated_amounts_float = {
                region_ids[i]: float(alloc) / SCALE_FACTOR
                for i, alloc in enumerate(allocated_amounts_int)
            }

            # Send the actual transaction to emit events (fire-and-forget for simulation return value)
            tx_result = self._send_transaction(function_call)
            if not tx_result or tx_result.get('status') != 'success':
                 console.print(f"[yellow]Warning: Transaction for executeFairAllocation failed or was not sent. Allocation based on simulation call result.[/]")
                 # Continue with the simulated result anyway for the simulation flow

            return allocated_amounts_float

        except Exception as e:
            console.print(f"[red]Error calling/preparing executeFairAllocation: {e}[/]")
            # Fallback: Return None to signal failure
            return None

    # --- Read Methods ---

    def get_regional_case_count(self, region_id: int) -> Optional[int]:
        """Reads the latest case count for a region from the contract."""
        try:
            # Add a retry mechanism for reads as nodes can be temporarily unavailable
            for attempt in range(3):
                try:
                    count = self.contract.functions.getRegionalCaseCount(region_id).call()
                    return count
                except Exception as read_e:
                    if attempt < 2:
                        console.print(f"[yellow]Retrying getRegionalCaseCount for region {region_id} after error: {read_e}[/]")
                        time.sleep(1) # Short delay before retry
                    else:
                        raise read_e # Raise error after final attempt
            return None # Should not be reached if raise happens
        except Exception as e:
            console.print(f"[red]Error reading getRegionalCaseCount for region {region_id} after retries: {e}[/]")
            return None # Return None on failure

    def get_drug_criticality(self, drug_id: int) -> Optional[int]:
        """Reads the drug criticality value from the contract."""
        try:
            return self.contract.functions.getDrugCriticality(drug_id).call()
        except Exception as e:
            console.print(f"[red]Error reading getDrugCriticality for drug {drug_id}: {e}[/]")
            return None

    def get_contract_owner(self) -> Optional[str]:
        """Reads the owner address from the contract."""
        try:
            return self.contract.functions.owner().call()
        except Exception as e:
            console.print(f"[red]Error reading contract owner: {e}[/]")
            return None

    def print_contract_state(self, num_regions: int = 5, num_drugs: int = 3):
        """Queries and prints some key states from the contract for debugging."""
        console.rule("[cyan]Querying Final Blockchain State[/cyan]")
        try:
            owner = self.get_contract_owner()
            console.print(f"Contract Owner: {owner}")

            console.print("\n[bold]Regional Case Counts:[/bold]")
            for r_id in range(num_regions):
                cases = self.get_regional_case_count(r_id)
                console.print(f"  Region {r_id}: {cases if cases is not None else '[red]Error[/]'}")

            console.print("\n[bold]Drug Criticalities:[/bold]")
            for d_id in range(num_drugs):
                 crit = self.get_drug_criticality(d_id)
                 console.print(f"  Drug {d_id}: {crit if crit is not None else '[red]Error[/]'}")

        except Exception as e:
            console.print(f"[red]Error querying final contract state: {e}[/]")
        console.rule()
# --- END OF FILE src/blockchain/interface.py ---

# --- START OF FILE src/tools/__init__.py ---

"""
Tools for the pandemic supply chain simulation agents.

This package provides various tools for forecasting, allocation, and situation assessment
that can be used by supply chain agents to make informed decisions.
"""
from typing import Dict, List, Optional

from src.tools.forecasting import epidemic_forecast_tool, disruption_prediction_tool
from src.tools.allocation import allocation_priority_tool, optimal_order_quantity_tool
from src.tools.assessment import criticality_assessment_tool
# Import BlockchainInterface for type hinting if needed, handle import error
try:
    from src.blockchain.interface import BlockchainInterface
except ImportError:
    BlockchainInterface = None


# For backwards compatibility, provide the PandemicSupplyChainTools class
class PandemicSupplyChainTools:
    """
    Collection of decision-making tools for supply chain agents.
    This class provides a unified interface to the various tool functions.
    """

    @staticmethod
    def epidemic_forecast_tool(current_cases, case_history, days_to_forecast=14):
        """Forecasts epidemic progression."""
        return epidemic_forecast_tool(current_cases, case_history, days_to_forecast)

    @staticmethod
    def disruption_prediction_tool(historical_disruptions, current_day, look_ahead_days=14):
        """Predicts likelihood of disruptions."""
        return disruption_prediction_tool(historical_disruptions, current_day, look_ahead_days)

    @staticmethod
    def allocation_priority_tool(drug_info, region_requests, region_cases, available_inventory):
        """Determines optimal allocation."""
        return allocation_priority_tool(drug_info, region_requests, region_cases, available_inventory)

    @staticmethod
    def optimal_order_quantity_tool(inventory_level, pipeline_quantity, daily_demand_forecast, lead_time=3, safety_stock_factor=1.5):
        """Calculates optimal order quantity."""
        return optimal_order_quantity_tool(inventory_level, pipeline_quantity, daily_demand_forecast, lead_time, safety_stock_factor)

    @staticmethod
    def criticality_assessment_tool(drug_info, stockout_history, unfulfilled_demand, total_demand):
        """Assesses situation criticality."""
        return criticality_assessment_tool(drug_info, stockout_history, unfulfilled_demand, total_demand)

    # --- NEW TOOL ---
    @staticmethod
    def get_blockchain_regional_cases_tool(
        blockchain_interface: Optional[BlockchainInterface],
        num_regions: int
    ) -> Optional[Dict[int, int]]:
        """
        Queries the blockchain for the latest case counts for all regions.

        Args:
            blockchain_interface: The initialized BlockchainInterface object.
            num_regions: The total number of regions in the simulation.

        Returns:
            A dictionary {region_id: case_count} if successful, else None.
            Returns None if blockchain interface is not available or query fails.
        """
        if blockchain_interface is None:
            # console.print("[yellow]Blockchain tool called, but interface is None.[/]") # Agent should print this
            return None

        regional_cases = {}
        all_successful = True
        for region_id in range(num_regions):
            case_count = blockchain_interface.get_regional_case_count(region_id)
            if case_count is None:
                # Log error within the interface, return default value (e.g., 0) here?
                # For now, let's signal failure by setting flag and potentially returning partial data or None
                all_successful = False
                regional_cases[region_id] = 0 # Default to 0 on error for agent use
            else:
                regional_cases[region_id] = case_count

        if not all_successful:
            # Decide: return partial data with defaults, or None?
            # Returning partial data with defaults might be more robust for agent logic.
            # console.print("[yellow]Blockchain tool: Failed to retrieve case counts for some regions. Using 0 default for failed regions.[/]")
            pass # Let agent handle logging

        return regional_cases


# Export the functions directly for direct import
__all__ = [
    'PandemicSupplyChainTools',
    'epidemic_forecast_tool',
    'disruption_prediction_tool',
    'allocation_priority_tool',
    'optimal_order_quantity_tool',
    'criticality_assessment_tool',
    'get_blockchain_regional_cases_tool' # Export new tool
]
# --- END OF FILE src/tools/__init__.py ---

# --- START OF FILE src/agents/base.py ---

"""
Base class for LLM-powered agents in the pandemic supply chain simulation.
"""

import json
from typing import Dict, List, Any, Optional # Added Optional
import numpy as np

from config import Colors
# Import BlockchainInterface for type hinting if needed, handle import error
try:
    from src.blockchain.interface import BlockchainInterface
except ImportError:
    BlockchainInterface = None
# Import the tools class
from src.tools import PandemicSupplyChainTools


class OpenAIPandemicLLMAgent:
    """
    Base class for LLM-powered agents using OpenAI API.
    """

    def __init__(
        self,
        agent_type: str,
        agent_id: Any, # Can be int (region) or 0 (manufacturer)
        tools: PandemicSupplyChainTools, # Expect the class instance now potentially
        openai_integration,
        memory_length: int = 10,
        verbose: bool = True,
        console=None,
        blockchain_interface: Optional[BlockchainInterface] = None # Add blockchain interface
    ):
        self.agent_type = agent_type
        self.agent_id = agent_id
        self.tools = tools # Store the tools instance
        self.openai = openai_integration
        self.memory_length = memory_length
        self.verbose = verbose
        self.console = console
        self.blockchain = blockchain_interface # Store the blockchain interface instance
        self.num_regions = 0 # Will be set if needed (e.g., by manufacturer)
        self.memory = []
        self.known_disruptions = []
        self.disruption_predictions = {} # Added to store tool output

    def _print(self, message):
        """Helper to safely print using the stored console if verbose."""
        if self.verbose and self.console:
            self.console.print(message)

    def add_to_memory(self, observation: Dict):
        """Add current observation to memory."""
        self.memory.append(observation)
        if len(self.memory) > self.memory_length:
            self.memory.pop(0)
        # Update known disruptions based on current observation
        if "disruptions" in observation:
             current_disruptions_keys = {(d['type'], d.get('drug_id', d.get('region_id', -1)), d['start_day']) # Use -1 default for target_id
                                    for d in observation['disruptions'] if isinstance(d, dict)} # Ensure d is a dict
             # Keep known disruptions that are still active or started recently
             cutoff_day = observation.get('day', 0) - 30 # Keep history for 30 days? Adjust as needed
             self.known_disruptions = [d for d in self.known_disruptions
                                      if isinstance(d, dict) and (d.get('end_day', -1) >= observation.get('day', 0) or d.get('start_day', -1) >= cutoff_day)]
             # Add new unique disruptions from current observation
             existing_keys = {(d['type'], d.get('drug_id', d.get('region_id', -1)), d.get('start_day'))
                              for d in self.known_disruptions if isinstance(d, dict)}
             for d in observation['disruptions']:
                  if isinstance(d, dict): # Check if d is a dict before accessing keys
                    key = (d.get('type'), d.get('drug_id', d.get('region_id', -1)), d.get('start_day'))
                    if key not in existing_keys:
                        self.known_disruptions.append(d)


    def decide(self, observation: Dict) -> Dict:
        """Make a decision (placeholder, override in subclasses)."""
        if self.verbose:
            agent_name = self._get_agent_name()
            agent_color = self._get_agent_color()
            self._print(f"\n[{agent_color}]🤔 {agent_name} making decision (Day {observation.get('day', '?')})...[/]")
        self.add_to_memory(observation)
        # Subclasses will implement the actual decision logic
        return {}

    def _get_agent_color(self):
        if self.agent_type == "manufacturer": return Colors.MANUFACTURER
        elif self.agent_type == "distributor": return Colors.DISTRIBUTOR
        elif self.agent_type == "hospital": return Colors.HOSPITAL
        return Colors.WHITE

    def _get_agent_name(self):
        if self.agent_type == "manufacturer": return "Manufacturer"
        elif self.agent_type == "distributor": return f"Distributor (Region {self.agent_id})"
        elif self.agent_type == "hospital": return f"Hospital (Region {self.agent_id})"
        return f"Agent {self.agent_id}"

    def _clean_observation_for_prompt(self, observation: Dict, max_len: int = 3500) -> str:
        """Creates a string representation of the observation, trying to keep it concise."""
        # --- IMPORTANT CHANGE: Remove specific fields we don't want LLM to rely on directly ---
        # Make a deep copy to avoid modifying the original observation used by rules/fallbacks
        cleaned_obs = json.loads(json.dumps(observation)) # Simple deep copy

        # Remove current_cases and case_trend from epidemiological_data if it exists
        if 'epidemiological_data' in cleaned_obs:
            for region_id_str in cleaned_obs['epidemiological_data']:
                if isinstance(cleaned_obs['epidemiological_data'][region_id_str], dict):
                    cleaned_obs['epidemiological_data'][region_id_str].pop('current_cases', None)
                    cleaned_obs['epidemiological_data'][region_id_str].pop('case_trend', None)
                    # Keep projected_demand if present

        # Simplify drug info
        if 'drug_info' in cleaned_obs:
            cleaned_obs['drug_info'] = {k: {'name': v.get('name'), 'crit': v.get('criticality_value'), 'demand_factor': v.get('base_demand')}
                                        for k, v in cleaned_obs.get('drug_info', {}).items()} # Use .get() for safety

        # Simplify history fields - keep fewer items
        history_limit = 5 # Reduced history limit
        for key in ['recent_orders', 'recent_allocations', 'demand_history', 'stockout_history', 'pending_releases']:
            if key in cleaned_obs and isinstance(cleaned_obs[key], list):
                cleaned_obs[key] = cleaned_obs[key][-history_limit:]

        # Round floats in numerical dicts/lists to 1 decimal place
        # Use a recursive function for safer rounding in nested structures
        def round_nested_floats(item):
            if isinstance(item, dict):
                return {k: round_nested_floats(v) for k, v in item.items()}
            elif isinstance(item, list):
                return [round_nested_floats(elem) for elem in item]
            elif isinstance(item, (float, np.floating)):
                return round(item, 1)
            elif isinstance(item, (int, np.integer)):
                 return int(item) # Keep ints as ints
            else:
                return item

        cleaned_obs = round_nested_floats(cleaned_obs)

        # Convert to JSON string, then potentially truncate if too long
        try:
            # Sort keys for consistency, handle potential non-serializable types
            json_string = json.dumps(cleaned_obs, indent=None, separators=(',', ':'), default=str, sort_keys=True)
            if len(json_string) > max_len:
                 # Basic truncation
                 truncated_json = json_string[:max_len]
                 # Try to find the last comma or brace to make it potentially valid-ish
                 last_sep = max(truncated_json.rfind(','), truncated_json.rfind('}'), truncated_json.rfind(']'))
                 if last_sep > 0:
                     return truncated_json[:last_sep] + '...[TRUNCATED]"}'
                 else: # Fallback if no separator found in truncated part
                     return truncated_json + '...[TRUNCATED]"}'
            return json_string
        except TypeError as e:
            self._print(f"[yellow]Could not serialize observation to JSON for prompt: {e}. Using basic string representation.[/]")
            fallback_str = f'"day": {observation.get("day")}, "inventories": {observation.get("inventories")}, "...(error serializing)"'
            return "{" + fallback_str[:max_len] + "}"

    def _create_decision_prompt(self, observation: Dict, decision_type: str) -> str:
        """Create a detailed prompt for the OpenAI LLM with enhanced guidance."""
        current_day = observation.get("day", "N/A")
        agent_name = self._get_agent_name()

        role_desc = f"You are the {agent_name} in a pandemic supply chain simulation."
        if self.agent_type == "manufacturer":
             role_desc += " Your tasks are production planning and allocating drugs to regional distributors."
        elif self.agent_type == "distributor":
             role_desc += f" Your tasks are ordering drugs from the manufacturer and allocating available drugs to the hospital in your Region {self.agent_id}."
        elif self.agent_type == "hospital":
             role_desc += f" Your task is ordering drugs from the distributor in Region {self.agent_id} to meet patient demand."


        json_guidance = "Respond ONLY with a valid JSON object with string keys (drug IDs, or region IDs if nested) and numerical amounts."
        if decision_type == "production": task = f"Determine optimal production quantity for each drug. {json_guidance} Example: {{\"0\": 1000.0, \"1\": 500.0}}"
        elif decision_type == "allocation" and self.agent_type == "manufacturer": task = f"Determine optimal allocation amounts of AVAILABLE drugs to each requesting regional distributor. {json_guidance} Example: {{\"0\": {{\"0\": 100.0, \"1\": 50.0}}, \"1\": {{...}} }}"
        elif decision_type == "allocation" and self.agent_type == "distributor": task = f"Determine optimal allocation amounts of AVAILABLE drugs to the hospital in Region {self.agent_id}. {json_guidance} Example: {{\"0\": 100.0, \"1\": 50.0}}"
        elif decision_type == "order" and self.agent_type == "distributor": task = f"Determine optimal order quantity for each drug to request from the manufacturer. {json_guidance} Example: {{\"0\": 1000.0, \"1\": 500.0}}"
        elif decision_type == "order" and self.agent_type == "hospital": task = f"Determine optimal order quantity for each drug to request from the distributor. {json_guidance} Example: {{\"0\": 100.0, \"1\": 50.0}}"
        else: task = f"Determine the appropriate {decision_type} actions. {json_guidance}"


        considerations = f"""
        Consider the following factors:
        - Current day ({current_day}) and overall pandemic timeline.
        - Current inventory levels (own stock, warehouse if applicable). Check available amounts carefully.
        - Inbound and outbound pipeline quantities (drugs in transit).
        - Epidemiological context: Use projected_demand figures. NOTE: Current regional case counts are NOT in the JSON below; assume you have access to trusted case counts from an external source (blockchain query tool) if needed for prioritization, but base quantities primarily on projected_demand and inventory/pipeline status.
        - Drug characteristics (criticality, base demand factor). Prioritize critical drugs.
        - Active and predicted supply chain disruptions (manufacturing, transportation). Increase buffers if risk is high.
        - Recent orders/allocations history (use mainly for context).
        """
        # Add agent/decision specific considerations
        if self.agent_type == "manufacturer":
            if decision_type == "allocation":
                considerations += """
        - Manufacturer Allocation: PRIORITIZE DOWNSTREAM NEEDS. Base allocations on projected demand across regions (see downstream_projected_demand_summary, estimate regional share using projected_demand). Be MORE GENEROUS if downstream inventory (downstream_inventory_summary) is low relative to projected demand OR if trusted case counts (from external tool) are high/rising, especially for critical drugs. Allocate ONLY AVAILABLE inventory from your 'inventories' field. Use fair allocation principles (considering trusted cases and criticality) if supply is limited. Adhere to batching schedules (is_batch_processing_day, days_to_next_batch_process).
                """
            elif decision_type == "production":
                 considerations += """
        - Manufacturer Production: PRODUCE PROACTIVELY. Base production heavily on the SUM of projected demand across all regions (downstream_projected_demand_summary) and anticipate future growth indicated by rising trusted case counts (from external tool). Build warehouse buffers ahead of surges, especially for critical drugs or before batch allocation days. Consider production capacity and active disruptions.
                 """
        elif self.agent_type == "distributor":
             if decision_type == "allocation":
                  considerations += f"""
        - Distributor Allocation: Allocate ONLY to your single hospital (Region {self.agent_id}) based on their likely need (use projected_demand). Allocate ONLY AVAILABLE 'inventories'.
                 """
             elif decision_type == "order":
                  considerations += """
        - Distributor Ordering: Aim for sufficient stock to cover lead time demand + safety stock. PRIORITIZE FUTURE DEMAND. Use hospital's projected_demand and consider regional case trends (obtained externally if needed). If cases are rising, order SIGNIFICANTLY MORE than current needs suggest to build safety stock proactively. Consider manufacturer lead times and potential transport disruptions.
                 """
        elif self.agent_type == "hospital":
             if decision_type == "order":
                  considerations += """
        - Hospital Ordering: Order to meet patient demand. PRIORITIZE FUTURE DEMAND. Use your projected_demand and consider case trends (obtained externally if needed). If cases are rising, order SIGNIFICANTLY MORE than current needs suggest to build safety stock proactively. Consider distributor lead times and potential transport disruptions.
                 """

        final_instruction = "Based *only* on the information provided above (and assuming access to trusted external case data), determine the decision. Respond ONLY with the valid JSON object specified in the task, without any additional text, explanations, or markdown formatting."

        observation_summary = self._clean_observation_for_prompt(observation)

        full_prompt = f"""
        {role_desc}

        Current Simulation Day: {current_day}

        Your Task: {task}

        Key Considerations:
        {considerations}

        Current Situation & Data (JSON format, possibly truncated, current cases excluded):
        {observation_summary}

        {final_instruction}
        """
        # self._print(f"DEBUG: Generated Prompt:\n{full_prompt}") # Uncomment for debugging prompts
        return full_prompt

    # --- Tool execution methods ---

    def _run_epidemic_forecast_tool(self, observation: Dict) -> List[float]:
        """Run epidemic forecasting tool."""
        # This tool now primarily uses projected demand from observation if available,
        # as current cases are removed from the direct observation for the prompt.
        # If projected_demand is not available, it might need a different fallback.
        tool_name = "Epidemic Forecast (Demand Based)"
        tool_result = []
        try:
            # Manufacturer: Average projected demands
            if self.agent_type == "manufacturer":
                # Use the downstream summary from the observation
                forecast_input = []
                if "downstream_projected_demand_summary" in observation:
                     demands = [float(v) for v in observation["downstream_projected_demand_summary"].values() if v is not None]
                     if demands:
                          avg_proj_demand = np.mean(demands)
                          # Simple projection: repeat average projected demand
                          # A more sophisticated approach could use the trend from the blockchain cases
                          forecast_input = [avg_proj_demand] * 14 # Default forecast length
                     else: forecast_input = [0.0] * 14 # Default if no demand data
                else: forecast_input = [0.0] * 14
                tool_result = forecast_input # Simplified: just return the repeated average

            # Distributor or Hospital: Use their specific projected demand
            elif "epidemiological_data" in observation:
                region_proj_demand = observation["epidemiological_data"].get("projected_demand", {})
                # Average the projected demand across drugs for a general forecast? Or forecast per drug?
                # Let's return a constant forecast based on average projected demand for simplicity
                avg_proj_demand = 0.0
                count = 0
                for drug_id, demand in region_proj_demand.items():
                     try: avg_proj_demand += float(demand); count += 1
                     except (ValueError, TypeError): continue
                if count > 0: avg_proj_demand /= count

                tool_result = [avg_proj_demand] * 14 # Repeat average projection
            else:
                 tool_result = [0.0] * 14 # Default

            if self.verbose:
                 print_res = [f"{x:.1f}" for x in tool_result[:7]]
                 self._print(f"[{Colors.TOOL_OUTPUT}][TOOL]{self._get_agent_name()} - {tool_name} Output (first 7d):[/] {print_res}{'...' if len(tool_result) > 7 else ''}")

        except Exception as e:
             self._print(f"[red]Error running {tool_name} tool for {self.agent_type} {self.agent_id}: {e}[/]")
             tool_result = [0.0] * 14 # Default on error
        return tool_result

    def _run_disruption_prediction_tool(self, observation: Dict) -> Dict:
        """Run disruption prediction tool."""
        tool_name = "Disruption Prediction"
        tool_result = {"manufacturing": {}, "transportation": {}}
        try:
             current_day = observation.get("day", 0)
             tool_result = self.tools.disruption_prediction_tool(self.known_disruptions, current_day)
             self.disruption_predictions = tool_result # Store the prediction

             if self.verbose:
                  non_zero_preds = {
                      "manufacturing": {k: f"{v:.2f}" for k, v in tool_result.get("manufacturing", {}).items() if v > 0.01},
                      "transportation": {k: f"{v:.2f}" for k, v in tool_result.get("transportation", {}).items() if v > 0.01},
                  }
                  self._print(f"[{Colors.TOOL_OUTPUT}][TOOL]{self._get_agent_name()} - {tool_name} Output (Prob > 0.01):[/] {non_zero_preds if any(non_zero_preds.values()) else '{}'}")
        except Exception as e:
             self._print(f"[red]Error running {tool_name} tool for {self.agent_type} {self.agent_id}: {e}[/]")
             if self.verbose: self._print(f"[{Colors.TOOL_OUTPUT}][TOOL]{self._get_agent_name()} - {tool_name} Output:[/] [red]Error - Using empty dict.[/]")
             tool_result = {"manufacturing": {}, "transportation": {}}
             self.disruption_predictions = tool_result # Store empty prediction on error
        return tool_result

    def _run_allocation_priority_tool(self, drug_info: Dict, region_requests: Dict, region_cases: Dict, available: float) -> Dict:
        """Run allocation priority tool."""
        tool_name = "Allocation Priority"
        tool_result = {}
        fallback_result = {}
        try:
            int_key_requests = {int(k): float(v) for k, v in region_requests.items() if str(k).isdigit() and v is not None}
            # Ensure region_cases keys are integers for the tool
            int_key_cases = {int(k): float(v) for k, v in region_cases.items() if str(k).isdigit() and v is not None}

            valid_drug_info = drug_info if isinstance(drug_info, dict) else {}
            tool_result = self.tools.allocation_priority_tool(valid_drug_info, int_key_requests, int_key_cases, max(0.0, float(available)))

            if self.verbose:
                print_res = {k: f"{v:.1f}" for k, v in tool_result.items()}
                self._print(f"[{Colors.TOOL_OUTPUT}][TOOL]{self._get_agent_name()} - {tool_name} Output:[/] {print_res}")

        except Exception as e:
            self._print(f"[red]Error running {tool_name} tool for {self.agent_type} {self.agent_id}: {e}[/]")
            num_requesters = len(region_requests)
            if num_requesters > 0 and available > 0:
                 positive_requests = {r: req for r, req in region_requests.items() if req > 0}
                 num_positive_requesters = len(positive_requests)
                 if num_positive_requesters > 0:
                      equal_share = available / num_positive_requesters
                      fallback_result = {r: min(req, equal_share) for r, req in positive_requests.items()}
                 else: fallback_result = {}
            else: fallback_result = {}

            if self.verbose:
                 print_res = {k: f"{v:.1f}" for k, v in fallback_result.items()}
                 self._print(f"[{Colors.TOOL_OUTPUT}][TOOL]{self._get_agent_name()} - {tool_name} Output:[/] [red]Error - Using fallback allocation:[/]{print_res}")
            tool_result = fallback_result
        return tool_result


    def _run_optimal_order_quantity_tool(self, inventory: float, pipeline: float, demand_forecast: List[float], lead_time: int = 3, criticality: float = 1.0) -> float:
        """Run optimal order quantity tool."""
        tool_name = "Optimal Order Quantity"
        tool_result = 0.0
        try:
            safety_stock_factor = 1.0 + ((criticality - 1) / 3.0) * 1.0
            safety_stock_factor = max(1.0, safety_stock_factor)
            cleaned_forecast = [float(d) for d in demand_forecast if isinstance(d, (int, float, np.number))]
            tool_result = self.tools.optimal_order_quantity_tool(
                float(inventory), float(pipeline), cleaned_forecast, int(lead_time), float(safety_stock_factor)
                )

            if self.verbose:
                self._print(f"[{Colors.TOOL_OUTPUT}][TOOL]{self._get_agent_name()} - {tool_name} Output:[/] {tool_result:.1f}")
        except Exception as e:
            self._print(f"[red]Error running {tool_name} tool for {self.agent_type} {self.agent_id}: {e}[/]")
            if self.verbose: self._print(f"[{Colors.TOOL_OUTPUT}][TOOL]{self._get_agent_name()} - {tool_name} Output:[/] [red]Error - Returning 0.0[/]")
            tool_result = 0.0
        return tool_result

    def _run_criticality_assessment_tool(self, drug_info: Dict, stockout_history: List[Dict], unfulfilled: float, total: float) -> Dict:
        """Run criticality assessment tool."""
        tool_name = "Criticality Assessment"
        fallback_assessment = {
                 "drug_name": drug_info.get("name", "Unknown"),
                 "criticality_score": 0, "category": "Error",
                 "stockout_days_recent": len(stockout_history) if isinstance(stockout_history, list) else 0,
                 "unfulfilled_percentage_recent": 0,
                 "recommendations": ["Assessment tool failed."]
             }
        tool_result = fallback_assessment
        try:
            stockout_history = stockout_history if isinstance(stockout_history, list) else []
            tool_result = self.tools.criticality_assessment_tool(
                drug_info, stockout_history, float(unfulfilled), max(1.0, float(total))
            )

            if self.verbose:
                print_res = f"Score: {tool_result.get('criticality_score', 'N/A')}, Category: {tool_result.get('category', 'N/A')}"
                self._print(f"[{Colors.TOOL_OUTPUT}][TOOL]{self._get_agent_name()} - {tool_name} (Drug: {drug_info.get('name', '?')}) Output:[/] {print_res}")
        except Exception as e:
             self._print(f"[red]Error running {tool_name} tool for {self.agent_type} {self.agent_id}: {e}[/]")
             if self.verbose: self._print(f"[{Colors.TOOL_OUTPUT}][TOOL]{self._get_agent_name()} - {tool_name} (Drug: {drug_info.get('name', '?')}) Output:[/] [red]Error - Using fallback assessment.[/]")
             tool_result = fallback_assessment
        return tool_result

    # --- NEW TOOL RUNNER ---
    def _run_blockchain_regional_cases_tool(self) -> Optional[Dict[int, int]]:
        """
        Runs the tool to query regional case counts from the blockchain.
        Requires self.blockchain and self.num_regions to be set.
        """
        tool_name = "Blockchain Regional Cases"
        tool_result = None
        if not self.blockchain:
             if self.verbose:
                 self._print(f"[{Colors.BLOCKCHAIN}]Skipping {tool_name} tool: Blockchain interface not available.[/]")
             return None # Cannot run without blockchain

        if not hasattr(self, 'num_regions') or self.num_regions <= 0:
             # This might happen if called by non-manufacturer before num_regions is known? Add safeguards.
             # Try to get num_regions from memory if possible as a fallback
             if self.memory:
                 last_obs = self.memory[-1]
                 if 'downstream_projected_demand_summary' in last_obs: # Manufacturer specific field
                     # Infer num_regions from length of downstream summaries or epi data if present
                     if 'epidemiological_data' in last_obs:
                         self.num_regions = len(last_obs['epidemiological_data'])
                 elif 'epidemiological_data' in last_obs: # Distributor/Hospital
                      # Less reliable, assume agent_id relates if not manufacturer
                      # This part is tricky, relying on num_regions being passed correctly is better
                      pass # Avoid setting num_regions incorrectly here

             if not hasattr(self, 'num_regions') or self.num_regions <= 0:
                 self._print(f"[red]Error running {tool_name} tool: Number of regions unknown for {self._get_agent_name()}.[/]")
                 return None

        try:
            # Call the static tool method, passing the blockchain interface
            tool_result = self.tools.get_blockchain_regional_cases_tool(
                blockchain_interface=self.blockchain,
                num_regions=self.num_regions
            )

            if self.verbose:
                if tool_result is not None:
                    print_res = {k: f"{v}" for k, v in tool_result.items()} # Show exact counts
                    self._print(f"[{Colors.TOOL_OUTPUT}][TOOL]{self._get_agent_name()} - {tool_name} Output:[/] {print_res}")
                else:
                     self._print(f"[{Colors.TOOL_OUTPUT}][TOOL]{self._get_agent_name()} - {tool_name} Output:[/] [red]Failed to retrieve data[/]")

        except Exception as e:
             self._print(f"[red]Error running {tool_name} tool for {self.agent_type} {self.agent_id}: {e}[/]")
             if self.verbose: self._print(f"[{Colors.TOOL_OUTPUT}][TOOL]{self._get_agent_name()} - {tool_name} Output:[/] [red]Error - Returning None[/]")
             tool_result = None
        return tool_result


# --- END OF FILE src/agents/base.py ---

# --- START OF FILE src/agents/manufacturer.py ---

"""
Manufacturer agent implementation for the pandemic supply chain simulation.
"""

from typing import Dict, List, Optional
import numpy as np
import json

from .base import OpenAIPandemicLLMAgent
from config import Colors
# Import BlockchainInterface for type hinting if needed, handle import error
try:
    from src.blockchain.interface import BlockchainInterface
except ImportError:
    BlockchainInterface = None
# Import the tools class
from src.tools import PandemicSupplyChainTools


class ManufacturerAgent(OpenAIPandemicLLMAgent):
    """LLM-powered manufacturer agent using OpenAI."""

    def __init__(self, tools: PandemicSupplyChainTools, openai_integration, num_regions: int, memory_length=10, verbose=True, console=None, blockchain_interface: Optional[BlockchainInterface] = None):
        super().__init__("manufacturer", 0, tools, openai_integration, memory_length, verbose, console=console, blockchain_interface=blockchain_interface)
        self.num_regions = num_regions # Store number of regions

    def decide(self, observation: Dict) -> Dict:
        """Make production and allocation decisions using OpenAI."""
        self.add_to_memory(observation) # Store current state

        # --- Query Blockchain for Trusted Data ---
        blockchain_cases = None
        if self.blockchain: # Check if blockchain is enabled for this agent
             blockchain_cases = self._run_blockchain_regional_cases_tool()
             if blockchain_cases is None and self.verbose:
                 self._print(f"[{Colors.FALLBACK}]Manufacturer failed to get cases from Blockchain. Fallback logic will use defaults/projections.[/]")
        # Use a default if blockchain query failed
        if blockchain_cases is None:
             blockchain_cases = {r: 0 for r in range(self.num_regions)} # Default to 0 if unavailable

        # --- Use Other Tools (Run predictions first) ---
        epidemic_forecast_tool_output = self._run_epidemic_forecast_tool(observation) # Now demand-based
        disruption_predictions = self._run_disruption_prediction_tool(observation)

        # --- Production Decision ---
        production_decisions = self._make_production_decisions(observation, disruption_predictions, blockchain_cases)

        # --- Allocation Decision ---
        # Pass blockchain_cases to allocation decision method
        allocation_decisions = self._make_allocation_decisions(observation, disruption_predictions, blockchain_cases)

        return {
            "manufacturer_production": production_decisions,
            "manufacturer_allocation": allocation_decisions
        }

    def _make_production_decisions(self, observation: Dict, disruption_predictions: Dict, blockchain_cases: Dict[int, int]) -> Dict:
        """Determine production quantities using OpenAI API, with enhanced rules using blockchain cases."""
        decision_type = "production"
        # Prompt now implicitly uses observation where current_cases are removed
        prompt = self._create_decision_prompt(observation, decision_type)

        structured_decision = self.openai.generate_structured_decision(prompt, decision_type)

        if self.verbose:
            self._print(f"[{Colors.LLM_DECISION}][LLM Raw Decision ({self._get_agent_name()} - {decision_type})][/] {structured_decision}")

        production_decisions = {}
        num_drugs = len(observation.get("drug_info", {}))
        llm_success = False

        if structured_decision and isinstance(structured_decision, dict):
             processed_llm = {}
             for drug_id_key, amount in structured_decision.items():
                 try:
                      drug_id = int(drug_id_key)
                      if 0 <= drug_id < num_drugs:
                           prod_amount = max(0.0, float(amount))
                           processed_llm[drug_id] = prod_amount
                      else:
                           if self.verbose: self._print(f"[yellow]Skipping invalid drug_id '{drug_id_key}' in {decision_type} decision.[/]")
                 except (ValueError, TypeError) as e:
                      if self.verbose: self._print(f"[yellow]Error processing {decision_type} decision item '{drug_id_key}': {amount} -> {e}. Skipping.[/]")

             if processed_llm:
                 llm_success = True
                 production_decisions = processed_llm
                 for drug_id in list(production_decisions.keys()):
                     capacity = observation.get("production_capacity", {}).get(str(drug_id), 0)
                     production_decisions[drug_id] = min(production_decisions[drug_id], capacity)


        if not llm_success:
             if self.verbose:
                 self._print(f"[{Colors.FALLBACK}][FALLBACK] LLM {self.agent_type} {decision_type} decision failed/invalid. Using fallback: Max capacity.[/]")
             for drug_id in range(num_drugs):
                 capacity = observation.get("production_capacity", {}).get(str(drug_id), 0)
                 production_decisions[drug_id] = capacity

        decisions_before_rules = production_decisions.copy()
        rules_applied_flag = False

        # --- Apply Rule-Based Overrides/Adjustments ---

        # 1. Forecasting-based scaling (MODIFIED: Use blockchain cases)
        # Estimate trend crudely from total cases now vs simple avg? Or just scale based on total?
        # Let's scale based on total current cases from blockchain relative to some baseline.
        total_blockchain_cases = sum(blockchain_cases.values())
        # Simplistic: If total cases exceed N * 1000, boost production? (Needs better logic)
        # Alternative: Use projection summary directly
        total_projected_demand = sum(float(v) for v in observation.get("downstream_projected_demand_summary", {}).values() if v is not None)
        # Compare projected demand to total capacity?
        total_capacity = sum(float(c) for c in observation.get("production_capacity", {}).values() if c is not None)

        production_scale_factor = 1.0
        if total_capacity > 0 and total_projected_demand > total_capacity * 0.8: # If proj demand high relative to capacity
            production_scale_factor = 1.2 # Boost production slightly
            if self.verbose:
                self._print(f"[{Colors.RULE}][RULE ADJUSTMENT] {self._get_agent_name()} - {decision_type}: Scaling up due to high projected demand vs capacity (factor: {production_scale_factor:.2f}).[/]")
                rules_applied_flag = True
        elif total_blockchain_cases > self.num_regions * 500: # Simple case threshold scaling
             production_scale_factor = 1.1
             if self.verbose:
                 self._print(f"[{Colors.RULE}][RULE ADJUSTMENT] {self._get_agent_name()} - {decision_type}: Scaling up due to high total blockchain cases ({total_blockchain_cases}) (factor: {production_scale_factor:.2f}).[/]")
                 rules_applied_flag = True

        if abs(production_scale_factor - 1.0) > 0.01:
             for drug_id in production_decisions:
                 capacity = observation.get("production_capacity", {}).get(str(drug_id), 0)
                 scaled_prod = production_decisions[drug_id] * production_scale_factor
                 production_decisions[drug_id] = min(scaled_prod, capacity)

        # 2. Disruption-aware buffer planning (Existing, unchanged)
        for drug_id in list(production_decisions.keys()):
            disruption_risk = disruption_predictions.get("manufacturing", {}).get(str(drug_id), 0)
            if disruption_risk > 0.1:
                 capacity = observation.get("production_capacity", {}).get(str(drug_id), 0)
                 disruption_factor = (1 + 3 * disruption_risk)
                 if self.verbose:
                     self._print(f"[{Colors.RULE}][RULE ADJUSTMENT] {self._get_agent_name()} - {decision_type} (Drug {drug_id}): Applying disruption buffer (risk: {disruption_risk:.2f}, factor: {disruption_factor:.2f}).[/]")
                     rules_applied_flag = True
                 disruption_adjusted_prod = production_decisions[drug_id] * disruption_factor
                 production_decisions[drug_id] = min(disruption_adjusted_prod, capacity)

        # 3. Warehouse Buffer Adjustments (Existing, unchanged conceptually)
        for drug_id in list(production_decisions.keys()):
            manu_inv = observation.get("inventories", {}).get(str(drug_id), 0)
            wh_inv = observation.get("warehouse_inventories", {}).get(str(drug_id), 0)
            total_inv = manu_inv + wh_inv
            capacity = observation.get("production_capacity", {}).get(str(drug_id), 0)

            adjustment_factor = 1.0
            if capacity > 0:
                inv_days_cover = total_inv / capacity if capacity > 1 else total_inv
                if inv_days_cover > 7: adjustment_factor = 0.7
                elif inv_days_cover > 4: adjustment_factor = 0.9
                elif inv_days_cover < 1.5: adjustment_factor = 1.5

            if abs(adjustment_factor - 1.0) > 0.01:
                 if self.verbose:
                     self._print(f"[{Colors.RULE}][RULE ADJUSTMENT] {self._get_agent_name()} - {decision_type} (Drug {drug_id}): Applying warehouse buffer adjustment (cover: {inv_days_cover:.1f}d, factor: {adjustment_factor:.2f}).[/]")
                     rules_applied_flag = True
                 adjusted_prod = production_decisions[drug_id] * adjustment_factor
                 min_prod = capacity * 0.2
                 production_decisions[drug_id] = min(max(adjusted_prod, min_prod), capacity)

        # 4. Batch Allocation Awareness (Existing, unchanged)
        if "batch_allocation_frequency" in observation:
            days_to_next_batch = observation.get("days_to_next_batch_process", 0) # Use correct key
            batch_freq = observation.get("batch_allocation_frequency", 1)
            if batch_freq > 1 and days_to_next_batch <= 2:
                batch_boost_factor = 1.2
                if self.verbose:
                     self._print(f"[{Colors.RULE}][RULE ADJUSTMENT] {self._get_agent_name()} - {decision_type}: Boosting production before batch day (factor: {batch_boost_factor:.2f}).[/]")
                     rules_applied_flag = True
                for drug_id in production_decisions:
                    capacity = observation.get("production_capacity", {}).get(str(drug_id), 0)
                    batch_boosted_prod = production_decisions[drug_id] * batch_boost_factor
                    production_decisions[drug_id] = min(batch_boosted_prod, capacity)

        # --- Print final adjusted decision ---
        if self.verbose and rules_applied_flag:
             print_before = {k: f"{v:.1f}" for k, v in decisions_before_rules.items()}
             print_after = {k: f"{v:.1f}" for k, v in production_decisions.items()}
             if json.dumps(print_before) != json.dumps(print_after):
                 self._print(f"[{Colors.RULE}][RULE FINAL] {self._get_agent_name()} - {decision_type} After Rules:[/]\n  Before: {print_before}\n   After: {print_after}")
             else:
                  self._print(f"[{Colors.DECISION}][FINAL Decision] {self._get_agent_name()} - {decision_type} (Rules checked, no change):[/] {print_after}")
        elif self.verbose:
             print_after = {k: f"{v:.1f}" for k, v in production_decisions.items()}
             self._print(f"[{Colors.DECISION}][FINAL Decision] {self._get_agent_name()} - {decision_type}:[/] {print_after}")

        return {int(k): v for k, v in production_decisions.items()}


    def _make_allocation_decisions(self, observation: Dict, disruption_predictions: Dict, blockchain_cases: Dict[int, int]) -> Dict:
        """Determine allocation quantities using OpenAI API, using blockchain cases for rules/fallback."""
        decision_type = "allocation"
        prompt = self._create_decision_prompt(observation, decision_type) # Prompt implicitly uses cleaned obs

        structured_decision = self.openai.generate_structured_decision(prompt, decision_type)

        if self.verbose:
            self._print(f"[{Colors.LLM_DECISION}][LLM Raw Decision ({self._get_agent_name()} - {decision_type})][/] {structured_decision}")

        allocation_decisions = {} # Stores {drug_id: {region_id: amount}}
        num_drugs = len(observation.get("drug_info", {}))
        # num_regions is already available as self.num_regions
        llm_success = False

        if structured_decision and isinstance(structured_decision, dict):
             processed_llm = {}
             for drug_id_key, region_allocs in structured_decision.items():
                 try:
                      drug_id = int(drug_id_key)
                      if 0 <= drug_id < num_drugs and isinstance(region_allocs, dict):
                           drug_allocs = {}
                           for region_id_key, amount in region_allocs.items():
                                try:
                                     region_id = int(region_id_key)
                                     if 0 <= region_id < self.num_regions:
                                          alloc_amount = max(0.0, float(amount))
                                          drug_allocs[region_id] = alloc_amount
                                     else:
                                         if self.verbose: self._print(f"[yellow]Skipping invalid region_id '{region_id_key}' in allocation for Drug {drug_id}.[/]")
                                except (ValueError, TypeError):
                                     if self.verbose: self._print(f"[yellow]Error processing allocation amount for Drug {drug_id}, Region '{region_id_key}': {amount}. Skipping.[/]")
                           if drug_allocs:
                               processed_llm[drug_id] = drug_allocs
                 except (ValueError, TypeError) as e:
                      if self.verbose: self._print(f"[yellow]Error processing allocation decision item '{drug_id_key}': {region_allocs} -> {e}. Skipping.[/]")

             if processed_llm:
                 llm_success = True
                 allocation_decisions = processed_llm


        if not llm_success:
             if self.verbose:
                 self._print(f"[{Colors.FALLBACK}][FALLBACK] LLM {self.agent_type} {decision_type} decision failed/invalid. Using fallback: Fair allocation based on projected demand & blockchain cases.[/]")

             allocation_decisions = {}
             lookahead_days = 7

             for drug_id in range(num_drugs):
                 available_inventory = observation.get("inventories", {}).get(str(drug_id), 0)
                 if available_inventory <= 0: continue

                 drug_info = observation.get("drug_info", {}).get(str(drug_id), {})

                 # --- MODIFIED: Use blockchain cases for priority, projected demand for quantity ---
                 region_needs = {} # Estimate 'need' based on projected demand
                 total_proj_demand = observation.get("downstream_projected_demand_summary", {}).get(str(drug_id))
                 total_blockchain_cases = sum(blockchain_cases.values())

                 if total_proj_demand is not None:
                     # Use projected demand summary and distribute by *case proportion*
                     for region_id in range(self.num_regions):
                          current_bc_cases = blockchain_cases.get(region_id, 0)
                          case_proportion = (current_bc_cases / total_blockchain_cases) if total_blockchain_cases > 0 else (1 / self.num_regions if self.num_regions > 0 else 0)
                          estimated_daily_regional_demand = total_proj_demand * case_proportion
                          total_need = estimated_daily_regional_demand * lookahead_days
                          region_needs[region_id] = max(0, total_need) # Need based on projection share
                 else:
                     # Fallback: Estimate need based purely on cases * demand factor (less ideal)
                     if self.verbose: self._print(f"[yellow]Fallback allocation for Drug {drug_id}: Downstream demand summary missing, estimating from blockchain cases.[/]")
                     base_demand_per_1k_cases = drug_info.get("base_demand", 10) / 1000
                     for region_id in range(self.num_regions):
                          current_bc_cases = blockchain_cases.get(region_id, 0)
                          estimated_daily_regional_demand = current_bc_cases * base_demand_per_1k_cases
                          total_need = estimated_daily_regional_demand * lookahead_days
                          region_needs[region_id] = max(0, total_need)

                 if not region_needs or available_inventory <= 0: continue

                 # Call the allocation tool using the blockchain cases for priority
                 # Note: region_needs (based on projection) is passed as 'requests'
                 fair_allocations = self._run_allocation_priority_tool(
                     drug_info, region_needs, blockchain_cases, available_inventory
                 )

                 if fair_allocations:
                    if drug_id not in allocation_decisions: allocation_decisions[drug_id] = {}
                    allocation_decisions[drug_id].update(fair_allocations)


        decisions_before_rules = {
            drug_id: regs.copy() for drug_id, regs in allocation_decisions.items()
        } if allocation_decisions else {}
        rules_applied_flag = False

        # --- Apply Rule-Based Overrides/Adjustments ---

        # Proactive Allocation Rule (MODIFIED: Check trend based on blockchain cases)
        critical_downstream_days = 5
        proactive_allocation_factor = 0.3
        # Estimate trend from blockchain cases history (requires storing history) - Simplification: Boost if total cases > threshold
        is_overall_trend_positive = sum(blockchain_cases.values()) > self.num_regions * 300 # Simple threshold trigger

        if is_overall_trend_positive:
            for drug_id_str, summary_data in observation.get("downstream_inventory_summary", {}).items():
                try:
                    drug_id = int(drug_id_str)
                    current_manu_inv = observation.get("inventories", {}).get(drug_id_str, 0)
                    if current_manu_inv <= 0: continue

                    total_downstream_inv = summary_data.get("total_downstream", 0)
                    total_downstream_proj_demand = observation.get("downstream_projected_demand_summary", {}).get(drug_id_str, 0)

                    if total_downstream_proj_demand > 0:
                        days_cover = total_downstream_inv / total_downstream_proj_demand

                        if days_cover < critical_downstream_days:
                            proactive_amount_to_allocate = current_manu_inv * proactive_allocation_factor

                            if proactive_amount_to_allocate > 1:
                                 if self.verbose:
                                     self._print(f"[{Colors.RULE}][RULE ADJUSTMENT] {self._get_agent_name()} - {decision_type} (Drug {drug_id}): Proactively allocating {proactive_amount_to_allocate:.1f} units due to low downstream cover ({days_cover:.1f}d < {critical_downstream_days}d) & high cases.[/]")
                                     rules_applied_flag = True

                                 # --- Estimate regional needs again using blockchain cases for distribution ---
                                 region_needs = {} # Estimate 'need' based on projected demand share by case proportion
                                 total_proj_demand_for_dist = observation.get("downstream_projected_demand_summary", {}).get(str(drug_id), 0)
                                 total_blockchain_cases_inner = sum(blockchain_cases.values())

                                 if total_proj_demand_for_dist is not None and total_blockchain_cases_inner > 0:
                                     for region_id_inner in range(self.num_regions):
                                         current_bc_cases_inner = blockchain_cases.get(region_id_inner, 0)
                                         case_proportion = current_bc_cases_inner / total_blockchain_cases_inner
                                         region_needs[region_id_inner] = max(0, total_proj_demand_for_dist * case_proportion)
                                 else: # Fallback estimation if needed
                                      drug_info = observation.get("drug_info", {}).get(str(drug_id), {})
                                      base_demand_per_1k_cases = drug_info.get("base_demand", 10) / 1000
                                      for region_id_inner in range(self.num_regions):
                                           current_bc_cases_inner = blockchain_cases.get(region_id_inner, 0)
                                           region_needs[region_id_inner] = max(0, current_bc_cases_inner * base_demand_per_1k_cases)


                                 proactive_fair_allocs = self._run_allocation_priority_tool(
                                     observation.get("drug_info", {}).get(str(drug_id), {}),
                                     region_needs,
                                     blockchain_cases, # Use BC cases for priority
                                     proactive_amount_to_allocate
                                 )

                                 if drug_id not in allocation_decisions: allocation_decisions[drug_id] = {}
                                 for region_id, amount in proactive_fair_allocs.items():
                                     current_alloc = allocation_decisions[drug_id].get(region_id, 0)
                                     allocation_decisions[drug_id][region_id] = current_alloc + amount

                except (ValueError, KeyError, TypeError) as e:
                    if self.verbose: self._print(f"[yellow]Warning during proactive allocation rule for drug {drug_id_str}: {e}[/]")
                    continue


        # Batch Allocation Adjustments (Existing - applied AFTER proactive rule)
        is_batch_day = observation.get("is_batch_processing_day", True) # Use correct key
        batch_freq = observation.get("batch_allocation_frequency", 1)
        if batch_freq > 1 and not is_batch_day: # Apply scale-down only if batching AND not a batch day
             if self.verbose:
                  self._print(f"[{Colors.RULE}][RULE ADJUSTMENT] {self._get_agent_name()} - {decision_type}: Not a batch day (Freq={batch_freq}d), scaling down non-critical allocations.[/]")
                  rules_applied_flag = True
             for drug_id in list(allocation_decisions.keys()):
                  drug_info = observation.get("drug_info", {}).get(str(drug_id), {})
                  is_critical = drug_info.get("criticality_value", 0) >= 4 # Check value
                  if not is_critical:
                     if drug_id in allocation_decisions:
                         for region_id in allocation_decisions[drug_id]:
                             allocation_decisions[drug_id][region_id] *= 0.25


        # --- Print final adjusted decision ---
        if self.verbose and rules_applied_flag:
             print_before = {dr_id: {r_id: f"{v:.1f}" for r_id, v in regs.items()} for dr_id, regs in decisions_before_rules.items()}
             print_after = {dr_id: {r_id: f"{v:.1f}" for r_id, v in regs.items()} for dr_id, regs in allocation_decisions.items()}
             if json.dumps(print_before) != json.dumps(print_after):
                 self._print(f"[{Colors.RULE}][RULE FINAL] {self._get_agent_name()} - {decision_type} After Rules:[/]\n  Before: {print_before}\n   After: {print_after}")
             else:
                  self._print(f"[{Colors.DECISION}][FINAL Decision] {self._get_agent_name()} - {decision_type} (Rules checked, no change):[/] {print_after}")
        elif self.verbose:
             print_after = {dr_id: {r_id: f"{v:.1f}" for r_id, v in regs.items()} for dr_id, regs in allocation_decisions.items()}
             self._print(f"[{Colors.DECISION}][FINAL Decision] {self._get_agent_name()} - {decision_type}:[/] {print_after}")


        # Ensure final decisions use integer keys and filter small amounts
        final_allocations = {}
        for drug_id, allocs in allocation_decisions.items():
            int_allocs = {int(k): v for k, v in allocs.items() if v > 0.01}
            if int_allocs:
                 final_allocations[int(drug_id)] = int_allocs
        return final_allocations


def create_openai_manufacturer_agent(
    tools: PandemicSupplyChainTools,
    openai_integration,
    num_regions: int, # Added num_regions
    memory_length=10,
    verbose=True,
    console=None,
    blockchain_interface: Optional[BlockchainInterface] = None # Added blockchain interface
):
    """Create a manufacturer agent powered by OpenAI."""
    return ManufacturerAgent(
        tools=tools,
        openai_integration=openai_integration,
        num_regions=num_regions, # Pass num_regions
        memory_length=memory_length,
        verbose=verbose,
        console=console,
        blockchain_interface=blockchain_interface # Pass interface
    )

# --- END OF FILE src/agents/manufacturer.py ---


# --- START OF FILE src/agents/distributor.py ---
# --- START OF FILE src/agents/distributor.py ---

"""
Distributor agent implementation for the pandemic supply chain simulation.
"""

from typing import Dict, List, Optional
from .base import OpenAIPandemicLLMAgent
from config import Colors

import json # Import json for checking rule changes
# Import BlockchainInterface for type hinting if needed, handle import error
try:
    from src.blockchain.interface import BlockchainInterface
except ImportError:
    BlockchainInterface = None
# Import the tools class
from src.tools import PandemicSupplyChainTools


class DistributorAgent(OpenAIPandemicLLMAgent):
    """LLM-powered distributor agent using OpenAI."""

    def __init__(
        self,
        region_id,
        tools: PandemicSupplyChainTools,
        openai_integration,
        num_regions: int, # Keep num_regions for calculating hospital ID
        memory_length=10,
        verbose=True,
        console=None,
        blockchain_interface: Optional[BlockchainInterface] = None # Add interface
        ):
        super().__init__(
            "distributor",
            region_id,
            tools,
            openai_integration,
            memory_length,
            verbose,
            console=console,
            blockchain_interface=blockchain_interface # Pass interface to base
        )
        # No need to store num_regions again if base class handles it, but keep for clarity if needed
        self.num_regions = num_regions

    def decide(self, observation: Dict) -> Dict:
        """Make ordering and allocation decisions using OpenAI."""
        self.add_to_memory(observation)

        # --- Use Tools (Run predictions first) ---
        # Distributor *could* query blockchain cases for its region, but might not be necessary
        # as its main drivers are hospital projected demand and its own inventory.
        # Let's keep it simpler and not have distributors query cases for now.
        # blockchain_cases_region = None
        # if self.blockchain:
        #     # Requires modification of the tool/runner to query specific region
        #     pass

        epidemic_forecast_tool_output = self._run_epidemic_forecast_tool(observation) # Now demand-based
        disruption_predictions = self._run_disruption_prediction_tool(observation)

        # --- Ordering decisions (to manufacturer) ---
        order_decisions = self._make_order_decisions(observation, epidemic_forecast_tool_output, disruption_predictions)

        # --- Allocation decisions (to hospital) ---
        allocation_decisions = self._make_allocation_decisions(observation, epidemic_forecast_tool_output, disruption_predictions)

        # Structure the output correctly
        return {
            "distributor_orders": {self.agent_id: order_decisions},
            "distributor_allocation": {self.agent_id: allocation_decisions}
        }

    def _make_order_decisions(self, observation: Dict, epidemic_forecast_tool_output: List[float], disruption_predictions: Dict) -> Dict:
        """Determine order quantities from manufacturer using OpenAI, with enhanced fallback and rules."""
        decision_type = "order"
        # Prompt uses cleaned observation (no direct cases/trend)
        prompt = self._create_decision_prompt(observation, decision_type)

        structured_decision = self.openai.generate_structured_decision(prompt, decision_type)

        if self.verbose:
            self._print(f"[{Colors.LLM_DECISION}][LLM Raw Decision ({self._get_agent_name()} - {decision_type})][/] {structured_decision}")

        order_decisions = {}
        num_drugs = len(observation.get("drug_info", {}))
        llm_success = False

        if structured_decision and isinstance(structured_decision, dict):
            processed_llm = {}
            for drug_id_key, amount in structured_decision.items():
                 try:
                     drug_id = int(drug_id_key)
                     if 0 <= drug_id < num_drugs:
                         order_amount = max(0.0, float(amount))
                         processed_llm[drug_id] = order_amount
                     else:
                         if self.verbose: self._print(f"[yellow]Skipping invalid drug_id '{drug_id_key}' in {self.agent_type} {decision_type} (Region {self.agent_id}).[/]")
                 except (ValueError, TypeError) as e:
                      if self.verbose: self._print(f"[yellow]Error processing {self.agent_type} {decision_type} item '{drug_id_key}': {amount} -> {e} (Region {self.agent_id}). Skipping.[/]")

            if processed_llm:
                llm_success = True
                order_decisions = processed_llm

        if not llm_success:
             if self.verbose:
                 self._print(f"[{Colors.FALLBACK}][FALLBACK] LLM {self.agent_type} {decision_type} decision failed/invalid (Region {self.agent_id}). Using fallback: Rule-based optimal order.[/]")
             # Fallback: Use rule-based optimal order tool (based on projected demand)
             for drug_id in range(num_drugs):
                 inventory = observation.get("inventories", {}).get(str(drug_id), 0)
                 pipeline = observation.get("inbound_pipeline", {}).get(str(drug_id), 0)
                 drug_info = observation.get("drug_info", {}).get(str(drug_id), {})
                 criticality = drug_info.get("criticality_value", 1)

                 # Use hospital's projected demand from observation
                 hospital_projected_demand = observation.get("epidemiological_data", {}).get("projected_demand", {}).get(str(drug_id), 0)
                 hospital_projected_demand = max(0, float(hospital_projected_demand))

                 # Estimate lead time based on disruption risk
                 transport_risk = disruption_predictions.get("transportation", {}).get(str(self.agent_id), 0)
                 base_lead_time = 3
                 lead_time = base_lead_time + int(round(transport_risk * 5))
                 lead_time = max(1, lead_time)

                 # Create forecast list based on *hospital's* projected demand.
                 demand_forecast_for_tool = [hospital_projected_demand] * (lead_time + 1)

                 order_qty = self._run_optimal_order_quantity_tool(
                     inventory, pipeline, demand_forecast_for_tool, lead_time, criticality
                 )
                 order_decisions[drug_id] = order_qty


        decisions_before_rules = order_decisions.copy()
        rules_applied_flag = False

        # --- Rule-Based Adjustments (No changes needed here, already based on risk/cover) ---

        # 1. Existing Disruption/Criticality Buffer
        for drug_id in list(order_decisions.keys()):
             drug_info = observation.get("drug_info", {}).get(str(drug_id), {})
             criticality = drug_info.get("criticality_value", 1)
             transport_risk = disruption_predictions.get("transportation", {}).get(str(self.agent_id), 0)
             if criticality >= 3 and transport_risk > 0.4:
                  buffer_factor = 1.3
                  if self.verbose:
                       self._print(f"[{Colors.RULE}][RULE ADJUSTMENT] {self._get_agent_name()} - {decision_type} (Drug {drug_id}): Applying disruption/criticality buffer (factor: {buffer_factor:.2f}).[/]")
                       rules_applied_flag = True
                  order_decisions[drug_id] *= buffer_factor

        # 2. Emergency Override based on distributor cover vs hospital projected demand
        for drug_id in list(order_decisions.keys()):
            inventory = observation.get("inventories", {}).get(str(drug_id), 0)
            pipeline = observation.get("inbound_pipeline", {}).get(str(drug_id), 0)
            hospital_proj_demand = observation.get("epidemiological_data", {}).get("projected_demand", {}).get(str(drug_id), 0)
            hospital_proj_demand = max(1e-6, float(hospital_proj_demand))

            inventory_position = inventory + pipeline
            days_cover = inventory_position / hospital_proj_demand

            emergency_boost_factor = 1.0
            if days_cover < 2: emergency_boost_factor = 2.0
            elif days_cover < 5: emergency_boost_factor = 1.3

            if abs(emergency_boost_factor - 1.0) > 0.01:
                if self.verbose:
                    self._print(f"[{Colors.RULE}][RULE ADJUSTMENT] {self._get_agent_name()} - {decision_type} (Drug {drug_id}): Applying low distributor cover EMERGENCY boost (Cover: {days_cover:.1f}d vs Hospital Demand, Factor: {emergency_boost_factor:.2f}).[/]")
                    rules_applied_flag = True
                order_decisions[drug_id] *= emergency_boost_factor


        # --- Print final adjusted decision ---
        if self.verbose and rules_applied_flag:
             print_before = {k: f"{v:.1f}" for k, v in decisions_before_rules.items()}
             print_after = {k: f"{v:.1f}" for k, v in order_decisions.items()}
             if json.dumps(print_before) != json.dumps(print_after):
                 self._print(f"[{Colors.RULE}][RULE FINAL] {self._get_agent_name()} - {decision_type} After Rules:[/]\n  Before: {print_before}\n   After: {print_after}")
             else:
                  self._print(f"[{Colors.DECISION}][FINAL Decision] {self._get_agent_name()} - {decision_type} (Rules checked, no change):[/] {print_after}")
        elif self.verbose:
             print_after = {k: f"{v:.1f}" for k, v in order_decisions.items()}
             self._print(f"[{Colors.DECISION}][FINAL Decision] {self._get_agent_name()} - {decision_type}:[/] {print_after}")

        return {int(k): v for k, v in order_decisions.items() if v > 0.01}



    def _make_allocation_decisions(self, observation: Dict, epidemic_forecast_tool_output: List[float], disruption_predictions: Dict) -> Dict:
        """Determine allocation to hospital using OpenAI."""
        decision_type = "allocation"
        # Prompt uses cleaned observation
        prompt = self._create_decision_prompt(observation, decision_type)

        structured_decision = self.openai.generate_structured_decision(prompt, decision_type)

        if self.verbose:
            self._print(f"[{Colors.LLM_DECISION}][LLM Raw Decision ({self._get_agent_name()} - {decision_type})][/] {structured_decision}")

        allocation_decisions = {}
        num_drugs = len(observation.get("drug_info", {}))
        llm_success = False

        if structured_decision and isinstance(structured_decision, dict):
             processed_llm = {}
             for drug_id_key, value in structured_decision.items():
                  try:
                      drug_id = int(drug_id_key)
                      if 0 <= drug_id < num_drugs:
                           alloc_amount = 0.0
                           # Simplified parsing assuming direct value or value under '0'/'agent_id' key
                           target_keys = ['0', str(self.agent_id)]
                           parsed_val = value
                           if isinstance(value, dict):
                               found = False
                               for t_key in target_keys:
                                   if t_key in value:
                                       parsed_val = value[t_key]; found = True; break
                               if not found and value: # Fallback: take first value if dict not empty
                                    parsed_val = next(iter(value.values()))

                           try: alloc_amount = max(0.0, float(parsed_val))
                           except (ValueError, TypeError):
                               if self.verbose: self._print(f"[{Colors.YELLOW}]{self.agent_type} {decision_type} Drug {drug_id} (Region {self.agent_id}): Cannot convert LLM value '{parsed_val}' to float. Allocating 0.[/]")
                           processed_llm[drug_id] = alloc_amount
                      else:
                           if self.verbose: self._print(f"[{Colors.YELLOW}]Skipping invalid drug_id '{drug_id_key}' in {self.agent_type} {decision_type} (Region {self.agent_id}).[/]")
                  except (ValueError, TypeError) as e:
                      if self.verbose: self._print(f"[{Colors.YELLOW}]Error processing {self.agent_type} {decision_type} key '{drug_id_key}' (Region {self.agent_id}): {e}. Skipping.[/]")

             if processed_llm:
                 llm_success = True
                 allocation_decisions = processed_llm

        if not llm_success:
             if self.verbose:
                 self._print(f"[{Colors.FALLBACK}][FALLBACK] LLM {self.agent_type} {decision_type} decision failed or invalid format (Region {self.agent_id}). Using fallback: Fulfill recent order/projected demand.[/]")
             # Fallback: Allocate based on recent hospital order or projected demand
             for drug_id in range(num_drugs):
                 inventory = observation.get("inventories", {}).get(str(drug_id), 0)
                 if inventory <= 0:
                      allocation_decisions[drug_id] = 0
                      continue

                 requested_amount = 0
                 recent_orders = observation.get("recent_orders", [])
                 # Calculate relevant hospital ID based on num_regions and distributor's region_id (agent_id)
                 hospital_id = self.num_regions + 1 + self.agent_id

                 hospital_orders_for_drug = [o for o in recent_orders if o.get("from_id") == hospital_id and o.get("drug_id") == drug_id]
                 if hospital_orders_for_drug:
                     requested_amount = sum(o.get("amount", 0) for o in hospital_orders_for_drug)
                 else:
                     # Fallback: Estimate demand using projected demand from observation
                     projected_demand = observation.get("epidemiological_data", {}).get("projected_demand", {}).get(str(drug_id))
                     if projected_demand is not None:
                         requested_amount = max(0, float(projected_demand))
                     # No further fallback needed as projected demand should always be present

                 allocation_decisions[drug_id] = min(max(0, requested_amount), inventory)


        # --- Apply Final Inventory Cap (No other rules applied here) ---
        final_capped_allocations = {}
        for drug_id, amount in allocation_decisions.items():
            inventory = observation.get("inventories", {}).get(str(drug_id), 0)
            final_capped_allocations[drug_id] = min(max(0, amount), inventory)


        # --- Print final adjusted decision ---
        if self.verbose:
             print_after = {k: f"{v:.1f}" for k, v in final_capped_allocations.items()}
             log_prefix = f"[{Colors.DECISION}][FINAL Decision]"
             if not llm_success:
                  log_prefix = f"[{Colors.FALLBACK}][FALLBACK FINAL Decision]"
             elif allocation_decisions != final_capped_allocations:
                  log_prefix = f"[{Colors.RULE}][CAPPED FINAL Decision]"

             self._print(f"{log_prefix} {self._get_agent_name()} - {decision_type}:[/] {print_after}")


        return {int(k): v for k, v in final_capped_allocations.items() if v > 0.01}


def create_openai_distributor_agent(
    region_id,
    tools: PandemicSupplyChainTools, # Pass tools instance
    openai_integration,
    num_regions: int,
    memory_length=10,
    verbose=True,
    console=None,
    blockchain_interface: Optional[BlockchainInterface] = None # Added interface
):
    """Create a distributor agent powered by OpenAI."""
    return DistributorAgent(
        region_id=region_id,
        tools=tools,
        openai_integration=openai_integration,
        num_regions=num_regions,
        memory_length=memory_length,
        verbose=verbose,
        console=console,
        blockchain_interface=blockchain_interface # Pass interface
        )

# --- END OF FILE src/agents/distributor.py ---

# --- START OF FILE src/agents/hospital.py ---
# --- START OF FILE src/agents/hospital.py ---

"""
Hospital agent implementation for the pandemic supply chain simulation.
"""

from typing import Dict, List, Optional
from .base import OpenAIPandemicLLMAgent
from config import Colors

import json
# Import BlockchainInterface for type hinting if needed, handle import error
try:
    from src.blockchain.interface import BlockchainInterface
except ImportError:
    BlockchainInterface = None
# Import the tools class
from src.tools import PandemicSupplyChainTools

class HospitalAgent(OpenAIPandemicLLMAgent):
    """LLM-powered hospital agent using OpenAI."""

    def __init__(
        self,
        region_id,
        tools: PandemicSupplyChainTools,
        openai_integration,
        memory_length=10,
        verbose=True,
        console=None,
        blockchain_interface: Optional[BlockchainInterface] = None # Add interface
        ):
        super().__init__(
            "hospital",
            region_id,
            tools,
            openai_integration,
            memory_length,
            verbose,
            console=console,
            blockchain_interface=blockchain_interface # Pass interface to base
            )
        # Hospitals typically don't need num_regions, but could be added if needed

    def decide(self, observation: Dict) -> Dict:
        """Make ordering decisions using OpenAI."""
        self.add_to_memory(observation)

        # --- Use Tools ---
        # Hospital doesn't need blockchain cases directly for its order decision logic (uses projected demand)
        epidemic_forecast_tool_output = self._run_epidemic_forecast_tool(observation) # Demand-based
        disruption_predictions = self._run_disruption_prediction_tool(observation)

        # --- Ordering decisions (to distributor) ---
        order_decisions = self._make_order_decisions(observation, epidemic_forecast_tool_output, disruption_predictions)

        # Structure the output correctly
        return {"hospital_orders": {self.agent_id: order_decisions}}

    def _make_order_decisions(self, observation: Dict, epidemic_forecast_tool_output: List[float], disruption_predictions: Dict) -> Dict:
        """Determine order quantities from distributor using OpenAI, with enhanced fallback and rules."""
        decision_type = "order"
        # Prompt uses cleaned observation
        prompt = self._create_decision_prompt(observation, decision_type)

        structured_decision = self.openai.generate_structured_decision(prompt, decision_type)

        if self.verbose:
            self._print(f"[{Colors.LLM_DECISION}][LLM Raw Decision ({self._get_agent_name()} - {decision_type})][/] {structured_decision}")

        order_decisions = {}
        num_drugs = len(observation.get("drug_info", {}))
        llm_success = False

        if structured_decision and isinstance(structured_decision, dict):
             processed_llm = {}
             for drug_id_key, amount in structured_decision.items():
                  try:
                      drug_id = int(drug_id_key)
                      if 0 <= drug_id < num_drugs:
                           order_amount = max(0.0, float(amount))
                           processed_llm[drug_id] = order_amount
                      else:
                          if self.verbose: self._print(f"[yellow]Skipping invalid drug_id '{drug_id_key}' in {self.agent_type} {decision_type} (Region {self.agent_id}).[/]")
                  except (ValueError, TypeError) as e:
                      if self.verbose: self._print(f"[yellow]Error processing {self.agent_type} {decision_type} item '{drug_id_key}': {amount} -> {e} (Region {self.agent_id}). Skipping.[/]")

             if processed_llm:
                llm_success = True
                order_decisions = processed_llm

        if not llm_success:
             if self.verbose:
                 self._print(f"[{Colors.FALLBACK}][FALLBACK] LLM {self.agent_type} {decision_type} decision failed/invalid (Region {self.agent_id}). Using fallback: Rule-based assessment & optimal order.[/]")
             # Fallback: Use criticality assessment and optimal order tool (based on projected demand)
             for drug_id in range(num_drugs):
                  inventory = observation.get("inventories", {}).get(str(drug_id), 0)
                  pipeline = observation.get("inbound_pipeline", {}).get(str(drug_id), 0)
                  drug_info = observation.get("drug_info", {}).get(str(drug_id), {})
                  criticality = drug_info.get("criticality_value", 1)

                  # Use hospital's own projected demand from observation
                  next_day_projected_demand = observation.get("epidemiological_data", {}).get("projected_demand", {}).get(str(drug_id), 0)
                  next_day_projected_demand = max(0, float(next_day_projected_demand))

                  # Estimate lead time based on disruption risk
                  transport_risk = disruption_predictions.get("transportation", {}).get(str(self.agent_id), 0)
                  base_lead_time = 1
                  lead_time = base_lead_time + int(round(transport_risk * 3))
                  lead_time = max(1, lead_time)

                  # Create forecast list based on own projected demand
                  demand_forecast_for_tool = [next_day_projected_demand] * (lead_time + 1)

                  order_qty = self._run_optimal_order_quantity_tool(
                      inventory, pipeline, demand_forecast_for_tool, lead_time, criticality
                  )
                  order_decisions[drug_id] = order_qty


        # --- Apply Rule-Based Adjustments (No changes needed, assessment based on local history) ---
        decisions_before_rules = order_decisions.copy()
        rules_applied_flag = False

        for drug_id in range(num_drugs):
             drug_info = observation.get("drug_info", {}).get(str(drug_id), {})
             stockout_hist = observation.get("stockout_history", [])
             demand_hist = observation.get("demand_history", [])
             stockout_hist = stockout_hist if isinstance(stockout_hist, list) else []
             demand_hist = demand_hist if isinstance(demand_hist, list) else []

             unfulfilled = sum(s.get('unfulfilled', 0) for s in stockout_hist if isinstance(s, dict))
             total_demand_hist = sum(d.get('demand', 0) for d in demand_hist if isinstance(d, dict))

             situation = self._run_criticality_assessment_tool(
                 drug_info, stockout_hist, unfulfilled, max(1, total_demand_hist)
             )

             crit_category = situation.get("category", "")
             base_multiplier = 1.0
             if crit_category == "Critical Emergency": base_multiplier = 3.0
             elif crit_category == "Severe Shortage": base_multiplier = 2.0
             elif crit_category == "Moderate Concern": base_multiplier = 1.5

             emergency_boost = 1.0
             if crit_category == "Critical Emergency": emergency_boost = 2.5
             elif crit_category == "Severe Shortage": emergency_boost = 1.8

             final_multiplier = max(base_multiplier, emergency_boost)

             if abs(final_multiplier - 1.0) > 0.01:
                 if self.verbose:
                     reason = "EMERGENCY override boost" if final_multiplier == emergency_boost and emergency_boost > base_multiplier else "criticality assessment multiplier"
                     self._print(f"[{Colors.RULE}][RULE ADJUSTMENT] {self._get_agent_name()} - {decision_type} (Drug {drug_id}): Applying {reason} (Category: {crit_category}, Factor: {final_multiplier:.2f}).[/]")
                     rules_applied_flag = True
                 current_order = order_decisions.get(drug_id, 0)
                 order_decisions[drug_id] = current_order * final_multiplier


        # --- Print final adjusted decision ---
        if self.verbose and rules_applied_flag:
             print_before = {k: f"{v:.1f}" for k, v in decisions_before_rules.items()}
             print_after = {k: f"{v:.1f}" for k, v in order_decisions.items()}
             if json.dumps(print_before) != json.dumps(print_after):
                 self._print(f"[{Colors.RULE}][RULE FINAL] {self._get_agent_name()} - {decision_type} After Rules:[/]\n  Before: {print_before}\n   After: {print_after}")
             else:
                 self._print(f"[{Colors.DECISION}][FINAL Decision] {self._get_agent_name()} - {decision_type} (Rules checked, no change):[/] {print_after}")
        elif self.verbose:
             print_after = {k: f"{v:.1f}" for k, v in order_decisions.items()}
             self._print(f"[{Colors.DECISION}][FINAL Decision] {self._get_agent_name()} - {decision_type}:[/] {print_after}")

        return {int(k): v for k, v in order_decisions.items() if v > 0.01}


def create_openai_hospital_agent(
    region_id,
    tools: PandemicSupplyChainTools, # Pass tools instance
    openai_integration,
    memory_length=10,
    verbose=True,
    console=None,
    blockchain_interface: Optional[BlockchainInterface] = None # Added interface
):
    """Create a hospital agent powered by OpenAI."""
    return HospitalAgent(
        region_id=region_id,
        tools=tools,
        openai_integration=openai_integration,
        memory_length=memory_length,
        verbose=verbose,
        console=console,
        blockchain_interface=blockchain_interface # Pass interface
        )

# --- END OF FILE src/agents/hospital.py ---


# --- START OF FILE src/environment/supply_chain.py ---

"""
Pandemic supply chain environment simulation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import math
import json # For potential debug printing

# Import the specific allocation tool needed for the fallback logic
from src.tools.allocation import allocation_priority_tool

# Import the BlockchainInterface if needed (type hinting, instantiation)
try:
    from src.blockchain.interface import BlockchainInterface
except ImportError:
    BlockchainInterface = None # Allow running without blockchain dependencies if not installed

class PandemicSupplyChainEnvironment:
    """Simulates a pandemic supply chain with manufacturers, distributors, and hospitals."""

    def __init__(
        self,
        scenario_generator,
        initial_manufacturer_inventory: float = 5000,
        initial_distributor_inventory: float = 2000,
        initial_hospital_inventory: float = 500,
        initial_warehouse_inventory: float = 0, # Warehouse starts empty
        blockchain_interface: Optional[BlockchainInterface] = None, # Modified type hint
        use_blockchain: bool = False,
        console = None # Rich console instance
    ):
        self.scenario = scenario_generator
        self.num_regions = len(scenario_generator.regions)
        self.num_drugs = len(scenario_generator.drugs)
        self.scenario_length = scenario_generator.scenario_length
        self.current_day = 0
        self.console = console

        # Store blockchain info
        self.blockchain = blockchain_interface
        self.use_blockchain = use_blockchain and self.blockchain is not None

        # Initialize inventories
        self.inventories = {} # {drug_id: {node_id: quantity}}
        for drug_id in range(self.num_drugs):
            self.inventories[drug_id] = {0: initial_manufacturer_inventory} # Manufacturer node_id = 0
            for region_id in range(self.num_regions):
                dist_id = region_id + 1 # Distributor node_ids = 1 to num_regions
                hosp_id = self.num_regions + 1 + region_id # Hospital node_ids = num_regions+1 to 2*num_regions
                self.inventories[drug_id][dist_id] = initial_distributor_inventory
                self.inventories[drug_id][hosp_id] = initial_hospital_inventory

        self.warehouse_inventories = {drug_id: initial_warehouse_inventory for drug_id in range(self.num_drugs)}
        self.pipelines = {}
        node_ids = list(range(2 * self.num_regions + 1))
        for from_id in node_ids:
            self.pipelines[from_id] = {}
            for to_id in node_ids:
                 if from_id != to_id:
                    self.pipelines[from_id][to_id] = {drug_id: [] for drug_id in range(self.num_drugs)}

        self.stockouts = {drug_id: {r: 0 for r in range(self.num_regions)} for drug_id in range(self.num_drugs)}
        self.unfulfilled_demand = {drug_id: {r: 0.0 for r in range(self.num_regions)} for drug_id in range(self.num_drugs)}
        self.total_demand = {drug_id: {r: 0.0 for r in range(self.num_regions)} for drug_id in range(self.num_drugs)}
        self.patient_impact = {r: 0.0 for r in range(self.num_regions)}

        self.demand_history = []
        self.order_history = []
        self.allocation_history = []
        self.stockout_history = []
        self.production_history = []
        self.warehouse_release_history = []
        self.inventory_history = {}
        self.warehouse_history = {}
        self.pending_allocations = {}

        self.verbose = False
        self.warehouse_release_delay = 0
        self.allocation_batch_frequency = 1

        self._record_daily_history()
        if self.use_blockchain:
            self._initialize_blockchain_state()

    def _print(self, message):
        if self.verbose and self.console:
            self.console.print(message)

    def _initialize_blockchain_state(self):
        """Sets initial static data on the blockchain (e.g., drug criticalities)."""
        if not self.use_blockchain:
            return
        self.console.print("[cyan]Initializing blockchain state (drug criticalities)...[/]")
        setup_successful = True
        for drug in self.scenario.drugs:
            drug_id = drug['id']
            crit_val = drug.get('criticality_value', 1)
            try:
                tx_result = self.blockchain.set_drug_criticality(drug_id, crit_val)
                if not tx_result or tx_result.get('status') != 'success':
                    self._print(f"[red]Failed to set blockchain criticality for Drug {drug_id}[/]")
                    setup_successful = False
            except Exception as e:
                self._print(f"[red]Error setting blockchain criticality for Drug {drug_id}: {e}[/]")
                setup_successful = False
        if setup_successful:
            self.console.print("[green]✓ Blockchain state initialized.[/]")
        else:
            self.console.print("[yellow]Warning: Failed to initialize some blockchain state.[/]")

    def _record_daily_history(self):
        """Records inventory snapshots for the current day."""
        current_inv_snapshot = json.loads(json.dumps(self.inventories)) # Deep copy
        self.inventory_history[self.current_day] = current_inv_snapshot
        self.warehouse_history[self.current_day] = self.warehouse_inventories.copy()

    def reset(self):
        """Reset the environment to its initial state."""
        scenario = self.scenario # Keep the same scenario generator
        initial_manufacturer_inventory = 5000
        initial_distributor_inventory = 2000
        initial_hospital_inventory = 500
        initial_warehouse_inventory = 0
        self.__init__(
            scenario_generator=scenario,
            initial_manufacturer_inventory=initial_manufacturer_inventory,
            initial_distributor_inventory=initial_distributor_inventory,
            initial_hospital_inventory=initial_hospital_inventory,
            initial_warehouse_inventory=initial_warehouse_inventory,
            blockchain_interface=self.blockchain,
            use_blockchain=self.use_blockchain,
            console=self.console
        )
        self.verbose = False
        self.warehouse_release_delay = 0
        self.allocation_batch_frequency = 1
        return self.get_observations()

    def _process_production(self, production_actions: Dict):
        """Process drug production -> warehouse."""
        for drug_id, amount in production_actions.items():
            try:
                drug_id = int(drug_id)
                if not (0 <= drug_id < self.num_drugs): continue
                capacity = self.scenario.get_manufacturing_capacity(self.current_day, drug_id)
                actual_production = min(max(0.0, float(amount)), capacity)

                if actual_production > 0:
                    self.warehouse_inventories[drug_id] = self.warehouse_inventories.get(drug_id, 0.0) + actual_production
                    self.production_history.append({
                        "day": self.current_day, "drug_id": drug_id, "amount": actual_production,
                        "released": False, "release_day": None
                    })
                    if self.verbose:
                        drug_name = self.scenario.drugs[drug_id].get("name", f"Drug-{drug_id}")
                        self._print(f"[blue]Production: {actual_production:.1f} units of {drug_name} -> warehouse[/]")
            except (ValueError, TypeError, KeyError) as e:
                 self._print(f"[yellow]Error processing production for drug_id {drug_id}: {e}[/]")

    def _process_warehouse_release(self):
        """Release inventory warehouse -> manufacturer after delay."""
        release_delay = getattr(self, 'warehouse_release_delay', 0)
        if release_delay < 0: release_delay = 0

        eligible = [e for e in self.production_history if not e.get("released", False) and e["day"] <= self.current_day - release_delay]

        for entry in eligible:
            try:
                drug_id = entry["drug_id"]
                amount_produced = entry["amount"]
                available_in_warehouse = self.warehouse_inventories.get(drug_id, 0.0)
                amount_to_release = min(amount_produced, available_in_warehouse)

                if amount_to_release > 0:
                    self.warehouse_inventories[drug_id] -= amount_to_release
                    if drug_id not in self.inventories: self.inventories[drug_id] = {0: 0.0}
                    self.inventories[drug_id][0] = self.inventories[drug_id].get(0, 0.0) + amount_to_release

                    entry["released"] = True
                    entry["release_day"] = self.current_day
                    self.warehouse_release_history.append({
                        "day": self.current_day, "drug_id": drug_id, "amount": amount_to_release,
                        "production_day": entry["day"], "delay_days": self.current_day - entry["day"]
                    })
                    if self.verbose:
                        drug_name = self.scenario.drugs[drug_id].get("name", f"Drug-{drug_id}")
                        delay_taken = self.current_day - entry['day']
                        self._print(f"[cyan]Warehouse release: {amount_to_release:.1f} units of {drug_name} released after {delay_taken} days[/]")
            except (KeyError, ValueError, TypeError) as e:
                self._print(f"[yellow]Error processing warehouse release for entry {entry}: {e}[/]")

    def _process_batch_allocation(self, allocation_actions: Dict):
        """ Accumulates or processes manufacturer allocations based on frequency. """
        batch_frequency = getattr(self, 'allocation_batch_frequency', 1)
        is_batch_processing_day = self.current_day > 0 and self.current_day % batch_frequency == 0

        # Store current day's intended allocations
        for drug_id, region_allocs in allocation_actions.items():
            try:
                drug_id_str = str(int(drug_id))
                if drug_id_str not in self.pending_allocations: self.pending_allocations[drug_id_str] = {}
                for region_id, amount in region_allocs.items():
                    region_id_str = str(int(region_id))
                    current = self.pending_allocations[drug_id_str].get(region_id_str, 0.0)
                    self.pending_allocations[drug_id_str][region_id_str] = current + max(0.0, float(amount))
            except (ValueError, TypeError) as e:
                self._print(f"[yellow]Error processing/accumulating allocation action: {e}[/]")
                continue

        # Process accumulated batch if it's batch day or daily allocation
        if batch_frequency == 1 or is_batch_processing_day:
            if self.verbose and batch_frequency > 1 and self.pending_allocations:
                self._print(f"[green]Allocation batching: Processing accumulated batch from previous {batch_frequency} days on day {self.current_day}[/]")

            integer_allocations_requests = {}
            for drug_id_str, region_allocs in self.pending_allocations.items():
                try:
                    drug_id = int(drug_id_str)
                    integer_allocations_requests[drug_id] = {}
                    for region_id_str, amount in region_allocs.items():
                        try:
                            region_id = int(region_id_str)
                            if amount > 0.01:
                                integer_allocations_requests[drug_id][region_id] = amount
                        except ValueError: continue
                    if not integer_allocations_requests.get(drug_id):
                         if drug_id in integer_allocations_requests:
                              del integer_allocations_requests[drug_id]
                except ValueError: continue

            if integer_allocations_requests:
                 self._process_manufacturer_allocation_shipment(integer_allocations_requests)

            self.pending_allocations = {} # Clear pending batch
        elif self.verbose and batch_frequency > 1:
             next_batch_day = (self.current_day // batch_frequency + 1) * batch_frequency
             self._print(f"[dim]Allocation batching: Accumulating allocations, next processing day {next_batch_day}[/dim]")

    def _process_manufacturer_allocation_shipment(self, allocation_requests: Dict):
        """Process actual allocation shipment manufacturer -> distributors (creates pipeline entries)."""
        for drug_id, region_requests in allocation_requests.items():
            try:
                drug_id = int(drug_id)
                available_inventory = self.inventories.get(drug_id, {}).get(0, 0.0)
                if available_inventory <= 0: continue

                requesting_region_ids = list(region_requests.keys())
                requested_amounts = [region_requests[r_id] for r_id in requesting_region_ids]

                # --- Execute Allocation Logic (Blockchain or Local Fallback) ---
                final_allocations = self._calculate_fair_allocation(
                    drug_id,
                    dict(zip(requesting_region_ids, requested_amounts)),
                    available_inventory
                )
                # ----------------------------------------------------------------

                if not final_allocations:
                     self._print(f"[yellow]Allocation calculation failed for Drug {drug_id}, skipping shipment.[/]")
                     continue

                total_allocated_this_drug = 0
                for region_id, amount_to_allocate in final_allocations.items():
                    if amount_to_allocate <= 1e-6: continue
                    region_id = int(region_id)
                    distributor_id = region_id + 1

                    current_manu_inv = self.inventories.get(drug_id, {}).get(0, 0.0)
                    actual_shipment = min(amount_to_allocate, current_manu_inv)
                    if actual_shipment <= 1e-6: continue

                    base_lead_time = 1 + np.random.poisson(1)
                    transport_capacity_factor = self.scenario.get_transportation_capacity(self.current_day, region_id)
                    adjusted_lead_time = max(1, int(round(base_lead_time / max(0.1, transport_capacity_factor))))
                    arrival_day = self.current_day + adjusted_lead_time

                    # Ensure pipeline structure exists
                    self.pipelines.setdefault(0, {}).setdefault(distributor_id, {}).setdefault(drug_id, []).append((actual_shipment, arrival_day))

                    self.inventories[drug_id][0] -= actual_shipment
                    total_allocated_this_drug += actual_shipment

                    self.allocation_history.append({
                        "day": self.current_day, "drug_id": drug_id, "from_id": 0,
                        "to_id": distributor_id, "amount": actual_shipment, "arrival_day": arrival_day
                    })

                if self.verbose and total_allocated_this_drug > 0:
                    drug_name = self.scenario.drugs[drug_id].get("name", f"Drug-{drug_id}")
                    self._print(f"[green]Shipped {total_allocated_this_drug:.1f} units of {drug_name} from Manufacturer.[/]")

            except (KeyError, ValueError, TypeError) as e:
                self._print(f"[yellow]Error processing manufacturer shipment for drug_id {drug_id}: {e}[/]")
                continue

    def _process_distributor_orders(self, distributor_orders: Dict):
        """Record orders distributor -> manufacturer."""
        for region_id, drug_orders in distributor_orders.items():
            try:
                region_id = int(region_id); distributor_id = region_id + 1
                for drug_id, amount in drug_orders.items():
                    try:
                         drug_id = int(drug_id); amount = float(amount)
                         if amount > 0.01:
                             self.order_history.append({
                                 "day": self.current_day, "drug_id": drug_id, "from_id": distributor_id,
                                 "to_id": 0, "amount": amount
                             })
                    except (ValueError, TypeError): continue
            except (ValueError, TypeError): continue

    def _process_distributor_allocation(self, allocation_actions: Dict):
        """Process allocation distributor -> hospital (creates pipeline entries)."""
        for region_id, drug_allocations in allocation_actions.items():
             try:
                region_id = int(region_id); distributor_id = region_id + 1
                hospital_id = self.num_regions + 1 + region_id
                for drug_id, amount in drug_allocations.items():
                     try:
                        drug_id = int(drug_id); amount = float(amount)
                        if amount <= 1e-6: continue

                        available_inventory = self.inventories.get(drug_id, {}).get(distributor_id, 0.0)
                        actual_allocation = min(amount, available_inventory)
                        if actual_allocation <= 1e-6: continue

                        base_lead_time = 1
                        transport_capacity_factor = self.scenario.get_transportation_capacity(self.current_day, region_id)
                        adjusted_lead_time = max(1, int(round(base_lead_time / max(0.1, transport_capacity_factor))))
                        arrival_day = self.current_day + adjusted_lead_time

                        # Ensure pipeline structure exists
                        self.pipelines.setdefault(distributor_id, {}).setdefault(hospital_id, {}).setdefault(drug_id, []).append((actual_allocation, arrival_day))

                        self.inventories[drug_id][distributor_id] -= actual_allocation
                        self.allocation_history.append({
                            "day": self.current_day, "drug_id": drug_id, "from_id": distributor_id,
                            "to_id": hospital_id, "amount": actual_allocation, "arrival_day": arrival_day
                        })
                     except (ValueError, TypeError, KeyError) as e:
                         self._print(f"[yellow]Error in distributor allocation for region {region_id}, drug {drug_id}: {e}[/]")
                         continue
             except (ValueError, TypeError): continue

    def _process_hospital_orders(self, hospital_orders: Dict):
        """Record orders hospital -> distributor."""
        for region_id, drug_orders in hospital_orders.items():
            try:
                region_id = int(region_id); hospital_id = self.num_regions + 1 + region_id
                distributor_id = region_id + 1
                for drug_id, amount in drug_orders.items():
                    try:
                        drug_id = int(drug_id); amount = float(amount)
                        if amount > 0.01:
                            self.order_history.append({
                                "day": self.current_day, "drug_id": drug_id, "from_id": hospital_id,
                                "to_id": distributor_id, "amount": amount
                            })
                    except (ValueError, TypeError): continue
            except (ValueError, TypeError): continue

    def _process_deliveries(self):
        """Process arrivals from pipelines, updating inventories."""
        # Iterate safely while modifying pipeline
        for from_id in list(self.pipelines.keys()):
            if from_id not in self.pipelines: continue
            for to_id in list(self.pipelines[from_id].keys()):
                if to_id not in self.pipelines.get(from_id, {}): continue
                for drug_id in list(self.pipelines[from_id][to_id].keys()):
                    if drug_id not in self.pipelines.get(from_id, {}).get(to_id, {}): continue

                    current_pipeline = self.pipelines[from_id][to_id].get(drug_id, [])
                    arrived = [(amt, day) for amt, day in current_pipeline if day <= self.current_day]
                    remaining = [(amt, day) for amt, day in current_pipeline if day > self.current_day]

                    total_arrived_amount = sum(amt for amt, day in arrived)

                    if total_arrived_amount > 1e-6:
                        self.inventories.setdefault(drug_id, {}).setdefault(to_id, 0.0)
                        self.inventories[drug_id][to_id] += total_arrived_amount
                        if self.verbose:
                             drug_name = self.scenario.drugs[drug_id].get("name", f"Drug-{drug_id}")
                             self._print(f"[dim]Delivery: {total_arrived_amount:.1f} of {drug_name} arrived at Node {to_id} from Node {from_id}[/dim]")

                    # Update pipeline
                    if remaining:
                         self.pipelines[from_id][to_id][drug_id] = remaining
                    elif drug_id in self.pipelines.get(from_id, {}).get(to_id, {}): # Check before deleting
                         del self.pipelines[from_id][to_id][drug_id]

                # Clean up empty drug entries
                if from_id in self.pipelines and to_id in self.pipelines.get(from_id, {}) and not self.pipelines[from_id][to_id]:
                     del self.pipelines[from_id][to_id]
            # Clean up empty to_id entries
            if from_id in self.pipelines and not self.pipelines[from_id]:
                 del self.pipelines[from_id]

    def _process_patient_demand(self):
        """Process patient demand at hospitals, calculate metrics, and update BC cases."""
        regional_current_cases = {} # Store cases for blockchain update

        for region_id in range(self.num_regions):
            hospital_id = self.num_regions + 1 + region_id
            region_current_cases = 0 # Reset for region
            day_idx = min(self.current_day, len(self.scenario.epidemic_curves.get(region_id, [])) - 1)
            if day_idx >= 0 and region_id in self.scenario.epidemic_curves:
                region_current_cases = self.scenario.epidemic_curves[region_id][day_idx]
            regional_current_cases[region_id] = int(round(max(0, region_current_cases)))

            for drug_id in range(self.num_drugs):
                 try:
                    demand = self.scenario.get_daily_drug_demand(self.current_day, region_id, drug_id)
                    demand = max(0.0, float(demand))
                    if demand <= 1e-6: continue

                    available = self.inventories.get(drug_id, {}).get(hospital_id, 0.0)
                    self.total_demand[drug_id][region_id] += demand
                    self.demand_history.append({
                        "day": self.current_day, "drug_id": drug_id, "region_id": region_id,
                        "demand": demand, "available": available
                    })

                    fulfilled = min(demand, available)
                    unfulfilled = demand - fulfilled

                    if fulfilled > 0:
                         self.inventories[drug_id][hospital_id] -= fulfilled

                    if unfulfilled > 1e-6:
                        self.stockouts[drug_id][region_id] += 1
                        self.unfulfilled_demand[drug_id][region_id] += unfulfilled
                        self.stockout_history.append({
                            "day": self.current_day, "drug_id": drug_id, "region_id": region_id,
                            "demand": demand, "unfulfilled": unfulfilled
                        })
                        drug_criticality = self.scenario.drugs[drug_id].get("criticality_value", 1)
                        impact = unfulfilled * drug_criticality
                        self.patient_impact[region_id] += impact

                 except KeyError as e:
                      self._print(f"[red]Inventory key error processing demand: {e}. Hospital {hospital_id}, Drug {drug_id}.[/]")
                      continue
                 except Exception as e:
                      self._print(f"[red]Unexpected error processing demand for hospital {region_id}, drug {drug_id}: {e}[/]")
                      continue

        # --- Blockchain Call for Case Data Update ---
        if self.use_blockchain:
            for region_id, cases_int in regional_current_cases.items():
                try:
                    # Optional: Read current BC cases first to avoid redundant updates?
                    # current_bc_cases = self.blockchain.get_regional_case_count(region_id)
                    # if current_bc_cases is None or current_bc_cases != cases_int:
                    tx_result = self.blockchain.update_regional_case_count(region_id=int(region_id), cases=cases_int)
                    if tx_result is None or tx_result.get('status') != 'success':
                         self._print(f"[{Colors.BLOCKCHAIN}][yellow]BC Tx Failed (Case Data R{region_id}): {tx_result.get('error', 'Unknown BC error') if tx_result else 'Comm error'}[/]")
                    # else:
                    #     self._print(f"[dim]BC Case count for R{region_id} already up-to-date.[/dim]")
                except Exception as e:
                    self._print(f"[{Colors.BLOCKCHAIN}][yellow]BC Error calling update_case_data for R{region_id}: {e}[/]")

    def _calculate_rewards(self) -> Dict:
        """Calculate simple rewards (negative values indicating penalties)."""
        rewards = {
            "manufacturer": 0.0,
            "distributors": {r: 0.0 for r in range(self.num_regions)},
            "hospitals": {r: 0.0 for r in range(self.num_regions)}
        }
        total_unfulfilled_all = sum(sum(v.values()) for v in self.unfulfilled_demand.values())
        rewards["manufacturer"] -= 0.001 * total_unfulfilled_all
        for r in range(self.num_regions):
            region_unfulfilled = sum(self.unfulfilled_demand[d][r] for d in range(self.num_drugs))
            rewards["distributors"][r] -= 0.002 * region_unfulfilled
            rewards["hospitals"][r] -= 0.01 * self.patient_impact[r]
        return rewards

    def _calculate_fair_allocation(
            self,
            drug_id: int,
            requested_amounts_dict: Dict[int, float],
            available_inventory: float
        ) -> Optional[Dict[int, float]]:
        """Calculate fair allocation. Uses blockchain if enabled, otherwise local logic."""
        if available_inventory <= 1e-6:
            return {r_id: 0.0 for r_id in requested_amounts_dict}

        # --- Try Blockchain Strategy First (if enabled) ---
        if self.use_blockchain and self.blockchain:
            try:
                region_ids = list(requested_amounts_dict.keys())
                requested_amounts_list = list(requested_amounts_dict.values())
                if not region_ids or not requested_amounts_list:
                     self._print(f"[{Colors.BLOCKCHAIN}][yellow]Blockchain allocation for Drug {drug_id}: No valid requests provided.[/]")
                     return {r_id: 0.0 for r_id in requested_amounts_dict}

                self._print(f"[{Colors.BLOCKCHAIN}]Using blockchain allocation strategy for Drug-{drug_id}...[/]")
                bc_allocations_dict = self.blockchain.execute_fair_allocation(
                    drug_id=int(drug_id), region_ids=region_ids,
                    requested_amounts=requested_amounts_list,
                    available_inventory=available_inventory
                )

                if bc_allocations_dict is not None:
                    self._print(f"[{Colors.BLOCKCHAIN}]Blockchain allocation result for Drug-{drug_id}: { {k: f'{v:.1f}' for k, v in bc_allocations_dict.items()} }[/]")
                    final_alloc = {r_id: 0.0 for r_id in requested_amounts_dict}
                    total_allocated_bc = 0
                    for r_id_res, amount_res in bc_allocations_dict.items():
                         final_alloc[r_id_res] = max(0.0, amount_res)
                         total_allocated_bc += max(0.0, amount_res)

                    if total_allocated_bc > available_inventory * 1.01:
                        self._print(f"[{Colors.BLOCKCHAIN}][yellow]Warning: Blockchain allocation for Drug {drug_id} exceeded available ({total_allocated_bc:.1f} > {available_inventory:.1f}). Scaling down.[/]")
                        if total_allocated_bc > 0:
                            scale_down = available_inventory / total_allocated_bc
                            final_alloc = {r_id: amount * scale_down for r_id, amount in final_alloc.items()}
                        else: final_alloc = {r_id: 0.0 for r_id in final_alloc}
                    return final_alloc
                else:
                    self._print(f"[{Colors.BLOCKCHAIN}][yellow]Blockchain allocation strategy call failed for Drug-{drug_id}. Falling back to local logic.[/]")

            except Exception as e:
                self._print(f"[{Colors.BLOCKCHAIN}][yellow]Blockchain allocation strategy error for Drug-{drug_id}: {e}. Falling back to local logic.[/]")

        # --- Local Fair Allocation Fallback ---
        self._print(f"[{Colors.FALLBACK}]Using local allocation logic for Drug-{drug_id}...[/]")
        drug_info = self.scenario.drugs[drug_id]
        # --- Get cases from simulation curve ONLY FOR LOCAL FALLBACK ---
        # This ideally shouldn't be needed if blockchain is primary, but required for the tool
        current_day_idx = min(self.current_day, self.scenario_length - 1)
        region_cases_fallback = {}
        for r_id in requested_amounts_dict.keys():
            if r_id < len(self.scenario.epidemic_curves) and current_day_idx < len(self.scenario.epidemic_curves[r_id]):
                 region_cases_fallback[r_id] = self.scenario.epidemic_curves[r_id][current_day_idx]
            else: region_cases_fallback[r_id] = 0

        local_allocations = allocation_priority_tool(
            drug_info, requested_amounts_dict, region_cases_fallback, available_inventory
            )
        if self.verbose:
            self._print(f"[{Colors.FALLBACK}]Local allocation result for Drug-{drug_id}: { {k: f'{v:.1f}' for k, v in local_allocations.items()} }[/dim]")
        return local_allocations

    # --- Observation Methods ---

    def _get_manufacturer_observation(self) -> Dict:
        """Get observation for manufacturer agent, EXCLUDING current cases/trend."""
        current_day = min(self.current_day, self.scenario_length - 1)
        obs = {
            "day": self.current_day,
            "inventories": {str(drug_id): self.inventories.get(drug_id, {}).get(0, 0.0) for drug_id in range(self.num_drugs)},
            "warehouse_inventories": {str(drug_id): self.warehouse_inventories.get(drug_id, 0.0) for drug_id in range(self.num_drugs)},
            "production_capacity": {str(drug_id): self.scenario.get_manufacturing_capacity(current_day, drug_id) for drug_id in range(self.num_drugs)},
            "pipeline": {}, # Outgoing pipeline (Manu -> Dists)
            "recent_orders": [], # Incoming orders from distributors (Dist -> Manu)
            "drug_info": {str(i): d for i, d in enumerate(self.scenario.drugs)},
            # --- MODIFIED: Remove current cases/trend ---
            "epidemiological_data": {}, # Regional projected demand ONLY
            # --- End Modification ---
            "disruptions": [d for d in self.scenario.disruptions if isinstance(d, dict) and d.get("type") == "manufacturing" and d.get("drug_id") is not None and d.get("start_day", -1) <= current_day <= d.get("end_day", -1)],
            "pending_releases": [], # From warehouse -> Manu
            "pending_allocations": getattr(self, 'pending_allocations', {}),
            "batch_allocation_frequency": getattr(self, 'allocation_batch_frequency', 1),
            "is_batch_processing_day": self.current_day > 0 and self.current_day % max(1, getattr(self, 'allocation_batch_frequency', 1)) == 0,
            "days_to_next_batch_process": 0,
            "downstream_inventory_summary": {str(drug_id): {"total_distributor": 0.0, "total_hospital": 0.0, "total_downstream": 0.0} for drug_id in range(self.num_drugs)},
            "downstream_pipeline_summary": {str(drug_id): {"manu_to_dist": 0.0, "dist_to_hosp": 0.0, "total_inbound_downstream": 0.0} for drug_id in range(self.num_drugs)},
            "downstream_projected_demand_summary": {str(drug_id): 0.0 for drug_id in range(self.num_drugs)},
        }

        freq = obs["batch_allocation_frequency"]
        if freq > 0:
            current_cycle_day = self.current_day % freq
            days_remaining = freq - current_cycle_day
            obs["days_to_next_batch_process"] = days_remaining
            obs["next_batch_process_day"] = self.current_day + days_remaining
        else:
            obs["days_to_next_batch_process"] = 0
            obs["next_batch_process_day"] = self.current_day

        # Populate Downstream Summaries and Projected Demand (Keep this part)
        for drug_id in range(self.num_drugs):
            drug_id_str = str(drug_id)
            total_dist_inv = 0.0; total_hosp_inv = 0.0
            total_dist_pipeline = 0.0; total_hosp_pipeline = 0.0
            total_proj_demand = 0.0

            for region_id in range(self.num_regions):
                region_id_str = str(region_id)
                dist_id = region_id + 1
                hosp_id = self.num_regions + 1 + region_id

                # --- MODIFIED: Populate ONLY projected demand ---
                projected_demand = self.scenario.get_daily_drug_demand(current_day, region_id, drug_id)
                if region_id_str not in obs["epidemiological_data"]:
                    obs["epidemiological_data"][region_id_str] = {}
                obs["epidemiological_data"][region_id_str]["projected_demand"] = {drug_id_str: projected_demand} # Nested dict per region/drug
                # --- End Modification ---

                total_dist_inv += self.inventories.get(drug_id, {}).get(dist_id, 0.0)
                total_hosp_inv += self.inventories.get(drug_id, {}).get(hosp_id, 0.0)
                total_dist_pipeline += sum(amount for amount, _ in self.pipelines.get(0, {}).get(dist_id, {}).get(drug_id, []))
                total_hosp_pipeline += sum(amount for amount, _ in self.pipelines.get(dist_id, {}).get(hosp_id, {}).get(drug_id, []))
                total_proj_demand += projected_demand # Sum total projected demand

            # Store summarized info (Unchanged)
            obs["downstream_inventory_summary"][drug_id_str]["total_distributor"] = total_dist_inv
            obs["downstream_inventory_summary"][drug_id_str]["total_hospital"] = total_hosp_inv
            obs["downstream_inventory_summary"][drug_id_str]["total_downstream"] = total_dist_inv + total_hosp_inv
            obs["downstream_pipeline_summary"][drug_id_str]["manu_to_dist"] = total_dist_pipeline
            obs["downstream_pipeline_summary"][drug_id_str]["dist_to_hosp"] = total_hosp_pipeline
            obs["downstream_pipeline_summary"][drug_id_str]["total_inbound_downstream"] = total_dist_pipeline + total_hosp_pipeline
            obs["downstream_projected_demand_summary"][drug_id_str] = total_proj_demand

        # Populate Outgoing Pipeline (Unchanged)
        obs["pipeline"]["total_to_distributors"] = {}
        for drug_id in range(self.num_drugs):
             total_outgoing = sum(amount for r_id in range(self.num_regions) for amount, _ in self.pipelines.get(0, {}).get(r_id + 1, {}).get(drug_id, []))
             obs["pipeline"]["total_to_distributors"][str(drug_id)] = total_outgoing

        # Populate Recent Orders (Unchanged)
        obs["recent_orders"] = [o for o in self.order_history if o["to_id"] == 0 and o["day"] > self.current_day - 7]

        # Populate Pending Releases (Unchanged)
        release_delay = getattr(self, 'warehouse_release_delay', 0)
        obs["pending_releases"] = [{
            "drug_id": str(entry["drug_id"]), "amount": entry["amount"], "production_day": entry["day"],
            "days_in_warehouse": self.current_day - entry["day"],
            "expected_release_day": entry["day"] + release_delay
        } for entry in self.production_history if not entry.get("released", False)]

        return obs

    def _get_distributor_observation(self, region_id: int) -> Dict:
        """Get observation for distributor agent, EXCLUDING current cases/trend."""
        distributor_id = region_id + 1
        hospital_id = self.num_regions + 1 + region_id
        current_day = min(self.current_day, self.scenario_length - 1)

        obs = {
            "day": self.current_day,
            "region_id": region_id,
            "inventories": {str(drug_id): self.inventories.get(drug_id, {}).get(distributor_id, 0.0) for drug_id in range(self.num_drugs)},
            "inbound_pipeline": {str(drug_id): 0.0 for drug_id in range(self.num_drugs)}, # Manu -> This Dist
            "outbound_pipeline": {str(drug_id): 0.0 for drug_id in range(self.num_drugs)}, # This Dist -> Hosp
            "recent_orders": [], # Incoming orders from hospital
            "recent_allocations": [], # Incoming allocations from manufacturer
            "drug_info": {str(i): d for i, d in enumerate(self.scenario.drugs)},
            "region_info": self.scenario.regions[region_id],
            # --- MODIFIED: Remove current cases/trend ---
            "epidemiological_data": {}, # Projected hospital demand ONLY
            # --- End Modification ---
            "disruptions": [d for d in self.scenario.disruptions if isinstance(d, dict) and d.get("type") == "transportation" and d.get("region_id") == region_id and d.get("start_day", -1) <= current_day <= d.get("end_day", -1)]
        }

        # Populate Projected Demand for the hospital in this region
        projected_demand_this_region = {}
        for drug_id in range(self.num_drugs):
            projected_demand_this_region[str(drug_id)] = self.scenario.get_daily_drug_demand(current_day, region_id, drug_id)
        obs["epidemiological_data"]["projected_demand"] = projected_demand_this_region

        # Populate Inbound Pipeline (Unchanged)
        for drug_id in range(self.num_drugs):
             obs["inbound_pipeline"][str(drug_id)] = sum(amount for amount, _ in self.pipelines.get(0, {}).get(distributor_id, {}).get(drug_id, []))

        # Populate Outbound Pipeline (Unchanged)
        for drug_id in range(self.num_drugs):
             obs["outbound_pipeline"][str(drug_id)] = sum(amount for amount, _ in self.pipelines.get(distributor_id, {}).get(hospital_id, {}).get(drug_id, []))

        # Populate Recent Orders (Hosp -> This Dist) (Unchanged)
        obs["recent_orders"] = [o for o in self.order_history if o.get("to_id") == distributor_id and o.get("day", -1) > self.current_day - 7]

        # Populate Recent Allocations (Manu -> This Dist) (Unchanged)
        obs["recent_allocations"] = [a for a in self.allocation_history if a.get("to_id") == distributor_id and a.get("day", -1) > self.current_day - 7]

        return obs

    def _get_hospital_observation(self, region_id: int) -> Dict:
        """Get observation for hospital agent, EXCLUDING current cases/trend."""
        hospital_id = self.num_regions + 1 + region_id
        distributor_id = region_id + 1
        current_day = min(self.current_day, self.scenario_length - 1)

        obs = {
            "day": self.current_day,
            "region_id": region_id,
            "inventories": {str(drug_id): self.inventories.get(drug_id, {}).get(hospital_id, 0.0) for drug_id in range(self.num_drugs)},
            "inbound_pipeline": {str(drug_id): 0.0 for drug_id in range(self.num_drugs)}, # Dist -> This Hosp
            "recent_allocations": [], # Incoming allocations from distributor
            "demand_history": [], # Demand experienced by this hospital
            "stockout_history": [], # Stockouts experienced by this hospital
            "drug_info": {str(i): d for i, d in enumerate(self.scenario.drugs)},
            "region_info": self.scenario.regions[region_id],
            # --- MODIFIED: Remove current cases/trend ---
            "epidemiological_data": {}, # Projected hospital demand ONLY
            # --- End Modification ---
            "disruptions": [d for d in self.scenario.disruptions if isinstance(d, dict) and d.get("type") == "transportation" and d.get("region_id") == region_id and d.get("start_day", -1) <= current_day <= d.get("end_day", -1)]
        }

        # Populate Projected Demand for this hospital's region
        projected_demand_this_region = {}
        for drug_id in range(self.num_drugs):
            projected_demand_this_region[str(drug_id)] = self.scenario.get_daily_drug_demand(current_day, region_id, drug_id)
        obs["epidemiological_data"]["projected_demand"] = projected_demand_this_region

        # Populate Inbound Pipeline (Unchanged)
        for drug_id in range(self.num_drugs):
            obs["inbound_pipeline"][str(drug_id)] = sum(amount for amount, _ in self.pipelines.get(distributor_id, {}).get(hospital_id, {}).get(drug_id, []))

        # Populate Recent Allocations (Dist -> This Hosp) (Unchanged)
        obs["recent_allocations"] = [a for a in self.allocation_history if a.get("to_id") == hospital_id and a.get("day", -1) > self.current_day - 7]

        # Populate Demand History (This Hosp) (Unchanged)
        obs["demand_history"] = [d for d in self.demand_history if d.get("region_id") == region_id and d.get("day", -1) > self.current_day - 7]

        # Populate Stockout History (This Hosp) (Unchanged)
        obs["stockout_history"] = [s for s in self.stockout_history if s.get("region_id") == region_id and s.get("day", -1) > self.current_day - 7]

        return obs

    def get_observations(self) -> Dict:
        """Get observations for all agents."""
        manufacturer_obs = self._get_manufacturer_observation()
        distributor_obs = {r: self._get_distributor_observation(r) for r in range(self.num_regions)}
        hospital_obs = {r: self._get_hospital_observation(r) for r in range(self.num_regions)}

        return {
            "manufacturer": manufacturer_obs,
            "distributors": distributor_obs,
            "hospitals": hospital_obs
        }

    def step(self, actions: Dict):
        """Execute one day simulation step."""

        # 1. Production (to warehouse)
        self._process_production(actions.get("manufacturer_production", {}))

        # 2. Warehouse Release (to manufacturer inventory)
        self._process_warehouse_release()

        # 3. Record Orders (does not move inventory)
        self._process_distributor_orders(actions.get("distributor_orders", {}))
        self._process_hospital_orders(actions.get("hospital_orders", {}))

        # 4. Process Allocations (accumulates or triggers shipments)
        self._process_batch_allocation(actions.get("manufacturer_allocation", {}))
        self._process_distributor_allocation(actions.get("distributor_allocation", {}))

        # 5. Process Deliveries (moves inventory from pipelines to destinations)
        self._process_deliveries()

        # 6. Process Patient Demand (consumes hospital inventory, calculates metrics, updates BC cases)
        self._process_patient_demand()

        # 7. Increment Day and Record History
        self.current_day += 1
        self._record_daily_history()

        # 8. Calculate Rewards and Check Termination
        rewards = self._calculate_rewards()
        done = self.current_day >= self.scenario_length
        observations = self.get_observations() # Get observations for the START of the *next* day

        info = {
            "stockouts": {d: r.copy() for d, r in self.stockouts.items()},
            "unfulfilled_demand": {d: r.copy() for d, r in self.unfulfilled_demand.items()},
            "patient_impact": self.patient_impact.copy(),
            "current_day": self.current_day,
            "warehouse_inventory": self.warehouse_inventories.copy(),
            "manufacturer_inventory": {str(drug_id): self.inventories.get(drug_id, {}).get(0, 0.0) for drug_id in range(self.num_drugs)},
            "pending_allocations": getattr(self, 'pending_allocations', {})
        }

        return observations, rewards, done, info

# --- END OF FILE src/environment/supply_chain.py ---

# --- START OF FILE main.py ---

#!/usr/bin/env python3
"""
Pandemic Supply Chain Simulation - Main Entry Point
"""

import argparse
import time
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
import os # Added for path joining
from typing import Optional # For type hinting

# Updated import order to match config first
from config import (
    console, Colors, ensure_folder_exists, save_console_html, OPENAI_API_KEY,
    NODE_URL, CONTRACT_ADDRESS, CONTRACT_ABI_PATH, BLOCKCHAIN_PRIVATE_KEY,
    check_blockchain_config
)
from src.scenario.generator import PandemicScenarioGenerator
from src.scenario.visualizer import (
    visualize_epidemic_curves,
    visualize_drug_demand,
    visualize_disruptions,
    visualize_sir_components,
    visualize_sir_simulation
)
from src.environment.supply_chain import PandemicSupplyChainEnvironment
from src.environment.metrics import track_service_levels, visualize_service_levels, visualize_performance, visualize_inventory_levels
from src.tools import PandemicSupplyChainTools # Import the class
from src.llm.openai_integration import OpenAILLMIntegration
from src.agents.manufacturer import create_openai_manufacturer_agent
from src.agents.distributor import create_openai_distributor_agent
from src.agents.hospital import create_openai_hospital_agent
# Import BlockchainInterface conditionally or handle ImportError
try:
    from src.blockchain.interface import BlockchainInterface
except ImportError:
    console.print("[yellow]Warning: Blockchain interface module not found. Blockchain features will be disabled.[/]")
    BlockchainInterface = None

import datetime

def run_openai_pandemic_simulation(
    console: Console,
    openai_api_key: str,
    num_regions: int = 5,
    num_drugs: int = 3,
    simulation_days: int = 180,
    pandemic_severity: float = 0.8,
    disruption_probability: float = 0.1,
    warehouse_release_delay: int = 3,
    allocation_batch_frequency: int = 7,
    model_name: str = "gpt-3.5-turbo",
    visualize: bool = True,
    verbose: bool = True,
    use_colors: bool = True,
    output_folder: str = "output",
    blockchain_interface: Optional[BlockchainInterface] = None, # Allow None
    use_blockchain: bool = False
):
    """Run simulation with OpenAI-powered agents."""

    if not use_colors: console.no_color = True
    sim_type = "OpenAI" + (" + Blockchain" if use_blockchain else "")
    console.print(f"[bold]Initializing {sim_type}-powered pandemic supply chain simulation...[/]")

    scenario_generator = PandemicScenarioGenerator(
        console=console,
        num_regions=num_regions, num_drugs=num_drugs,
        scenario_length=simulation_days, pandemic_severity=pandemic_severity,
        disruption_probability=disruption_probability
    )
    environment = PandemicSupplyChainEnvironment(
        scenario_generator,
        blockchain_interface=blockchain_interface,
        use_blockchain=use_blockchain,
        console=console
    )
    environment.warehouse_release_delay = warehouse_release_delay
    environment.allocation_batch_frequency = allocation_batch_frequency
    environment.verbose = verbose

    # Create tools instance
    tools = PandemicSupplyChainTools()

    try:
        openai_integration = OpenAILLMIntegration(openai_api_key, model_name, console=console )
    except Exception as e:
        console.print(f"[bold red]Failed to initialize OpenAI Integration: {e}. Aborting simulation.[/]")
        return None

    # Create agents, passing the blockchain interface instance if enabled
    manufacturer = create_openai_manufacturer_agent(
        tools, openai_integration, num_regions=num_regions, # Pass num_regions
        verbose=verbose, console=console, blockchain_interface=blockchain_interface if use_blockchain else None
    )
    distributors = [create_openai_distributor_agent(
        r, tools, openai_integration, num_regions=num_regions,
        verbose=verbose, console=console, blockchain_interface=blockchain_interface if use_blockchain else None
        ) for r in range(num_regions)]
    hospitals = [create_openai_hospital_agent(
        r, tools, openai_integration,
        verbose=verbose, console=console, blockchain_interface=blockchain_interface if use_blockchain else None
        ) for r in range(num_regions)]

    observations = environment.reset()
    metrics_history = {"stockouts": [], "unfulfilled_demand": [], "patient_impact": []}

    console.print(f"[bold]Running simulation for {simulation_days} days using {model_name}...[/]")
    start_time = time.time()

    for day_index in range(simulation_days):
        current_sim_day = day_index + 1
        console.rule(f"[bold cyan] Starting Day {current_sim_day}/{simulation_days} [/bold cyan]", style="cyan")

        # --- Print Daily Epidemic State (if verbose) ---
        if verbose:
            epi_table = Table(title=f"Epidemic State - Day {current_sim_day}", show_header=True, header_style="bold magenta", box=box.SIMPLE)
            epi_table.add_column("Region", style="cyan")
            epi_table.add_column("Cases (Sim)", style="white", justify="right") # Internal Simulation Cases
            # epi_table.add_column("Trend (7d)", style="yellow", justify="right") # Trend info removed from obs
            epi_table.add_column("Proj.Demand(Sum)", style="magenta", justify="right") # Show Projected Demand Summary
            if use_blockchain and blockchain_interface:
                 epi_table.add_column("Cases (BC)", style=Colors.BLOCKCHAIN, justify="right")

            scenario = environment.scenario
            num_regions_in_scenario = len(scenario.regions)

            for r_id in range(num_regions_in_scenario):
                 region_name = scenario.regions[r_id].get("name", f"Region-{r_id+1}")
                 bc_cases_str = "[dim]N/A[/]"

                 # Simulation internal state
                 if r_id in scenario.epidemic_curves:
                      curve = scenario.epidemic_curves[r_id]
                      current_idx = min(day_index, len(curve) - 1)
                      if current_idx >= 0:
                          sim_cases = curve[current_idx]
                          sim_cases_str = f"{sim_cases:.0f}"
                          # Calculate projected demand for this region (sum across drugs for display)
                          proj_demand_region = sum(scenario.get_daily_drug_demand(current_day=day_index, region_id=r_id, drug_id=d_id) for d_id in range(environment.num_drugs))
                          proj_demand_str = f"{proj_demand_region:.0f}"
                      else:
                          sim_cases_str = "[dim]N/A[/]"; proj_demand_str = "[dim]N/A[/]"
                 else:
                       sim_cases_str = "[dim]No Data[/]"; proj_demand_str = "[dim]No Data[/]"

                 # Query blockchain if enabled
                 if use_blockchain and blockchain_interface:
                     try:
                          bc_cases = blockchain_interface.get_regional_case_count(r_id)
                          bc_cases_str = f"{bc_cases}" if bc_cases is not None else "[red]Error[/]"
                     except Exception:
                          bc_cases_str = "[red]QueryErr[/]"

                 # Add row to table
                 if use_blockchain and blockchain_interface:
                       epi_table.add_row(region_name, sim_cases_str, proj_demand_str, bc_cases_str)
                 else:
                       epi_table.add_row(region_name, sim_cases_str, proj_demand_str)

            console.print(epi_table)
            console.print()

        # --- Get Decisions ---
        all_actions = {}
        manu_decision = {}; dist_orders = {}; dist_allocs = {}; hosp_orders = {}

        # Manufacturer
        try:
            manu_obs = observations.get("manufacturer", {})
            if manu_obs: manu_decision = manufacturer.decide(manu_obs)
            else: console.print("[yellow]Warning: No manufacturer observation found.[/]")
            all_actions.update(manu_decision or {}) # Use empty dict if decision failed
            if verbose and manu_decision.get("manufacturer_production"):
                prod_str = {k: f"{v:.1f}" for k, v in manu_decision["manufacturer_production"].items()}
                console.print(f"[{Colors.MANUFACTURER}]Manu Production:[/]{prod_str}")
            if verbose and manu_decision.get("manufacturer_allocation"):
                alloc_str = {k: {k2: f"{v2:.1f}" for k2, v2 in v.items()} for k, v in manu_decision["manufacturer_allocation"].items()}
                console.print(f"[{Colors.MANUFACTURER}]Manu Allocation Request:[/]{alloc_str}")
        except Exception as e:
            console.print(f"[bold red]Error during Manufacturer decision on day {current_sim_day}: {e}[/]")
            console.print_exception(show_locals=False)
            all_actions["manufacturer_production"] = {}
            all_actions["manufacturer_allocation"] = {}

        # Distributors
        for dist_agent in distributors:
            try:
                dist_obs = observations.get("distributors", {}).get(dist_agent.agent_id)
                if dist_obs: dist_decision = dist_agent.decide(dist_obs)
                else: console.print(f"[yellow]Warning: No observation found for Distributor {dist_agent.agent_id}.[/]"); dist_decision = {}

                if dist_decision.get("distributor_orders"):
                    dist_orders.update(dist_decision["distributor_orders"])
                if dist_decision.get("distributor_allocation"):
                    dist_allocs.update(dist_decision["distributor_allocation"])
                if verbose and dist_decision.get("distributor_orders"):
                    order_str = {k: f"{v:.1f}" for k, v in dist_decision["distributor_orders"].get(dist_agent.agent_id, {}).items()}
                    if order_str: console.print(f"[{Colors.DISTRIBUTOR}]Dist {dist_agent.agent_id} Orders:[/]{order_str}")
                if verbose and dist_decision.get("distributor_allocation"):
                    alloc_str = {k: f"{v:.1f}" for k, v in dist_decision["distributor_allocation"].get(dist_agent.agent_id, {}).items()}
                    if alloc_str: console.print(f"[{Colors.DISTRIBUTOR}]Dist {dist_agent.agent_id} Allocation:[/]{alloc_str}")
            except Exception as e:
                console.print(f"[bold red]Error during Distributor {dist_agent.agent_id} decision on day {current_sim_day}: {e}[/]")
                console.print_exception(show_locals=False)
                dist_orders[dist_agent.agent_id] = {}
                dist_allocs[dist_agent.agent_id] = {}
        all_actions["distributor_orders"] = dist_orders
        all_actions["distributor_allocation"] = dist_allocs

        # Hospitals
        for hosp_agent in hospitals:
            try:
                hosp_obs = observations.get("hospitals", {}).get(hosp_agent.agent_id)
                if hosp_obs: hosp_decision = hosp_agent.decide(hosp_obs)
                else: console.print(f"[yellow]Warning: No observation found for Hospital {hosp_agent.agent_id}.[/]"); hosp_decision = {}

                if hosp_decision.get("hospital_orders"):
                    hosp_orders.update(hosp_decision["hospital_orders"])
                if verbose and hosp_decision.get("hospital_orders"):
                    order_str = {k: f"{v:.1f}" for k, v in hosp_decision["hospital_orders"].get(hosp_agent.agent_id, {}).items()}
                    if order_str: console.print(f"[{Colors.HOSPITAL}]Hosp {hosp_agent.agent_id} Orders:[/]{order_str}")
            except Exception as e:
                console.print(f"[bold red]Error during Hospital {hosp_agent.agent_id} decision on day {current_sim_day}: {e}[/]")
                console.print_exception(show_locals=False)
                hosp_orders[hosp_agent.agent_id] = {}
        all_actions["hospital_orders"] = hosp_orders

        # --- Step Environment ---
        try:
            observations, rewards, done, info = environment.step(all_actions)
        except Exception as e:
            console.print(f"[bold red]CRITICAL ERROR during environment step on day {current_sim_day}: {e}[/]")
            console.print_exception(show_locals=True)
            console.print("Aborting simulation.")
            break

        # --- Logging / Metrics ---
        if verbose:
            wh_inv = info.get('warehouse_inventory', {})
            manu_inv = info.get('manufacturer_inventory', {})
            pending_str = ", PendingAlloc: Yes" if info.get('pending_allocations') else ""
            console.print(f"Day {info['current_day']} End: WH Inv:{sum(wh_inv.values()):.0f}, Manu Inv:{sum(float(v) for v in manu_inv.values()):.0f}{pending_str}")
            day_stockouts = sum(s['unfulfilled'] for s in environment.stockout_history if s['day']==info['current_day']-1)
            if day_stockouts > 0:
                console.print(f"[yellow]Stockouts recorded for day {info['current_day']-1}: {day_stockouts:.1f} units unfulfilled.[/]")

        metrics_history["stockouts"].append(info.get("stockouts", {}))
        metrics_history["unfulfilled_demand"].append(info.get("unfulfilled_demand", {}))
        metrics_history["patient_impact"].append(info.get("patient_impact", {}))

        if done:
            break

    # End simulation loop
    end_time = time.time()
    console.rule(f"\n[bold]Simulation complete. Total time: {end_time - start_time:.2f} seconds.[/]")

    final_stockouts = environment.stockouts
    final_unfulfilled = environment.unfulfilled_demand
    final_impact = environment.patient_impact
    service_levels = track_service_levels(environment)

    results = {
        "total_stockouts": final_stockouts,
        "total_unfulfilled_demand": final_unfulfilled,
        "patient_impact": final_impact,
        "metrics_history": metrics_history,
        "service_levels": service_levels,
        "total_demand": environment.total_demand,
        "scenario_regions": scenario_generator.regions,
        "scenario_drugs": scenario_generator.drugs
    }

    if visualize:
        console.print("[bold]Generating visualizations...[/]")
        try:
            visualize_epidemic_curves(scenario_generator, output_folder, console=console)
            visualize_drug_demand(scenario_generator, output_folder, console=console)
            visualize_disruptions(scenario_generator, output_folder, console=console)
            visualize_sir_components(scenario_generator, output_folder, console=console)
            for region_id in range(min(3, num_regions)):
                visualize_sir_simulation(scenario_generator, region_id, output_folder, console=console)
            visualize_performance(environment, output_folder, console=console)
            visualize_inventory_levels(environment, output_folder, console=console)
            visualize_service_levels(environment, output_folder, console=console)
        except Exception as e:
            console.print(f"[red]Error during visualization: {e}[/]")
            console.print_exception(show_locals=True)

    return results

if __name__ == "__main__":
    openai_key = OPENAI_API_KEY
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(description="Run pandemic supply chain simulation with OpenAI agents")
    parser.add_argument("--regions", type=int, default=5, help="Number of regions")
    parser.add_argument("--drugs", type=int, default=3, help="Number of drugs")
    parser.add_argument("--days", type=int, default=30, help="Simulation days")
    parser.add_argument("--severity", type=float, default=0.8, help="Pandemic severity (0-1)")
    parser.add_argument("--disrupt-prob", type=float, default=0.1, help="Base disruption probability factor")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="OpenAI model name")
    parser.add_argument("--no-viz", action="store_false", dest="visualize", help="Disable visualizations")
    parser.add_argument("--quiet", action="store_false", dest="verbose", help="Less verbose output")
    parser.add_argument("--no-colors", action="store_false", dest="use_colors", help="Disable colored output")
    parser.add_argument("--folder", type=str, default="output", help="Folder for simulation output")
    parser.add_argument("--warehouse-delay", type=int, default=3, help="Warehouse release delay (days)")
    parser.add_argument("--allocation-batch", type=int, default=7, help="Allocation batch frequency (days, 1=daily)")
    parser.add_argument("--use-blockchain", action="store_true", default=False, help="Enable blockchain integration")

    args = parser.parse_args()

    if not args.use_colors: console.no_color = True

    output_folder_path = f"{args.folder}_{timestamp}_regions{args.regions}_drug{args.drugs}_days{args.days}"
    if args.use_blockchain: output_folder_path += "_blockchain"
    console.print(Panel("[bold white]🦠 PANDEMIC SUPPLY CHAIN SIMULATION (using OpenAI) 🦠[/]", border_style="blue", expand=False, padding=(1,2)))

    config_table = Table(title="Simulation Configuration", show_header=True, header_style="bold cyan", box=box.ROUNDED)
    config_table.add_column("Parameter", style="cyan"); config_table.add_column("Value", style="white")
    config_table.add_row("Regions", str(args.regions)); config_table.add_row("Drugs", str(args.drugs))
    config_table.add_row("Simulation Days", str(args.days)); config_table.add_row("Pandemic Severity", f"{args.severity:.2f}")
    config_table.add_row("Disruption Probability Factor", f"{args.disrupt_prob:.2f}")
    config_table.add_row("Warehouse Delay", f"{args.warehouse_delay} days")
    config_table.add_row("Allocation Batch Frequency", f"{args.allocation_batch} days" if args.allocation_batch > 1 else "Daily")
    config_table.add_row("LLM Model", args.model)
    config_table.add_row("Visualizations", "Enabled" if args.visualize else "Disabled")
    config_table.add_row("Verbose Output", "Enabled" if args.verbose else "Disabled")
    config_table.add_row("Output Folder", output_folder_path)
    config_table.add_row("Blockchain", "[bold green]Enabled[/]" if args.use_blockchain else "Disabled")
    if args.use_blockchain:
         config_table.add_row("  Node URL", NODE_URL)
         config_table.add_row("  Contract Address", CONTRACT_ADDRESS or "[red]Not Set[/]")
         config_table.add_row("  ABI Path", CONTRACT_ABI_PATH)
         config_table.add_row("  Signer Key Loaded", "[green]Yes[/]" if BLOCKCHAIN_PRIVATE_KEY else "[red]No[/]")
    console.print(config_table); console.print()

    output_folder = ensure_folder_exists(console, output_folder_path)

    # --- INITIALIZE BLOCKCHAIN INTERFACE ---
    blockchain_interface_instance = None
    actual_use_blockchain_flag = False

    if args.use_blockchain:
        console.print("\n[bold cyan]Attempting Blockchain Integration...[/]")
        if BlockchainInterface is None:
             console.print("[bold red]❌ Blockchain support not available (missing dependencies?). Halting.[/]")
             exit(1)
        if not check_blockchain_config():
             console.print("[bold red]❌ Blockchain configuration incomplete in .env or ABI file missing. Halting.[/]")
             console.print("[bold red]   Please ensure NODE_URL, CONTRACT_ADDRESS, BLOCKCHAIN_PRIVATE_KEY are set and ABI exists.[/]")
             exit(1)
        try:
            blockchain_interface_instance = BlockchainInterface(
                node_url=NODE_URL, contract_address=CONTRACT_ADDRESS,
                contract_abi_path=CONTRACT_ABI_PATH, private_key=BLOCKCHAIN_PRIVATE_KEY
            )
            actual_use_blockchain_flag = True
            console.print(f"[bold green]✓ Connected to Ethereum node and loaded contract.[/]")
        except Exception as e:
            console.print(f"[bold red]❌ FATAL ERROR: Could not initialize Blockchain Interface: {e}[/]")
            console.print("[bold red]   Check node connection, contract address, ABI path, and private key format.[/]")
            console.print("[bold red]   Halting simulation execution.[/]")
            try: save_console_html(console, output_folder=output_folder, filename="simulation_error_report.html")
            except Exception as save_e: console.print(f"[red]Could not save error report: {save_e}[/]")
            exit(1)
    else:
        actual_use_blockchain_flag = False
        console.print("\n[yellow]Blockchain integration disabled by command-line argument.[/]")
        console.print("[yellow]Running in simulation-only mode.[/]")
    console.print("-" * 30)

    # --- Run Simulation ---
    results = run_openai_pandemic_simulation(
        console=console, openai_api_key=openai_key,
        num_regions=args.regions, num_drugs=args.drugs, simulation_days=args.days,
        pandemic_severity=args.severity, disruption_probability=args.disrupt_prob,
        warehouse_release_delay=args.warehouse_delay, allocation_batch_frequency=args.allocation_batch,
        model_name=args.model, visualize=args.visualize, verbose=args.verbose, use_colors=args.use_colors,
        output_folder=output_folder,
        blockchain_interface=blockchain_interface_instance, # Pass instance
        use_blockchain=actual_use_blockchain_flag # Pass flag
    )

    # --- Display Results ---
    if results:
        console.print(Panel("[bold white]Simulation Results Summary[/]", border_style="green", expand=False))
        drug_names = {d['id']: d['name'] for d in results.get('scenario_drugs', [])}
        region_names = {r['id']: r['name'] for r in results.get('scenario_regions', [])}

        # Stockouts Summary
        console.print("\n[bold red]Total Stockout Days by Drug and Region:[/]")
        stockout_table_summary = Table(show_header=True, header_style="bold", box=box.SIMPLE)
        stockout_table_summary.add_column("Drug", style="cyan", min_width=10)
        stockout_table_summary.add_column("Region", style="magenta", min_width=10)
        stockout_table_summary.add_column("Stockout Days", style="white", justify="right", min_width=15)
        total_stockout_days = 0
        for drug_id, regions in results["total_stockouts"].items():
             drug_name = drug_names.get(drug_id, f"Drug {drug_id}")
             for region_id, count in regions.items():
                  region_name = region_names.get(region_id, f"Region {region_id}")
                  if count > 0:
                       color = "red" if count > (args.days * 0.3) else "yellow"
                       stockout_table_summary.add_row(drug_name, region_name, f"[{color}]{count}[/]")
                       total_stockout_days += count
        if total_stockout_days == 0: console.print("[green]✓ No stockout days recorded.[/]")
        else:
             console.print(stockout_table_summary)
             stockout_severity_threshold = (args.days * args.regions * args.drugs) * 0.1
             color = "red" if total_stockout_days > stockout_severity_threshold * 2 else "yellow" if total_stockout_days > 0 else "green"
             console.print(f"  Total stockout days across system: [bold {color}]{total_stockout_days}[/]")

        # Unfulfilled Demand Summary
        total_unfulfilled = sum(sum(drug.values()) for drug in results["total_unfulfilled_demand"].values())
        total_demand_all = sum(sum(drug.values()) for drug in results.get("total_demand", {}).values())
        percent_unfulfilled_str = f" ({ (total_unfulfilled / total_demand_all * 100) if total_demand_all > 0 else 0 :.1f}%)" if total_demand_all > 0 else ""
        color = "red" if total_unfulfilled > 10000 else "yellow" if total_unfulfilled > 0 else "green"
        console.print(f"\n[bold]Total Unfulfilled Demand (units): [{color}]{total_unfulfilled:.1f}[/{color}]{percent_unfulfilled_str}")

        # Patient Impact Summary
        console.print("\n[bold red]Patient Impact Score by Region:[/]")
        impact_table_summary = Table(show_header=True, header_style="bold", box=box.SIMPLE)
        impact_table_summary.add_column("Region", style="magenta", min_width=10)
        impact_table_summary.add_column("Impact Score", style="white", justify="right", min_width=15)
        total_impact = sum(results["patient_impact"].values())
        for region_id, impact in results["patient_impact"].items():
             region_name = region_names.get(region_id, f"Region {region_id}")
             impact_color = "red" if impact > 10000 else "yellow" if impact > 100 else "green"
             impact_table_summary.add_row(region_name, f"[{impact_color}]{impact:.1f}[/]")
        console.print(impact_table_summary)
        color = "red" if total_impact > 50000 else "yellow" if total_impact > 500 else "green"
        console.print(f"  Total Patient Impact Score: [bold {color}]{total_impact:.1f}[/]")

        # Service Level Summary
        console.print("\n[bold cyan]Service Level Performance (% Demand Met):[/]")
        service_table_summary = Table(show_header=True, header_style="bold", box=box.SIMPLE)
        service_table_summary.add_column("Metric", style="cyan", min_width=15)
        service_table_summary.add_column("Service Level", style="white", justify="right", min_width=15)
        service_levels = results["service_levels"]
        if service_levels:
             avg_service = np.mean([item["service_level"] for item in service_levels]) if service_levels else 0
             min_service = min(item["service_level"] for item in service_levels) if service_levels else 0
             final_service = service_levels[-1]["service_level"] if service_levels else 0
             def get_service_color(level): return "green" if level >= 95 else "cyan" if level >= 90 else "yellow" if level >= 80 else "red"
             service_table_summary.add_row("Average", f"[{get_service_color(avg_service)}]{avg_service:.1f}%[/]")
             service_table_summary.add_row("Minimum", f"[{get_service_color(min_service)}]{min_service:.1f}%[/]")
             service_table_summary.add_row("Final Day", f"[{get_service_color(final_service)}]{final_service:.1f}%[/]")
             console.print(service_table_summary)
        else: console.print("[yellow]No service level data calculated.[/]")

        # Overall Performance Rating
        rating = "N/A"; rating_color="white"
        if service_levels:
             avg_service = np.mean([item["service_level"] for item in service_levels])
             if avg_service >= 95 and total_impact < (50 * args.days): rating, rating_color = "Excellent", "green"
             elif avg_service >= 90 and total_impact < (200 * args.days): rating, rating_color = "Good", "cyan"
             elif avg_service >= 80 and total_impact < (1000 * args.days): rating, rating_color = "Fair", "yellow"
             else: rating, rating_color = "Poor", "red"
        console.print(f"\n[bold]Overall Supply Chain Performance:[/] [{rating_color}]{rating}[/]")

    else:
        console.print("[bold red]Simulation did not complete successfully. No results to display.[/]")

    # --- PRINT FINAL BLOCKCHAIN STATE ---
    if actual_use_blockchain_flag and blockchain_interface_instance:
        console.print("\n[bold cyan]Querying Final Blockchain State...[/]")
        try:
            blockchain_interface_instance.print_contract_state(num_regions=args.regions, num_drugs=args.drugs)
        except Exception as e:
            console.print(f"[red]Error querying final blockchain state: {e}[/]")

    html_filename = "simulation_report_openai" + ("_blockchain" if actual_use_blockchain_flag else "") + ".html"
    save_console_html(console, output_folder=output_folder, filename=html_filename)
    console.print(f"\n[green]Visualizations and report saved to folder: '{output_folder}'[/]")

# --- END OF FILE main.py ---

# --- START OF FILE package.json ---
# No changes needed.
# Content omitted for brevity...
# --- END OF FILE package.json ---

# --- START OF FILE SupplyChainData.sol ---
# No changes needed. Contract already stores cases.
# Content omitted for brevity...
# --- END OF FILE SupplyChainData.sol ---

# --- START OF FILE deploy.js ---
# No changes needed.
# Content omitted for brevity...
# --- END OF FILE deploy.js ---

# --- START OF FILE src/llm/openai_integration.py ---
# No changes needed.
# Content omitted for brevity...
# --- END OF FILE src/llm/openai_integration.py ---

# --- START OF FILE src/scenario/generator.py ---
# No changes needed.
# Content omitted for brevity...
# --- END OF FILE src/scenario/generator.py ---

# --- START OF FILE src/scenario/visualizer.py ---
# No changes needed.
# Content omitted for brevity...
# --- END OF FILE src/scenario/visualizer.py ---

# --- START OF FILE src/environment/metrics.py ---
# No changes needed.
# Content omitted for brevity...
# --- END OF FILE src/environment/metrics.py ---

# --- START OF FILE src/tools/allocation.py ---
# No changes needed.
# Content omitted for brevity...
# --- END OF FILE src/tools/allocation.py ---

# --- START OF FILE src/tools/assessment.py ---
# No changes needed.
# Content omitted for brevity...
# --- END OF FILE src/tools/assessment.py ---

# --- START OF FILE src/tools/forecasting.py ---
# No changes needed.
# Content omitted for brevity...
# --- END OF FILE src/tools/forecasting.py ---

# --- START OF FILE Empty files like __init__.py ---
# No changes needed.
```

**Key Changes Summary:**

1.  **`src/tools/__init__.py`**: Added the new `get_blockchain_regional_cases_tool` static method to the `PandemicSupplyChainTools` class and exported it.
2.  **`src/agents/base.py`**:
    *   Added `blockchain_interface` to `__init__` and stored it as `self.blockchain`.
    *   Added `_run_blockchain_regional_cases_tool` method to call the static tool, passing `self.blockchain` and `self.num_regions`.
    *   Modified `_clean_observation_for_prompt` to *remove* `current_cases` and `case_trend` from the `epidemiological_data` section before stringifying for the LLM prompt.
    *   Adjusted `_create_decision_prompt` to reflect that `current_cases` are no longer directly in the observation JSON and should be considered external (blockchain) data.
    *   Adjusted `_run_epidemic_forecast_tool` to rely on `projected_demand` from the observation as its primary input, since direct `current_cases` are removed from the observation JSON sent to the LLM/used by base tools.
3.  **`src/agents/manufacturer.py`**:
    *   Added `num_regions` and `blockchain_interface` to `__init__` and passed them to the `super().__init__`. Stored `num_regions`.
    *   In `decide`, call `_run_blockchain_regional_cases_tool` to get `blockchain_cases`. Handle potential `None` return.
    *   Passed `blockchain_cases` to `_make_production_decisions` and `_make_allocation_decisions`.
    *   In `_make_production_decisions`: Modified the "Forecasting-based scaling" rule to use `blockchain_cases` or `downstream_projected_demand_summary` instead of observation's `epidemiological_data`.
    *   In `_make_allocation_decisions`:
        *   Modified the *fallback logic* to use `blockchain_cases` for prioritization and potentially distributing `total_proj_demand`.
        *   Modified the *proactive allocation rule* to use `blockchain_cases` for trend checking and distribution prioritization.
        *   Calls to `_run_allocation_priority_tool` (the local fallback version) now pass `blockchain_cases`.
    *   Updated `create_openai_manufacturer_agent` factory function to accept and pass `num_regions` and `blockchain_interface`.
4.  **`src/agents/distributor.py` & `src/agents/hospital.py`**:
    *   Added `blockchain_interface` to `__init__` and passed it to `super().__init__`.
    *   Updated their respective `create_openai_..._agent` factory functions to accept and pass `blockchain_interface`. (They don't actively use the blockchain query tool in this implementation, but the plumbing is there).
5.  **`src/environment/supply_chain.py`**:
    *   Modified `_get_manufacturer_observation`, `_get_distributor_observation`, `_get_hospital_observation` to *exclude* `current_cases` and `case_trend` from the `epidemiological_data` dict within the returned observation. They now only contain `projected_demand`.
    *   The `_process_patient_demand` function *still* calculates the internal simulation cases and *writes* them to the blockchain if enabled.
    *   The `_calculate_fair_allocation` function *still* uses the blockchain function (`execute_fair_allocation`) if enabled, which *reads* the cases from the blockchain internally. The local fallback uses simulation data as before (this could be changed to also use the blockchain query tool if desired, but kept separate for now).
6.  **`main.py`**:
    *   When creating agents, pass the `blockchain_interface_instance` (if `actual_use_blockchain_flag` is true).
    *   Pass `num_regions=args.regions` to `create_openai_manufacturer_agent`.
    *   Adjusted the daily status table printout slightly as `case_trend` is removed from observation (though the underlying data still exists in the scenario).

This setup ensures that the agents' prompts and Python-based rules/fallbacks operate without direct access to the simulation's internal case counts in their observation dictionary. Instead, the Manufacturer agent explicitly queries the blockchain via a tool to get this trusted data, fulfilling the requirements of Strategy 1.