# --- START OF FILE src/blockchain/interface.py ---

"""
Blockchain interface using Web3.py to interact with the SupplyChainData smart contract.
"""

import json
import time
import math
from web3 import Web3
from web3.exceptions import ContractLogicError # For handling reverts
from web3.middleware import geth_poa_middleware # For PoA networks like Sepolia, Goerli, maybe Ganache/Hardhat sometimes
from typing import Dict, List, Optional, Any
from rich.console import Console

# Use a shared console or create one if needed (import from config if available, otherwise create)
try:
    from config import console, Colors # Assuming console and Colors are defined in config
except ImportError:
    console = Console()
    # Define basic colors if Colors class not available
    class Colors:
        BLOCKCHAIN = "bright_black"
        YELLOW = "yellow"
        RED = "red"
        GREEN = "green"
        DIM = "dim"


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
        # Increased gas limit slightly as a buffer for contract logic
        self.gas_limit = 3500000
        # Scaling factor for converting float amounts to integers for the contract
        self.SCALE_FACTOR = 1000 # Represents 3 decimal places

        try:
            self.w3 = Web3(Web3.HTTPProvider(node_url))

            # Check connection first
            if not self.w3.is_connected():
                 raise ConnectionError(f"Failed to connect to Ethereum node at {node_url}")

            # Inject PoA middleware - necessary for some testnets and potentially local nodes
            # Wraps the provider, does not raise error if already wrapped or not needed
            self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)

            console.print(f"[{Colors.GREEN}]Connected to Ethereum node: {node_url} (Chain ID: {self.w3.eth.chain_id})[/]")

            # Load Contract ABI
            with open(contract_abi_path, 'r') as f:
                contract_abi = json.load(f)

            # Load Contract
            checksum_address = self.w3.to_checksum_address(contract_address)
            self.contract = self.w3.eth.contract(address=checksum_address, abi=contract_abi)
            console.print(f"[{Colors.GREEN}]SupplyChainData contract loaded at address: {contract_address}[/]")

            # Set up account for transactions if private key is provided
            if private_key:
                # Ensure private key has '0x' prefix
                if not private_key.startswith('0x'):
                    private_key = '0x' + private_key
                self.account = self.w3.eth.account.from_key(private_key)
                self.w3.eth.default_account = self.account.address # Set default account for calls if needed
                console.print(f"[{Colors.GREEN}]Transaction account set up: {self.account.address}[/]")
            else:
                console.print(f"[{Colors.YELLOW}]Warning: No private key provided. Only read operations possible.[/]")

            # Test contract connection by reading owner (optional but good check)
            try:
                 owner = self.contract.functions.owner().call()
                 console.print(f"[{Colors.DIM}]Contract owner found: {owner}[/]")
            except Exception as e:
                 # This might happen if ABI is wrong, contract not deployed, or network issue
                 console.print(f"[{Colors.YELLOW}]Warning: Could not call contract 'owner' function. Contract may not be deployed correctly or ABI mismatch? Error: {e}[/]")


        except FileNotFoundError:
            console.print(f"[bold {Colors.RED}]Error: Contract ABI file not found at {contract_abi_path}[/]")
            raise
        except ConnectionError as e:
            console.print(f"[bold {Colors.RED}]Error connecting to Ethereum node: {e}[/]")
            raise
        except Exception as e:
            # Catch other potential errors during init (e.g., invalid address format)
            console.print(f"[bold {Colors.RED}]Error initializing BlockchainInterface: {e}[/]")
            console.print_exception(show_locals=True) # Show traceback for debugging
            raise

    def _get_gas_price(self):
        """Gets gas price based on network conditions."""
        # For local nodes (Hardhat/Ganache), gas price is often negligible or fixed
        if self.w3.eth.chain_id in [1337, 31337]: # Common local chain IDs
            return self.w3.to_wei('10', 'gwei') # A reasonable default for local testing
        try:
             # Use eth_gasPrice for simplicity on testnets/mainnet
             # Add retry logic for robustness
             for attempt in range(3):
                 try:
                     return self.w3.eth.gas_price
                 except Exception as e:
                     if attempt < 2:
                         console.print(f"[{Colors.YELLOW}]Retrying gas price fetch after error: {e}[/]")
                         time.sleep(0.5 * (attempt + 1)) # Short backoff
                     else:
                         raise e # Raise error after final attempt
             return self.w3.to_wei('20', 'gwei') # Fallback if retries fail
        except Exception as e:
            console.print(f"[{Colors.YELLOW}]Warning: Could not fetch gas price, using default 20 gwei. Error: {e}[/]")
            return self.w3.to_wei('20', 'gwei')

    def _send_transaction(self, function_call) -> Optional[Dict[str, Any]]:
        """Builds, signs, sends a transaction and waits for the receipt."""
        if not self.account:
            console.print(f"[bold {Colors.RED}]Error: Cannot send transaction. No private key configured.[/]")
            return None
        try:
            # Use the configured account address
            sender_address = self.account.address
            nonce = self.w3.eth.get_transaction_count(sender_address)
            gas_price = self._get_gas_price()

            tx_params = {
                'from': sender_address,
                'nonce': nonce,
                'gas': self.gas_limit,
                'gasPrice': gas_price,
                # 'chainId': self.w3.eth.chain_id # Optional: explicitly set chain ID
            }

            # Estimate gas (optional, can help catch reverts early, but adds latency)
            # try:
            #     estimated_gas = function_call.estimate_gas(tx_params)
            #     tx_params['gas'] = int(estimated_gas * 1.2) # Add buffer
            # except ContractLogicError as gas_error:
            #     console.print(f"[bold {Colors.RED}]Gas estimation failed (potential revert): {gas_error}[/]")
            #     return {'status': 'failed', 'error': f'Gas estimation failed: {gas_error}'}
            # except Exception as estimate_e:
            #      console.print(f"[yellow]Warning: Gas estimation call failed: {estimate_e}. Using default gas limit.[/]")


            # Build transaction
            transaction = function_call.build_transaction(tx_params)

            # Sign transaction
            signed_tx = self.w3.eth.account.sign_transaction(transaction, self.private_key)

            # Send transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            console.print(f"[{Colors.DIM}]Transaction sent: {tx_hash.hex()}. Waiting for receipt...[/]", style=Colors.BLOCKCHAIN)

            # Wait for receipt with timeout
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180) # 3 min timeout

            if tx_receipt['status'] == 1:
                console.print(f"[{Colors.GREEN}]✓ Transaction successful! Block: {tx_receipt['blockNumber']}, Gas Used: {tx_receipt['gasUsed']}[/]", style=Colors.BLOCKCHAIN)
                return {'status': 'success', 'receipt': tx_receipt}
            else:
                console.print(f"[bold {Colors.RED}]❌ Transaction failed! Receipt: {tx_receipt}[/]", style=Colors.BLOCKCHAIN)
                # You might want to try decoding the revert reason here if possible/needed
                return {'status': 'failed', 'receipt': tx_receipt, 'error': 'Transaction reverted'}

        except ValueError as ve: # Catch specific web3 errors like insufficient funds
             console.print(f"[bold {Colors.RED}]Transaction ValueError: {ve}[/]")
             # console.print_exception(show_locals=False)
             return {'status': 'error', 'error': str(ve)}
        except Exception as e: # Catch broader errors
            console.print(f"[bold {Colors.RED}]Error sending transaction: {type(e).__name__} - {e}[/]")
            # console.print_exception(show_locals=False) # Optional: Add more detailed traceback
            return {'status': 'error', 'error': str(e)}


    # --- Write Methods ---

    def update_regional_case_count(self, region_id: int, cases: int) -> Optional[Dict[str, Any]]:
        """Updates the case count for a region via a transaction."""
        if not self.account:
             console.print(f"[{Colors.YELLOW}]Skipping blockchain case update for R{region_id}: No private key.[/]")
             return None
        try:
            console.print(f"[{Colors.DIM}]Preparing blockchain tx: updateRegionalCaseCount(regionId={region_id}, cases={cases})[/]", style=Colors.BLOCKCHAIN)
            function_call = self.contract.functions.updateRegionalCaseCount(region_id, cases)
            return self._send_transaction(function_call)
        except Exception as e:
            console.print(f"[{Colors.RED}]Error preparing updateRegionalCaseCount transaction: {e}[/]")
            return {'status': 'error', 'error': str(e)}

    def set_drug_criticality(self, drug_id: int, criticality_value: int) -> Optional[Dict[str, Any]]:
        """Sets the drug criticality value via a transaction (likely used during setup)."""
        if not self.account:
             console.print(f"[{Colors.YELLOW}]Skipping blockchain criticality set for D{drug_id}: No private key.[/]")
             return None
        try:
            console.print(f"[{Colors.DIM}]Preparing blockchain tx: setDrugCriticality(drugId={drug_id}, criticalityValue={criticality_value})[/]", style=Colors.BLOCKCHAIN)
            function_call = self.contract.functions.setDrugCriticality(drug_id, criticality_value)
            return self._send_transaction(function_call)
        except Exception as e:
            console.print(f"[{Colors.RED}]Error preparing setDrugCriticality transaction: {e}[/]")
            return {'status': 'error', 'error': str(e)}

    def execute_fair_allocation(self, drug_id: int, region_ids: List[int], requested_amounts: List[float], available_inventory: float) -> Optional[Dict[int, float]]:
        """
        Triggers the fair allocation logic on the smart contract.
        Uses call() to get simulated result for simulation, then sends the transaction.

        Args:
            drug_id: ID of the drug.
            region_ids: List of requesting region IDs.
            requested_amounts: List of corresponding requested amounts (float).
            available_inventory: Total available inventory (float).

        Returns:
            A dictionary {region_id: allocated_amount (float)} based on the simulated call(), or None if simulation fails.
        """
        try:
            # Convert float amounts to integers for the contract
            requested_amounts_int = [int(round(r * self.SCALE_FACTOR)) for r in requested_amounts]
            available_inventory_int = int(round(available_inventory * self.SCALE_FACTOR))

            # Handle edge case: If available inventory is zero or negative after scaling, skip blockchain call
            if available_inventory_int <= 0:
                 console.print(f"[{Colors.YELLOW}]execute_fair_allocation (Drug {drug_id}): Scaled available inventory is zero or less ({available_inventory_int}), skipping blockchain call.[/]")
                 return {r_id: 0.0 for r_id in region_ids} # Return zero allocations

            function_call = self.contract.functions.executeFairAllocation(
                drug_id, region_ids, requested_amounts_int, available_inventory_int
            )

            # --- Simulate the call() first to get the return value for the simulation ---
            allocated_amounts_int = None
            try:
                console.print(f"[{Colors.DIM}]Simulating executeFairAllocation call for Drug {drug_id}...[/]", style=Colors.BLOCKCHAIN)
                # Determine address to use for simulation call (sender might matter for some contract logic)
                simulated_from_address = self.account.address if self.account else self.w3.eth.accounts[0] if self.w3.eth.accounts else None
                if simulated_from_address is None:
                    console.print(f"[{Colors.YELLOW}]Warning: No account available for simulating call, allocation may fail if contract requires sender.[/]")
                    # Try calling without a 'from' address if no accounts are available
                    allocated_amounts_int = function_call.call()
                else:
                    allocated_amounts_int = function_call.call({'from': simulated_from_address})
                console.print(f"[{Colors.DIM}]Contract call simulation returned (int): {allocated_amounts_int}[/]", style=Colors.BLOCKCHAIN)
            except ContractLogicError as sim_error:
                 console.print(f"[{Colors.RED}]Contract logic error during executeFairAllocation simulation (call): {sim_error}[/]")
                 return None # Indicate failure to simulation caller
            except Exception as sim_e:
                 console.print(f"[{Colors.RED}]Error during executeFairAllocation simulation (call): {sim_e}[/]")
                 return None # Indicate failure

            # Convert simulated integer amounts back to floats for the return value
            simulated_allocations_float = {
                region_ids[i]: float(alloc) / self.SCALE_FACTOR
                for i, alloc in enumerate(allocated_amounts_int)
            }

            # --- Now send the actual transaction to change state / emit events ---
            # This runs *after* getting the simulated result needed for the Python environment flow.
            # We log the result but the simulation uses the value from call().
            if self.account: # Only send if a private key/account is configured
                tx_result = self._send_transaction(function_call)
                if not tx_result or tx_result.get('status') != 'success':
                     console.print(f"[{Colors.YELLOW}]Warning: Transaction submission for executeFairAllocation (Drug {drug_id}) failed or was not successful. Allocation based on simulation call result.[/]")
                     # Continue with the simulated result anyway
            else:
                 console.print(f"[{Colors.YELLOW}]Skipping actual transaction for executeFairAllocation (Drug {drug_id}): No private key.[/]")

            return simulated_allocations_float

        except Exception as e:
            # Catch errors during preparation (e.g., integer conversion)
            console.print(f"[{Colors.RED}]Error preparing/calling executeFairAllocation for Drug {drug_id}: {e}[/]")
            # console.print_exception(show_locals=False)
            return None # Indicate failure

    # --- Read Methods ---

    def _read_contract(self, function_call, *args) -> Optional[Any]:
        """Helper for reading data with retries."""
        max_retries = 3
        base_delay = 0.5
        for attempt in range(max_retries):
            try:
                # Use the function call object directly
                result = function_call(*args).call()
                return result
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    console.print(f"[{Colors.YELLOW}]Retrying read operation ({function_call.fn_name}) after error (Attempt {attempt+1}/{max_retries}): {e}. Waiting {delay:.1f}s...[/]")
                    time.sleep(delay)
                else:
                    console.print(f"[bold {Colors.RED}]Error reading contract function '{function_call.fn_name}' after {max_retries} attempts: {e}[/]")
                    return None # Return None after final attempt fails

    def get_regional_case_count(self, region_id: int) -> Optional[int]:
        """Reads the latest case count for a region from the contract with retries."""
        try:
            return self._read_contract(self.contract.functions.getRegionalCaseCount, region_id)
        except Exception as e:
            # This catch is mostly for unexpected errors not caught by _read_contract
            console.print(f"[{Colors.RED}]Unexpected error in get_regional_case_count setup for R{region_id}: {e}[/]")
            return None

    def get_drug_criticality(self, drug_id: int) -> Optional[int]:
        """Reads the drug criticality value from the contract with retries."""
        try:
            return self._read_contract(self.contract.functions.getDrugCriticality, drug_id)
        except Exception as e:
            console.print(f"[{Colors.RED}]Unexpected error in get_drug_criticality setup for D{drug_id}: {e}[/]")
            return None

    def get_contract_owner(self) -> Optional[str]:
        """Reads the owner address from the contract with retries."""
        try:
            return self._read_contract(self.contract.functions.owner) # No arguments needed for owner()
        except Exception as e:
            console.print(f"[{Colors.RED}]Unexpected error in get_contract_owner setup: {e}[/]")
            return None

    # --- Utility ---

    def print_contract_state(self, num_regions: int = 5, num_drugs: int = 3):
        """Queries and prints some key states from the contract for debugging."""
        console.rule(f"[{Colors.BLOCKCHAIN}]Querying Final Blockchain State[/]")
        try:
            owner = self.get_contract_owner()
            console.print(f"Contract Owner: {owner if owner else '[red]Error Reading[/]'}")

            console.print("\n[bold]Regional Case Counts:[/bold]")
            for r_id in range(num_regions):
                # Add slight delay between reads if node rate limits aggressively
                # time.sleep(0.1)
                cases = self.get_regional_case_count(r_id)
                console.print(f"  Region {r_id}: {cases if cases is not None else f'[{Colors.RED}]Read Error[/]'}")

            console.print("\n[bold]Drug Criticalities:[/bold]")
            for d_id in range(num_drugs):
                 # time.sleep(0.1)
                 crit = self.get_drug_criticality(d_id)
                 console.print(f"  Drug {d_id}: {crit if crit is not None else f'[{Colors.RED}]Read Error[/]'}")

        except Exception as e:
            console.print(f"[{Colors.RED}]Error querying final contract state: {e}[/]")
        console.rule()

# --- END OF FILE src/blockchain/interface.py ---