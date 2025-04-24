# --- START OF FILE src/blockchain/interface.py ---

"""
Blockchain interface using Web3.py to interact with the SupplyChainData smart contract.
"""

import json
import time
import math
import numpy as np  # Import numpy for calculations
from web3 import Web3
from web3.exceptions import ContractLogicError  # For handling reverts
from web3.middleware import geth_poa_middleware  # For PoA networks
from typing import Dict, List, Optional, Any
from rich.console import Console
from rich.table import Table  # Import Table for formatting
from rich import box

# Use a shared console or create one if needed (import from config if available, otherwise create)
try:
    # Assuming console and Colors are defined in config
    from config import console, Colors
except ImportError:
    console = Console()
    # Define basic colors if Colors class not available

    class Colors:
        BLOCKCHAIN = "bright_black"
        YELLOW = "yellow"
        RED = "red"
        GREEN = "green"
        DIM = "dim"
        CYAN = "cyan" # Added for consistency


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
        self.SCALE_FACTOR = 1000  # Represents 3 decimal places

        # --- Performance Metrics Attributes ---
        self.tx_latencies = []          # List of latencies for successful/failed tx receipts
        self.read_latencies = []        # List of latencies for read calls
        self.tx_sent_count = 0          # Total transactions attempted to send
        self.tx_success_count = 0       # Count of successful tx (status 1)
        self.tx_failure_count = 0       # Count of failed tx (status 0 or error before receipt)
        self.total_gas_used = 0         # Sum of gasUsed for successful tx
        self.read_call_count = 0        # Total read calls attempted
        self.read_error_count = 0       # Count of read calls that failed (returned None)
        self.last_tx_error = None       # Store last transaction error message
        self.last_read_error = None     # Store last read error message
        # -------------------------------------

        try:
            self.w3 = Web3(Web3.HTTPProvider(node_url))

            # Check connection first
            if not self.w3.is_connected():
                raise ConnectionError(
                    f"Failed to connect to Ethereum node at {node_url}")

            # Inject PoA middleware - necessary for some testnets and potentially local nodes
            self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)

            console.print(
                f"[{Colors.GREEN}]Connected to Ethereum node: {node_url} (Chain ID: {self.w3.eth.chain_id})[/]")

            # Load Contract ABI
            with open(contract_abi_path, 'r') as f:
                contract_abi = json.load(f)

            # Load Contract
            checksum_address = self.w3.to_checksum_address(contract_address)
            self.contract = self.w3.eth.contract(
                address=checksum_address, abi=contract_abi)
            console.print(
                f"[{Colors.GREEN}]SupplyChainData contract loaded at address: {contract_address}[/]")

            # Set up account for transactions if private key is provided
            if private_key:
                # Ensure private key has '0x' prefix
                if not private_key.startswith('0x'):
                    private_key = '0x' + private_key
                self.account = self.w3.eth.account.from_key(private_key)
                # Set default account for calls if needed
                self.w3.eth.default_account = self.account.address
                console.print(
                    f"[{Colors.GREEN}]Transaction account set up: {self.account.address}[/]")
            else:
                console.print(
                    f"[{Colors.YELLOW}]Warning: No private key provided. Only read operations possible.[/]")

            # Test contract connection by reading owner (optional but good check)
            try:
                owner = self.contract.functions.owner().call()
                console.print(f"[{Colors.DIM}]Contract owner found: {owner}[/]")
            except Exception as e:
                console.print(
                    f"[{Colors.YELLOW}]Warning: Could not call contract 'owner' function. Contract may not be deployed correctly or ABI mismatch? Error: {e}[/]")

        except FileNotFoundError:
            console.print(
                f"[bold {Colors.RED}]Error: Contract ABI file not found at {contract_abi_path}[/]")
            raise
        except ConnectionError as e:
            console.print(
                f"[bold {Colors.RED}]Error connecting to Ethereum node: {e}[/]")
            raise
        except Exception as e:
            console.print(
                f"[bold {Colors.RED}]Error initializing BlockchainInterface: {e}[/]")
            console.print_exception(show_locals=True)  # Show traceback for debugging
            raise

    def _get_gas_price(self):
        """Gets gas price based on network conditions."""
        if self.w3.eth.chain_id in [1337, 31337]:  # Common local chain IDs
            return self.w3.to_wei('10', 'gwei')
        try:
            for attempt in range(3):
                try:
                    return self.w3.eth.gas_price
                except Exception as e:
                    if attempt < 2:
                        console.print(
                            f"[{Colors.YELLOW}]Retrying gas price fetch after error: {e}[/]")
                        time.sleep(0.5 * (attempt + 1))
                    else:
                        raise e
            # Fallback if retries fail (should be caught by raise above but safer)
            return self.w3.to_wei('20', 'gwei')
        except Exception as e:
            console.print(
                f"[{Colors.YELLOW}]Warning: Could not fetch gas price, using default 20 gwei. Error: {e}[/]")
            return self.w3.to_wei('20', 'gwei')

    def _send_transaction(self, function_call) -> Optional[Dict[str, Any]]:
        """Builds, signs, sends a transaction and waits for the receipt, tracking metrics."""
        if not self.account:
            console.print(
                f"[bold {Colors.RED}]Error: Cannot send transaction. No private key configured.[/]")
            self.last_tx_error = "No private key configured"
            # Don't count config errors as tx failures
            return None

        start_time = time.monotonic()
        tx_receipt = None
        tx_status = 'error'  # Default status
        error_msg = 'Unknown error during transaction send'
        func_name = "UnknownFunction" # Default name
        try:
            # Get function name for logging
            func_name = function_call.abi['name']
        except (AttributeError, KeyError, TypeError):
            pass # Keep default name

        try:
            sender_address = self.account.address
            nonce = self.w3.eth.get_transaction_count(sender_address)
            gas_price = self._get_gas_price()

            tx_params = {
                'from': sender_address,
                'nonce': nonce,
                'gas': self.gas_limit,
                'gasPrice': gas_price,
            }

            transaction = function_call.build_transaction(tx_params)
            signed_tx = self.w3.eth.account.sign_transaction(
                transaction, self.private_key)

            self.tx_sent_count += 1  # Increment when attempting to send
            tx_hash = self.w3.eth.send_raw_transaction(
                signed_tx.raw_transaction)
            console.print(
                f"[{Colors.DIM}]Tx Sent ({func_name}): {tx_hash.hex()}. Waiting...[/]", style=Colors.BLOCKCHAIN)

            tx_receipt = self.w3.eth.wait_for_transaction_receipt(
                tx_hash, timeout=180)

            if tx_receipt['status'] == 1:
                console.print(
                    f"[{Colors.GREEN}]✓ Tx Success! Block: {tx_receipt['blockNumber']}, Gas: {tx_receipt['gasUsed']}[/]", style=Colors.BLOCKCHAIN)
                self.tx_success_count += 1
                self.total_gas_used += tx_receipt['gasUsed']
                tx_status = 'success'
                error_msg = None
            else:
                console.print(
                    f"[bold {Colors.RED}]❌ Tx Failed (Reverted)! Receipt: {tx_receipt}[/]", style=Colors.BLOCKCHAIN)
                self.tx_failure_count += 1
                tx_status = 'failed'
                error_msg = 'Transaction reverted'

        except ValueError as ve:
            console.print(
                f"[bold {Colors.RED}]Transaction ValueError ({func_name}): {ve}[/]")
            self.tx_failure_count += 1
            error_msg = str(ve)
        except Exception as e:
            console.print(
                f"[bold {Colors.RED}]Error sending transaction ({func_name}): {type(e).__name__} - {e}[/]")
            self.tx_failure_count += 1
            error_msg = str(e)
        finally:
            latency = time.monotonic() - start_time
            self.tx_latencies.append(latency)
            if error_msg:
                self.last_tx_error = error_msg

        if tx_status == 'success':
            return {'status': 'success', 'receipt': tx_receipt}
        else:
            return {'status': tx_status, 'receipt': tx_receipt, 'error': error_msg}

    def _read_contract(self, function_call, *args) -> Optional[Any]:
        """Helper for reading data with retries, tracking metrics."""
        start_time = time.monotonic()
        self.read_call_count += 1
        result = None
        error_msg = None
        func_name = "UnknownFunction" # Default name
        try:
            # Access the 'name' field from the ABI dictionary associated with the function
            func_name = function_call.abi['name']
        except (AttributeError, KeyError, TypeError):
            # Fallback if ABI access fails for some reason
            pass

        max_retries = 3
        base_delay = 0.5
        for attempt in range(max_retries):
            try:
                result = function_call(*args).call()
                error_msg = None  # Clear error on success
                break  # Exit retry loop on success
            except Exception as e:
                error_msg = f"Attempt {attempt+1}/{max_retries}: {type(e).__name__} - {e}"
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    console.print(
                        f"[{Colors.YELLOW}]Retrying read ({func_name}) after error: {e}. Wait {delay:.1f}s...[/]")
                    time.sleep(delay)
                else:
                    console.print(
                        f"[bold {Colors.RED}]Read failed ({func_name}) after {max_retries} attempts: {e}[/]")
                    self.read_error_count += 1  # Increment error count only on final failure
                    self.last_read_error = error_msg  # Store final error
                    result = None  # Ensure result is None on final failure

        latency = time.monotonic() - start_time
        self.read_latencies.append(latency)
        return result

    # --- Write Methods ---

    def update_regional_case_count(self, region_id: int, cases: int) -> Optional[Dict[str, Any]]:
        """Updates the case count for a region via a transaction."""
        if not self.account:
            console.print(
                f"[{Colors.YELLOW}]Skipping blockchain case update for R{region_id}: No private key.[/]")
            return None
        try:
            # console.print(f"[{Colors.DIM}]Preparing blockchain tx: updateRegionalCaseCount(regionId={region_id}, cases={cases})[/]", style=Colors.BLOCKCHAIN)
            function_call = self.contract.functions.updateRegionalCaseCount(
                region_id, cases)
            return self._send_transaction(function_call)
        except Exception as e:
            console.print(
                f"[{Colors.RED}]Error preparing updateRegionalCaseCount transaction: {e}[/]")
            return {'status': 'error', 'error': str(e)}

    def set_drug_criticality(self, drug_id: int, criticality_value: int) -> Optional[Dict[str, Any]]:
        """Sets the drug criticality value via a transaction (likely used during setup)."""
        if not self.account:
            console.print(
                f"[{Colors.YELLOW}]Skipping blockchain criticality set for D{drug_id}: No private key.[/]")
            return None
        try:
            # console.print(f"[{Colors.DIM}]Preparing blockchain tx: setDrugCriticality(drugId={drug_id}, criticalityValue={criticality_value})[/]", style=Colors.BLOCKCHAIN)
            function_call = self.contract.functions.setDrugCriticality(
                drug_id, criticality_value)
            return self._send_transaction(function_call)
        except Exception as e:
            console.print(
                f"[{Colors.RED}]Error preparing setDrugCriticality transaction: {e}[/]")
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
        # Convert float amounts to integers for the contract
        requested_amounts_int = [int(round(r * self.SCALE_FACTOR)) for r in requested_amounts]
        available_inventory_int = int(round(available_inventory * self.SCALE_FACTOR))

        # Handle edge case: If available inventory is zero or negative after scaling, skip blockchain call
        if available_inventory_int <= 0:
            console.print(
                f"[{Colors.YELLOW}]execute_fair_allocation (Drug {drug_id}): Scaled available inventory is zero or less ({available_inventory_int}), skipping blockchain call.[/]")
            return {r_id: 0.0 for r_id in region_ids}  # Return zero allocations

        func_name = "executeFairAllocation" # Default/fallback name
        try:  # Wrap the setup and call in a try block
            function_call = self.contract.functions.executeFairAllocation(
                drug_id, region_ids, requested_amounts_int, available_inventory_int
            )
            try: # Try getting name from ABI
                func_name = function_call.abi['name']
            except (AttributeError, KeyError, TypeError):
                 pass # Keep default name if ABI access fails

            # --- Simulate the call() first to get the return value for the simulation ---
            allocated_amounts_int = None
            try:
                console.print(f"[{Colors.DIM}]Simulating {func_name} call for Drug {drug_id}...[/]", style=Colors.BLOCKCHAIN)
                simulated_from_address = self.account.address if self.account else self.w3.eth.accounts[0] if self.w3.eth.accounts else None
                if simulated_from_address is None:
                    console.print(
                        f"[{Colors.YELLOW}]Warning: No account available for simulating call ({func_name}), allocation may fail if contract requires sender.[/]")
                    allocated_amounts_int = function_call.call()
                else:
                    allocated_amounts_int = function_call.call(
                        {'from': simulated_from_address})

                console.print(
                    f"[{Colors.DIM}]Contract call simulation ({func_name}) returned (int): {allocated_amounts_int}[/]", style=Colors.BLOCKCHAIN)

            # --- Specific Error Handling for the .call() simulation ---
            except ContractLogicError as sim_error:
                console.print(
                    f"[{Colors.RED}]Contract logic error during {func_name} simulation (call): {sim_error}[/]")
                self.last_read_error = f"Sim Call Error ({func_name}): {sim_error}"
                self.read_error_count += 1
                return None  # Indicate failure
            except Exception as sim_e:
                console.print(
                    f"[{Colors.RED}]Unexpected Error during {func_name} simulation (call): {sim_e}[/]")
                self.last_read_error = f"Sim Call Error ({func_name}): {sim_e}"
                self.read_error_count += 1
                return None  # Indicate failure
            # ----------------------------------------------------------

            # Convert simulated integer amounts back to floats for the return value
            simulated_allocations_float = {
                region_ids[i]: float(alloc) / self.SCALE_FACTOR
                for i, alloc in enumerate(allocated_amounts_int)
            }

            # --- Now send the actual transaction to change state / emit events ---
            if self.account:  # Only send if a private key/account is configured
                tx_result = self._send_transaction(function_call)
                if not tx_result or tx_result.get('status') != 'success':
                    console.print(
                        f"[{Colors.YELLOW}]Warning: Transaction submission for {func_name} (Drug {drug_id}) failed or was not successful. Allocation based on simulation call result.[/]")
            else:
                console.print(
                    f"[{Colors.YELLOW}]Skipping actual transaction for {func_name} (Drug {drug_id}): No private key.[/]")

            return simulated_allocations_float

        except Exception as e:
            # Catch errors during preparation before function_call is defined
            console.print(
                f"[{Colors.RED}]Error preparing/calling {func_name} for Drug {drug_id}: {e}[/]")
            self.last_read_error = f"Setup Error ({func_name}): {e}"
            self.read_error_count += 1
            return None  # Indicate failure

    # --- Read Methods ---

    def get_regional_case_count(self, region_id: int) -> Optional[int]:
        """Reads the latest case count for a region from the contract with retries."""
        try:
            # Pass the function object directly to _read_contract
            return self._read_contract(self.contract.functions.getRegionalCaseCount, region_id)
        except Exception as e:
            # This catch is mostly for unexpected errors during setup
            console.print(
                f"[{Colors.RED}]Unexpected error in get_regional_case_count setup for R{region_id}: {e}[/]")
            return None

    def get_drug_criticality(self, drug_id: int) -> Optional[int]:
        """Reads the drug criticality value from the contract with retries."""
        try:
            return self._read_contract(self.contract.functions.getDrugCriticality, drug_id)
        except Exception as e:
            console.print(
                f"[{Colors.RED}]Unexpected error in get_drug_criticality setup for D{drug_id}: {e}[/]")
            return None

    def get_contract_owner(self) -> Optional[str]:
        """Reads the owner address from the contract with retries."""
        try:
            # No arguments needed for owner()
            return self._read_contract(self.contract.functions.owner)
        except Exception as e:
            console.print(
                f"[{Colors.RED}]Unexpected error in get_contract_owner setup: {e}[/]")
            return None

    # --- Utility ---

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculates and returns performance metrics."""
        metrics = {}

        # Transaction Metrics
        metrics['tx_sent_count'] = self.tx_sent_count
        metrics['tx_success_count'] = self.tx_success_count
        metrics['tx_failure_count'] = self.tx_failure_count
        if self.tx_sent_count > 0:
            metrics['tx_success_rate'] = (
                self.tx_success_count / self.tx_sent_count) * 100
        else:
            metrics['tx_success_rate'] = "N/A (0 sent)"

        if self.tx_latencies:
            metrics['tx_latency_avg_s'] = np.mean(self.tx_latencies)
            metrics['tx_latency_max_s'] = np.max(self.tx_latencies)
            metrics['tx_latency_min_s'] = np.min(self.tx_latencies)
            metrics['tx_latency_p95_s'] = np.percentile(self.tx_latencies, 95)
        else:
            metrics['tx_latency_avg_s'] = 0
            metrics['tx_latency_max_s'] = 0
            metrics['tx_latency_min_s'] = 0
            metrics['tx_latency_p95_s'] = 0

        metrics['total_gas_used'] = self.total_gas_used
        if self.tx_success_count > 0:
            metrics['avg_gas_per_successful_tx'] = self.total_gas_used / self.tx_success_count
        else:
            metrics['avg_gas_per_successful_tx'] = 0

        # Read Metrics
        metrics['read_call_count'] = self.read_call_count
        metrics['read_error_count'] = self.read_error_count
        if self.read_call_count > 0:
            metrics['read_success_rate'] = (
                (self.read_call_count - self.read_error_count) / self.read_call_count) * 100
        else:
            metrics['read_success_rate'] = "N/A (0 reads)"

        if self.read_latencies:
            metrics['read_latency_avg_s'] = np.mean(self.read_latencies)
            metrics['read_latency_max_s'] = np.max(self.read_latencies)
            metrics['read_latency_min_s'] = np.min(self.read_latencies)
            metrics['read_latency_p95_s'] = np.percentile(self.read_latencies, 95)
        else:
            metrics['read_latency_avg_s'] = 0
            metrics['read_latency_max_s'] = 0
            metrics['read_latency_min_s'] = 0
            metrics['read_latency_p95_s'] = 0

        metrics['last_tx_error'] = self.last_tx_error
        metrics['last_read_error'] = self.last_read_error

        return metrics

    def print_contract_state(self, num_regions: int = 5, num_drugs: int = 3):
        """Queries and prints some key states from the contract for debugging."""
        console.rule(f"[{Colors.BLOCKCHAIN}]Querying Final Blockchain State[/]")
        try:
            owner = self.get_contract_owner()
            console.print(
                f"Contract Owner: {owner if owner else '[red]Error Reading[/]'}")

            console.print("\n[bold]Regional Case Counts:[/bold]")
            for r_id in range(num_regions):
                # Add slight delay between reads if node rate limits aggressively
                # time.sleep(0.1)
                cases = self.get_regional_case_count(r_id)
                console.print(
                    f"  Region {r_id}: {cases if cases is not None else f'[{Colors.RED}]Read Error[/]'}")

            console.print("\n[bold]Drug Criticalities:[/bold]")
            for d_id in range(num_drugs):
                # time.sleep(0.1)
                crit = self.get_drug_criticality(d_id)
                console.print(
                    f"  Drug {d_id}: {crit if crit is not None else f'[{Colors.RED}]Read Error[/]'}")

        except Exception as e:
            console.print(
                f"[{Colors.RED}]Error querying final contract state: {e}[/]")
        console.rule()

# --- END OF FILE src/blockchain/interface.py ---