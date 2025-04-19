"""
Interface between the pandemic simulation and the Ethereum blockchain.
"""

import json
import os
from web3 import Web3, HTTPProvider
# from web3.middleware import ExtraDataToPOAMiddleware # Removed for Hardhat

class BlockchainInterface:
    """Interface between the pandemic simulation and the Ethereum blockchain."""

    def __init__(self, node_url="http://127.0.0.1:8545"):
        # Connect to the Ethereum node
        self.w3 = Web3(HTTPProvider(node_url))
        self.node_url = node_url

        # Remove unnecessary middleware for Hardhat local node
        # self.w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

        # Check connection
        if not self.w3.is_connected():
            raise ConnectionError(f"Failed to connect to Ethereum node at {node_url}")

        # --- IMPORTANT: Use the ACTUAL deployed contract address ---
        # --- Run `npx hardhat run scripts/deploy.js --network localhost` first ---
        self.contract_address = "0x5FbDB2315678afecb367f032d93F642f64180aa3" # !!! UPDATE THIS ADDRESS !!!
        # ---

        # ABI (ensure this matches your compiled contract, including view functions)
        self.contract_abi = [
            # Actions requiring transactions
            {"inputs": [{"internalType": "string", "name": "_name", "type": "string"}, {"internalType": "uint8", "name": "_criticality", "type": "uint8"}, {"internalType": "uint256", "name": "_productionCapacity", "type": "uint256"}, {"internalType": "uint256", "name": "_baseProduction", "type": "uint256"}], "name": "addDrug", "outputs": [], "stateMutability": "nonpayable", "type": "function"},
            {"inputs": [{"internalType": "string", "name": "_name", "type": "string"}, {"internalType": "string", "name": "_regionType", "type": "string"}, {"internalType": "uint256", "name": "_population", "type": "uint256"}], "name": "addRegion", "outputs": [], "stateMutability": "nonpayable", "type": "function"},
            {"inputs": [{"internalType": "uint256", "name": "_regionId", "type": "uint256"}, {"internalType": "uint256", "name": "_newCases", "type": "uint256"}], "name": "updateCases", "outputs": [], "stateMutability": "nonpayable", "type": "function"},
            {"inputs": [{"internalType": "uint256", "name": "_drugId", "type": "uint256"}, {"internalType": "uint256", "name": "_regionId", "type": "uint256"}, {"internalType": "uint256", "name": "_amount", "type": "uint256"}], "name": "recordAllocation", "outputs": [], "stateMutability": "nonpayable", "type": "function"},
            # View functions (no transaction needed)
            {"inputs": [{"internalType": "uint256", "name": "_drugId", "type": "uint256"}, {"internalType": "uint256[]", "name": "_regionIds", "type": "uint256[]"}, {"internalType": "uint256[]", "name": "_requestedAmounts", "type": "uint256[]"}, {"internalType": "uint256", "name": "_availableInventory", "type": "uint256"}], "name": "calculateAllocation", "outputs": [{"internalType": "uint256[]", "name": "", "type": "uint256[]"}], "stateMutability": "view", "type": "function"},
            {"inputs": [], "name": "drugCount", "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"},
            {"inputs": [], "name": "regionCount", "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"},
            {"inputs": [{"internalType": "uint256", "name": "", "type": "uint256"}], "name": "drugs", "outputs": [{"internalType": "string", "name": "name", "type": "string"}, {"internalType": "enum AllocationStrategy.Criticality", "name": "criticality", "type": "uint8"}, {"internalType": "uint256", "name": "productionCapacity", "type": "uint256"}, {"internalType": "uint256", "name": "baseProduction", "type": "uint256"}, {"internalType": "bool", "name": "active", "type": "bool"}], "stateMutability": "view", "type": "function"},
            {"inputs": [{"internalType": "uint256", "name": "", "type": "uint256"}], "name": "regions", "outputs": [{"internalType": "string", "name": "name", "type": "string"}, {"internalType": "string", "name": "regionType", "type": "string"}, {"internalType": "uint256", "name": "population", "type": "uint256"}, {"internalType": "uint256", "name": "activeCases", "type": "uint256"}, {"internalType": "bool", "name": "active", "type": "bool"}], "stateMutability": "view", "type": "function"},
            {"inputs": [{"internalType": "uint256", "name": "_drugId", "type": "uint256"}], "name": "getAllocationHistory", "outputs": [{"components": [{"internalType": "uint256", "name": "drugId", "type": "uint256"}, {"internalType": "uint256", "name": "regionId", "type": "uint256"}, {"internalType": "uint256", "name": "amount", "type": "uint256"}, {"internalType": "uint256", "name": "timestamp", "type": "uint256"}], "internalType": "struct AllocationStrategy.Allocation[]", "name": "", "type": "tuple[]"}], "stateMutability": "view", "type": "function"}
            # Add other necessary ABI parts (e.g., events, owner, member checks if called directly)
        ]

        # Get contract instance
        self.contract = self.w3.eth.contract(address=self.contract_address, abi=self.contract_abi)

        # --- Define Addresses (Match your Hardhat setup from `setup-members.js`) ---
        self.owner_address = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"      # Account 0
        self.manufacturer_address = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8" # Account 1
        self.distributor_addresses = {
            0: "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC",  # Account 2 (Region 0)
            1: "0x90F79bf6EB2c4f870365E785982E1f101E93b906"   # Account 3 (Region 1)
            # Add more if needed
        }
        self.hospital_addresses = {
            0: "0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65",  # Account 4 (Region 0)
            1: "0x9965507D1a55bcC2695C58ba16FB37d819B0A4dc"   # Account 5 (Region 1)
            # Add more if needed
        }
        # ---

        # --- VERY IMPORTANT: Store PRIVATE KEYS for accounts sending transactions ---
        # --- These are the default Hardhat node keys. Replace if needed. ---
        # --- WARNING: NEVER commit real private keys to code repositories! Use environment variables or secure secret management in production. ---
        self.private_keys = {
            self.owner_address: "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80", # Key for Account 0
            self.manufacturer_address: "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d", # Key for Account 1
            self.distributor_addresses[0]: "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a", # Key for Account 2
            self.distributor_addresses[1]: "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6", # Key for Account 3
            self.hospital_addresses[0]: "0x47e179ec197488593b187f80a00eb0da91f1b9d0b13f8733639f19c30a34926a", # Key for Account 4
            self.hospital_addresses[1]: "0x8b3a350cf5c34c9194ca85829a2df0ec3153be0318b5e2d3348e872092edffba", # Key for Account 5
        }
        # ---

    def _send_transaction(self, function_call, from_account_address):
        """Builds, signs, and sends a transaction for the specified account."""
        normalized_from_address = self.w3.to_checksum_address(from_account_address)

        if normalized_from_address not in self.private_keys:
             print(f"[ERROR] Private key for account {normalized_from_address} not found in self.private_keys.")
             return {'status': 'error', 'message': f"Private key for account {normalized_from_address} not found."}

        private_key = self.private_keys[normalized_from_address]
        account = self.w3.eth.account.from_key(private_key)

        # Double-check the address derived from the key matches the expected address
        if account.address.lower() != normalized_from_address.lower():
             print(f"[ERROR] Address mismatch for private key! Expected {normalized_from_address}, derived {account.address}")
             return {'status': 'error', 'message': f"Address mismatch for private key! Expected {normalized_from_address}, derived {account.address}"}

        try:
            # Get the nonce for the *sending* account
            nonce = self.w3.eth.get_transaction_count(account.address)
            print(f"  [TX Info] Sending from: {account.address}, Nonce: {nonce}") # Debug log

            # Build transaction specifying the correct 'from' and 'nonce'
            tx_params = {
                'from': account.address,
                'nonce': nonce,
                # 'gas': 300000,  # Set a reasonable gas limit, or estimate below
                'gasPrice': self.w3.eth.gas_price
                # Optional: specify chainId for robustness, especially on non-dev networks
                # 'chainId': self.w3.eth.chain_id
            }

            # Estimate gas (optional but recommended)
            try:
                 estimated_gas = function_call.estimate_gas(tx_params)
                 tx_params['gas'] = int(estimated_gas * 1.2) # Add 20% buffer
                 print(f"  [TX Info] Estimated Gas: {estimated_gas}, Using Gas Limit: {tx_params['gas']}")
            except Exception as estimate_exception:
                 print(f"  [TX Warning] Gas estimation failed: {estimate_exception}. Using fixed limit 300000.")
                 # Often, estimation failure means the transaction will revert.
                 tx_params['gas'] = 300000


            # Build the final transaction object
            tx = function_call.build_transaction(tx_params)

            # Sign transaction with the correct private key
            signed_tx = self.w3.eth.account.sign_transaction(tx, private_key)

            # Send raw transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            print(f"  [TX Info] Transaction sent: {tx_hash.hex()}")

            # Wait for transaction receipt
            # Add a timeout to prevent hanging indefinitely if the node has issues
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            print(f"  [TX Info] Transaction confirmed in block: {tx_receipt.blockNumber}")

            # Check status code for success (1) or failure (0)
            if tx_receipt.status == 0:
                 print(f"  [TX ERROR] Transaction {tx_hash.hex()} failed (REVERTED). Status: {tx_receipt.status}. Check node logs for revert reason.")
                 # Attempt to get revert reason (experimental, may not work reliably)
                 # try:
                 #     revert_reason = self.w3.eth.call({'to': tx['to'], 'from': tx['from'], 'value': tx.get('value', 0), 'data': tx['data']}, tx_receipt.blockNumber - 1)
                 #     # Decode revert reason - this is complex, often needs specific ABI handling
                 #     print(f"    Potential Revert Reason (Hex): {revert_reason.hex()}")
                 # except Exception as call_e:
                 #     print(f"    Could not directly call failed transaction to get revert reason: {call_e}")

                 return {
                     'status': 'error',
                     'message': f'Transaction reverted. Hash: {tx_hash.hex()}',
                     'tx_hash': tx_hash.hex(),
                     'block': tx_receipt.blockNumber,
                     'receipt': tx_receipt # Include full receipt for debugging
                 }

            # Success
            return {
                'status': 'success',
                'tx_hash': tx_hash.hex(),
                'block': tx_receipt.blockNumber,
                'receipt': tx_receipt # Include full receipt for info
            }

        except ValueError as ve: # Catch specific web3 errors like nonce issues, insufficient funds
             print(f"  [TX ERROR] ValueError sending transaction from {account.address}: {ve}")
             # This might contain info about reverts if gas estimation failed earlier
             return {'status': 'error', 'message': f"ValueError: {ve}"}
        except Exception as e:
            # Catch other unexpected errors during transaction sending/waiting
            print(f"  [TX ERROR] Unexpected Error sending transaction from {account.address}: {repr(e)}")
            import traceback
            traceback.print_exc() # Print stack trace for debugging
            return {
                'status': 'error',
                'message': f"Unexpected Error: {repr(e)}"
            }

    def record_allocation(self, drug_id, region_id, amount):
        """Record a drug allocation on the blockchain. Sent by Manufacturer."""
        print(f"[Blockchain Call] recordAllocation(drug={drug_id}, region={region_id}, amount={amount})")
        # Manufacturer needs to send this transaction
        sender_address = self.manufacturer_address
        function_call = self.contract.functions.recordAllocation(
            int(drug_id), int(region_id), int(amount) # Ensure correct types
        )
        result = self._send_transaction(function_call, sender_address)
        print(f"  [Blockchain Result] recordAllocation: {result['status']}")
        return result

    def calculate_optimal_allocation(self, drug_id, region_requests, available_inventory):
        """Calculate optimal allocation using the smart contract (view function)."""
        print(f"[Blockchain Call] calculateAllocation(drug={drug_id}, requests={region_requests}, available={available_inventory}) - View")
        try:
            region_ids = []
            requested_amounts = []
            # Keys in region_requests might be str or int, ensure they are ints for contract
            for region_id, amount in region_requests.items():
                region_ids.append(int(region_id))
                requested_amounts.append(int(amount)) # Amount should likely be int for contract

            # This is a view function, no transaction needed, just .call()
            allocations = self.contract.functions.calculateAllocation(
                int(drug_id), region_ids, requested_amounts, int(available_inventory)
            ).call() # No 'from' needed for view calls

            result = {}
            for i, region_id in enumerate(region_ids):
                result[region_id] = allocations[i] # Contract returns uint[], keep as number

            print(f"  [Blockchain Result] calculateAllocation Result: {result}")
            return result
        except Exception as e:
            print(f"  [Blockchain ERROR] Error calling calculateAllocation view function: {e}")
            # Fallback logic remains the same
            result = {}
            num_regions = len(region_requests)
            if num_regions > 0 and available_inventory > 0:
                equal_amount = available_inventory / num_regions
                for region_id_key, req_amount in region_requests.items():
                     # Cap allocation at requested amount and ensure non-negative
                     result[int(region_id_key)] = max(0, min(equal_amount, req_amount))
            else: # No regions or no inventory
                 for region_id_key in region_requests:
                      result[int(region_id_key)] = 0
            print(f"  [Blockchain Fallback] Falling back to equal distribution: {result}")
            return result


    def update_case_data(self, region_id, new_cases):
        """Update case data for a region. Sent by Distributor/Hospital for their region, or Owner."""
        print(f"[Blockchain Call] updateCases(region={region_id}, cases={new_cases})")

        # Determine who *should* send based on the region
        # In the simulation, the Distributor/Hospital agent for region_id would trigger this.
        # Here, we map region_id back to the appropriate address.
        sender_address = self.distributor_addresses.get(region_id) # Prefer distributor
        if not sender_address:
            sender_address = self.hospital_addresses.get(region_id) # Fallback to hospital
        if not sender_address:
             # If no specific actor found (e.g., region ID invalid or not mapped), fallback to owner
             print(f"  [Blockchain Warning] No specific actor found for region {region_id} case update. Using owner address.")
             sender_address = self.owner_address

        # Ensure address is checksummed before use
        sender_address = self.w3.to_checksum_address(sender_address)

        function_call = self.contract.functions.updateCases(
            int(region_id), int(new_cases) # Ensure correct types
        )
        result = self._send_transaction(function_call, sender_address)
        print(f"  [Blockchain Result] updateCases: {result['status']}")
        return result

    def print_contract_state(self):
        """Print the current state of the contract."""
        try:
            print("\n--- Contract State ---")
            # Get drug count
            drug_count = self.contract.functions.drugCount().call()
            print(f"Drug count: {drug_count}")

            # Get drug details
            print("\nDrugs:")
            for i in range(drug_count):
                # Using the ABI definition: returns (string name, uint8 criticality, uint cap, uint base, bool active)
                drug = self.contract.functions.drugs(i).call()
                criticality_str = ["Low", "Medium", "High", "Critical"][drug[1]] # Convert enum index
                print(f"  Drug {i}: Name='{drug[0]}', Crit={criticality_str}({drug[1]}), Cap={drug[2]}, Base={drug[3]}, Active={drug[4]}")

            # Get region count
            region_count = self.contract.functions.regionCount().call()
            print(f"\nRegion count: {region_count}")

            # Get region details
            print("\nRegions:")
            for i in range(region_count):
                # Using the ABI definition: returns (string name, string type, uint pop, uint cases, bool active)
                region = self.contract.functions.regions(i).call()
                print(f"  Region {i}: Name='{region[0]}', Type='{region[1]}', Pop={region[2]}, Cases={region[3]}, Active={region[4]}")

            # Get allocation history for each drug
            print("\nAllocation History:")
            total_allocs = 0
            for i in range(drug_count):
                # Using the ABI definition: returns tuple(uint drugId, uint regionId, uint amount, uint timestamp)[]
                history = self.contract.functions.getAllocationHistory(i).call()
                total_allocs += len(history)
                print(f"  Drug {i} ('{self.contract.functions.drugs(i).call()[0]}'): {len(history)} allocations")
                # Print only a few recent ones if history is long
                max_to_print = 5
                start_index = max(0, len(history) - max_to_print)
                for j in range(start_index, len(history)):
                    alloc = history[j]
                    # Access tuple elements by index: alloc[0]=drugId, alloc[1]=regionId, alloc[2]=amount, alloc[3]=timestamp
                    print(f"    - Region: {alloc[1]}, Amount: {alloc[2]}, Timestamp: {alloc[3]}")
            print(f"Total allocations recorded across all drugs: {total_allocs}")
            print("--- End Contract State ---")

            return True
        except Exception as e:
            print(f"Error printing contract state: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            return False