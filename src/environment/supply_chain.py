# --- START OF FILE src/environment/supply_chain.py ---

"""
Pandemic supply chain environment simulation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import math
import json # For deep copying history

# Import the specific allocation tool needed for the fallback logic
from src.tools.allocation import allocation_priority_tool

# Import the BlockchainInterface if needed (type hinting, instantiation)
try:
    from src.blockchain.interface import BlockchainInterface
except ImportError:
    BlockchainInterface = None # Allow running without blockchain dependencies if not installed

# Import colors for printing
from config import Colors


class PandemicSupplyChainEnvironment:
    """Simulates a pandemic supply chain with manufacturers, distributors, and hospitals."""

    def __init__(
        self,
        scenario_generator,
        # --- INCREASED INITIAL INVENTORIES (as per previous recommendation) ---
        initial_manufacturer_inventory: float = 100000,
        initial_distributor_inventory: float = 40000,
        initial_hospital_inventory: float = 10000,
        initial_warehouse_inventory: float = 50000,
        # ------------------------------------
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
        # Ensure use_blockchain is only True if an interface object was actually provided
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

        # Warehouse inventory (separate from manufacturer's usable stock)
        # Initialize using the potentially non-zero initial_warehouse_inventory parameter
        self.warehouse_inventories = {drug_id: initial_warehouse_inventory for drug_id in range(self.num_drugs)}

        # Pipelines {from_id: {to_id: {drug_id: [(amount, arrival_day), ...]}}}
        self.pipelines = {}
        node_ids = list(range(2 * self.num_regions + 1)) # 0=Manu, 1..N=Dist, N+1..2N=Hosp
        for from_id in node_ids:
            self.pipelines[from_id] = {}
            for to_id in node_ids:
                 if from_id != to_id: # No pipeline to self
                    self.pipelines[from_id][to_id] = {drug_id: [] for drug_id in range(self.num_drugs)}

        # Metrics
        self.stockouts = {drug_id: {r: 0 for r in range(self.num_regions)} for drug_id in range(self.num_drugs)} # Days with stockout at hospital
        self.unfulfilled_demand = {drug_id: {r: 0.0 for r in range(self.num_regions)} for drug_id in range(self.num_drugs)} # Units unfulfilled at hospital
        self.total_demand = {drug_id: {r: 0.0 for r in range(self.num_regions)} for drug_id in range(self.num_drugs)} # Total units demanded at hospital
        self.patient_impact = {r: 0.0 for r in range(self.num_regions)} # Weighted score based on unfulfilled and criticality

        # History Tracking
        self.demand_history = []        # Records hospital demand events
        self.order_history = []         # Records orders placed between nodes
        self.allocation_history = []    # Records allocations shipped (pipeline entries)
        self.stockout_history = []      # Records hospital stockout events (unfulfilled > 0)
        self.production_history = []    # Records manufacturer production amounts and release status
        self.warehouse_release_history = [] # Records when items move WH -> Manu inventory
        self.inventory_history = {}     # Daily snapshot {day: {drug_id: {node_id: quantity}}}
        self.warehouse_history = {}     # Daily snapshot {day: {drug_id: quantity}}
        self.pending_allocations = {}   # For batching manufacturer allocations {drug_id_str: {region_id_str: amount}}

        # Configuration (can be set externally after init)
        self.verbose = False
        self.warehouse_release_delay = 1 # Default warehouse delay
        self.allocation_batch_frequency = 1 # Default: Daily allocation

        # Initialize history for day 0
        self._record_daily_history()

        # --- Blockchain Setup (Optional: Set initial state like criticalities) ---
        if self.use_blockchain:
            self._initialize_blockchain_state()

    def _print(self, message):
        """Helper to safely print using the stored console if verbose."""
        if self.verbose and self.console:
            self.console.print(message)

    def _initialize_blockchain_state(self):
        """Sets initial static data on the blockchain (e.g., drug criticalities)."""
        if not self.use_blockchain or not self.blockchain:
            return
        self.console.print("[cyan]Initializing blockchain state (drug criticalities)...[/]")
        setup_successful = True
        for drug in self.scenario.drugs:
            drug_id = drug['id']
            crit_val = drug.get('criticality_value', 1) # Default to 1 if missing
            try:
                # Optional: Check if already set
                current_bc_crit = self.blockchain.get_drug_criticality(drug_id)
                if current_bc_crit is None or current_bc_crit != crit_val:
                     self._print(f"[dim]Setting BC criticality for Drug {drug_id} to {crit_val} (Current: {current_bc_crit})...[/dim]")
                     tx_result = self.blockchain.set_drug_criticality(drug_id, crit_val)
                     if not tx_result or tx_result.get('status') != 'success':
                         self._print(f"[red]Failed to set blockchain criticality for Drug {drug_id}[/]")
                         setup_successful = False
                     # Add a small delay after write operations if needed
                     # import time; time.sleep(0.2)
                else:
                     self._print(f"[dim]BC criticality for Drug {drug_id} already set to {crit_val}.[/dim]")

            except Exception as e:
                self._print(f"[red]Error setting blockchain criticality for Drug {drug_id}: {e}[/]")
                setup_successful = False
        if setup_successful:
            self.console.print("[green]✓ Blockchain state initialized.[/]")
        else:
            self.console.print("[yellow]Warning: Failed to initialize some blockchain state.[/]")


    def _record_daily_history(self):
        """Records inventory snapshots for the current day."""
        # Deep copy inventories to avoid modifying history later
        try:
            current_inv_snapshot = json.loads(json.dumps(self.inventories)) # Simple deep copy using JSON
        except TypeError as e:
            self._print(f"[red]Error deep copying inventories for history: {e}. History may be inaccurate.[/]")
            current_inv_snapshot = self.inventories # Fallback to shallow copy (risk of modification)
        self.inventory_history[self.current_day] = current_inv_snapshot
        self.warehouse_history[self.current_day] = self.warehouse_inventories.copy()

    def reset(self):
        """Reset the environment to its initial state."""
        scenario = self.scenario # Keep the same scenario generator
        # Temporarily store configurable attributes
        current_verbose = getattr(self, 'verbose', False)
        current_delay = getattr(self, 'warehouse_release_delay', 1)
        current_freq = getattr(self, 'allocation_batch_frequency', 1)

        # Re-initialize using the __init__ method, which now has updated defaults
        self.__init__(
            scenario_generator=scenario,
            # Initial inventories will use the new defaults from __init__
            blockchain_interface=self.blockchain, # Keep existing interface object
            use_blockchain=self.use_blockchain,   # Keep existing flag
            console=self.console                  # Keep existing console
        )
        # Restore configurable attributes
        self.verbose = current_verbose
        self.warehouse_release_delay = current_delay
        self.allocation_batch_frequency = current_freq
        return self.get_observations()

    def get_current_simulation_cases(self) -> Dict[int, int]:
        """Calculates the simulated active cases for the *current* day for all regions."""
        regional_current_cases = {}
        # current_day is the day number we are STARTING (1-based in logs, 0-based index here)
        # scenario_length is total days, so max index is scenario_length - 1
        current_day_index = min(self.current_day, self.scenario_length - 1)

        for region_id in range(self.num_regions):
            region_current_cases = 0 # Default
            # Get current cases for this region from simulation curve
            # Ensure region_id exists and day index is valid
            if region_id in self.scenario.epidemic_curves and \
               current_day_index < len(self.scenario.epidemic_curves[region_id]):
                # Access curve using the 0-based index
                region_current_cases = self.scenario.epidemic_curves[region_id][current_day_index]
            else:
                 # Log a warning if data is missing, although curves should be generated for all regions
                 if self.verbose: self._print(f"[yellow]Warning: Missing or invalid epidemic curve data for Region {region_id} on Day Index {current_day_index}. Using 0 cases.[/]")

            regional_current_cases[region_id] = int(round(max(0, region_current_cases)))
        return regional_current_cases

    def _process_production(self, production_actions: Dict):
        """Process drug production -> warehouse."""
        for drug_id, amount in production_actions.items():
            try:
                drug_id = int(drug_id)
                if not (0 <= drug_id < self.num_drugs): continue
                # Use current_day index for capacity check
                day_idx = min(self.current_day, self.scenario_length - 1)
                capacity = self.scenario.get_manufacturing_capacity(day_idx, drug_id)
                actual_production = min(max(0.0, float(amount)), capacity) # Ensure non-negative and capped

                if actual_production > 0:
                    self.warehouse_inventories[drug_id] = self.warehouse_inventories.get(drug_id, 0.0) + actual_production
                    self.production_history.append({
                        "day": self.current_day, "drug_id": drug_id, "amount": actual_production,
                        "released": False, "release_day": None
                    })
                    if self.verbose:
                        drug_name = self.scenario.drugs[drug_id].get("name", f"Drug-{drug_id}")
                        self._print(f"[{Colors.MANUFACTURER}]Production: {actual_production:.1f} units of {drug_name} -> warehouse[/]")
            except (ValueError, TypeError, KeyError) as e:
                 self._print(f"[yellow]Error processing production for drug_id {drug_id}: {e}[/]")

    def _process_warehouse_release(self):
        """Release inventory warehouse -> manufacturer after delay."""
        # Use current value of the delay attribute
        release_delay = getattr(self, 'warehouse_release_delay', 1)
        if release_delay < 0: release_delay = 0 # Ensure non-negative delay

        eligible = [e for e in self.production_history if not e.get("released", False) and e["day"] <= self.current_day - release_delay]

        for entry in eligible:
            try:
                drug_id = entry["drug_id"]
                amount_produced = entry["amount"]
                # Ensure warehouse inventory exists for this drug
                available_in_warehouse = self.warehouse_inventories.get(drug_id, 0.0)
                amount_to_release = min(amount_produced, available_in_warehouse)

                if amount_to_release > 0.01: # Only release meaningful amounts
                    self.warehouse_inventories[drug_id] -= amount_to_release
                    # Ensure manufacturer inventory exists before adding
                    self.inventories.setdefault(drug_id, {}).setdefault(0, 0.0)
                    self.inventories[drug_id][0] += amount_to_release

                    entry["released"] = True
                    entry["release_day"] = self.current_day
                    self.warehouse_release_history.append({
                        "day": self.current_day, "drug_id": drug_id, "amount": amount_to_release,
                        "production_day": entry["day"], "delay_days": self.current_day - entry["day"]
                    })
                    if self.verbose:
                        drug_name = self.scenario.drugs[drug_id].get("name", f"Drug-{drug_id}")
                        delay_taken = self.current_day - entry['day']
                        self._print(f"[{Colors.CYAN}]Warehouse release: {amount_to_release:.1f} units of {drug_name} released after {delay_taken} days[/]")
            except (KeyError, ValueError, TypeError) as e:
                self._print(f"[yellow]Error processing warehouse release for entry {entry}: {e}[/]")

    def _process_batch_allocation(self, allocation_actions: Dict):
        """ Accumulates or processes manufacturer allocations based on frequency. """
        batch_frequency = getattr(self, 'allocation_batch_frequency', 1)
        # Use > 0 check for batch day calc, modulo works for day 1 if freq is 1
        is_batch_processing_day = self.current_day >= 0 and (self.current_day + 1) % batch_frequency == 0
        # Day 0 is not typically a processing day unless freq=1

        # Store current day's intended allocations from the agent
        for drug_id, region_allocs in allocation_actions.items():
            try:
                drug_id_str = str(int(drug_id))
                if drug_id_str not in self.pending_allocations: self.pending_allocations[drug_id_str] = {}
                for region_id, amount in region_allocs.items():
                    region_id_str = str(int(region_id))
                    current = self.pending_allocations[drug_id_str].get(region_id_str, 0.0)
                    # Accumulate requested amount
                    self.pending_allocations[drug_id_str][region_id_str] = current + max(0.0, float(amount))
            except (ValueError, TypeError) as e:
                self._print(f"[yellow]Error processing/accumulating allocation action: {e}[/]")
                continue

        # If it's batch day (or frequency is 1), process the accumulated allocations
        if batch_frequency == 1 or is_batch_processing_day:
            if self.verbose and batch_frequency > 1 and self.pending_allocations:
                processing_day_num = self.current_day + 1 # Use 1-based day for logging
                self._print(f"[{Colors.GREEN}]Allocation batching: Processing accumulated batch on day {processing_day_num} (Freq={batch_frequency}d)[/]")

            # Convert pending allocations to the format needed by shipment function {int_drug_id: {int_region_id: float_amount}}
            integer_allocations_requests = {}
            for drug_id_str, region_allocs in self.pending_allocations.items():
                try:
                    drug_id = int(drug_id_str)
                    integer_allocations_requests[drug_id] = {}
                    for region_id_str, amount in region_allocs.items():
                        try:
                            region_id = int(region_id_str)
                            if amount > 0.01: # Only process meaningful amounts
                                integer_allocations_requests[drug_id][region_id] = amount
                        except ValueError: continue
                    # Remove drug if no regions have allocations after conversion/filtering
                    if not integer_allocations_requests.get(drug_id):
                         if drug_id in integer_allocations_requests:
                              del integer_allocations_requests[drug_id]
                except ValueError: continue

            if integer_allocations_requests: # Only call process if there's something to allocate
                 # Process the actual shipment based on requests and availability
                 self._process_manufacturer_allocation_shipment(integer_allocations_requests)

            self.pending_allocations = {} # Clear pending batch regardless
        elif self.verbose and batch_frequency > 1:
             current_day_num = self.current_day + 1 # Use 1-based day for logging
             next_batch_day = ((current_day_num -1) // batch_frequency + 1) * batch_frequency
             self._print(f"[dim]Allocation batching: Accumulating allocations, next processing day {next_batch_day}[/dim]")


    def _process_manufacturer_allocation_shipment(self, allocation_requests: Dict):
        """Process actual allocation shipment manufacturer -> distributors (creates pipeline entries)."""
        # This function receives the potentially batched and combined *requests*
        for drug_id, region_requests in allocation_requests.items():
            try:
                drug_id = int(drug_id)
                available_inventory = self.inventories.get(drug_id, {}).get(0, 0.0)

                if available_inventory <= 0.01: continue # Skip if effectively no inventory

                requesting_region_ids = list(region_requests.keys())
                requested_amounts = [region_requests[r_id] for r_id in requesting_region_ids]

                # --- Execute Allocation Logic (Blockchain or Local Fallback) ---
                final_allocations = self._calculate_fair_allocation(
                    drug_id,
                    dict(zip(requesting_region_ids, requested_amounts)), # Recreate dict for consistency
                    available_inventory
                )
                # ----------------------------------------------------------------

                if not final_allocations:
                     self._print(f"[yellow]Allocation calculation failed for Drug {drug_id}, skipping shipment.[/]")
                     continue

                # Process the calculated final allocations
                total_allocated_this_drug = 0
                for region_id, amount_to_allocate in final_allocations.items():
                    if amount_to_allocate <= 1e-6: continue # Skip negligible allocations

                    region_id = int(region_id) # Ensure integer
                    distributor_id = region_id + 1 # Map region_id to distributor node_id

                    # Deduct from manufacturer inventory (capped at current available, though calc should handle it)
                    current_manu_inv = self.inventories.get(drug_id, {}).get(0, 0.0)
                    actual_shipment = min(amount_to_allocate, current_manu_inv)

                    if actual_shipment <= 1e-6: continue

                    # Calculate lead time considering disruptions
                    day_idx = min(self.current_day, self.scenario_length - 1)
                    base_lead_time = 1 + np.random.poisson(1) # e.g., 1-3 days average Manu -> Dist
                    transport_capacity_factor = self.scenario.get_transportation_capacity(day_idx, region_id)
                    adjusted_lead_time = max(1, int(round(base_lead_time / max(0.1, transport_capacity_factor)))) # Avoid division by zero
                    arrival_day = self.current_day + adjusted_lead_time # Arrival is relative to *current* day

                    # Add to pipeline (ensure structure exists)
                    self.pipelines.setdefault(0, {}).setdefault(distributor_id, {}).setdefault(drug_id, []).append((actual_shipment, arrival_day))

                    # Deduct from manufacturer inventory
                    self.inventories[drug_id][0] -= actual_shipment
                    total_allocated_this_drug += actual_shipment

                    # Log allocation history (represents what was shipped)
                    self.allocation_history.append({
                        "day": self.current_day, "drug_id": drug_id, "from_id": 0,
                        "to_id": distributor_id, "amount": actual_shipment, "arrival_day": arrival_day
                    })

                if self.verbose and total_allocated_this_drug > 0.01:
                    drug_name = self.scenario.drugs[drug_id].get("name", f"Drug-{drug_id}")
                    self._print(f"[{Colors.GREEN}]Shipped {total_allocated_this_drug:.1f} units of {drug_name} from Manufacturer.[/]")

            except (KeyError, ValueError, TypeError) as e:
                self._print(f"[yellow]Error processing manufacturer shipment for drug_id {drug_id}: {e}[/]")
                continue

    def _process_distributor_orders(self, distributor_orders: Dict):
        """Record orders distributor -> manufacturer."""
        for region_id_str, drug_orders in distributor_orders.items():
            try:
                region_id = int(region_id_str)
                distributor_id = region_id + 1
                for drug_id_str, amount in drug_orders.items():
                    try:
                         drug_id = int(drug_id_str)
                         amount = float(amount)
                         if amount > 0.01: # Only record meaningful orders
                             self.order_history.append({
                                 "day": self.current_day, "drug_id": drug_id, "from_id": distributor_id,
                                 "to_id": 0, "amount": amount # Order TO manufacturer (node 0)
                             })
                    except (ValueError, TypeError): continue
            except (ValueError, TypeError): continue

    def _process_distributor_allocation(self, allocation_actions: Dict):
        """Process allocation distributor -> hospital (creates pipeline entries)."""
        # Note: Distributor allocates directly based on its decision, not batched here.
        for region_id_str, drug_allocations in allocation_actions.items():
             try:
                region_id = int(region_id_str)
                distributor_id = region_id + 1
                hospital_id = self.num_regions + 1 + region_id

                for drug_id_str, amount in drug_allocations.items():
                     try:
                        drug_id = int(drug_id_str)
                        amount = float(amount)
                        if amount <= 1e-6: continue # Skip negligible amounts

                        # Check distributor inventory
                        available_inventory = self.inventories.get(drug_id, {}).get(distributor_id, 0.0)
                        actual_allocation = min(amount, available_inventory)

                        if actual_allocation <= 1e-6: continue

                        # Calculate lead time
                        day_idx = min(self.current_day, self.scenario_length - 1)
                        base_lead_time = 1 # Typically short Dist -> Hosp
                        transport_capacity_factor = self.scenario.get_transportation_capacity(day_idx, region_id)
                        adjusted_lead_time = max(1, int(round(base_lead_time / max(0.1, transport_capacity_factor))))
                        arrival_day = self.current_day + adjusted_lead_time # Arrival relative to current day

                        # Add to pipeline (ensure structure exists)
                        self.pipelines.setdefault(distributor_id, {}).setdefault(hospital_id, {}).setdefault(drug_id, []).append((actual_allocation, arrival_day))

                        # Deduct from distributor inventory
                        self.inventories[drug_id][distributor_id] -= actual_allocation

                        # Log allocation history
                        self.allocation_history.append({
                            "day": self.current_day, "drug_id": drug_id, "from_id": distributor_id,
                            "to_id": hospital_id, "amount": actual_allocation, "arrival_day": arrival_day
                        })
                        if self.verbose:
                             drug_name = self.scenario.drugs[drug_id].get("name", f"Drug-{drug_id}")
                             self._print(f"[{Colors.GREEN}]Shipped {actual_allocation:.1f} units of {drug_name} from Dist-{region_id} to Hosp-{region_id} (ETA: Day {arrival_day}).[/]")

                     except (ValueError, TypeError, KeyError) as e:
                         self._print(f"[yellow]Error in distributor allocation for region {region_id}, drug {drug_id}: {e}[/]")
                         continue
             except (ValueError, TypeError): continue

    def _process_hospital_orders(self, hospital_orders: Dict):
        """Record orders hospital -> distributor."""
        for region_id_str, drug_orders in hospital_orders.items():
            try:
                region_id = int(region_id_str)
                hospital_id = self.num_regions + 1 + region_id
                distributor_id = region_id + 1 # Corresponding distributor
                for drug_id_str, amount in drug_orders.items():
                    try:
                        drug_id = int(drug_id_str)
                        amount = float(amount)
                        if amount > 0.01: # Record meaningful orders
                            self.order_history.append({
                                "day": self.current_day, "drug_id": drug_id, "from_id": hospital_id,
                                "to_id": distributor_id, "amount": amount # Order TO distributor
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
                    # Check arrivals against *next* day start (current_day + 1)
                    arrival_check_day = self.current_day + 1
                    arrived = [(amt, day) for amt, day in current_pipeline if day < arrival_check_day]
                    remaining = [(amt, day) for amt, day in current_pipeline if day >= arrival_check_day]

                    total_arrived_amount = sum(amt for amt, day in arrived)

                    if total_arrived_amount > 1e-6:
                        # Ensure target inventory exists
                        self.inventories.setdefault(drug_id, {}).setdefault(to_id, 0.0)
                        self.inventories[drug_id][to_id] += total_arrived_amount
                        if self.verbose:
                             drug_name = self.scenario.drugs[drug_id].get("name", f"Drug-{drug_id}")
                             # Log arrival for the day it happens (current_day)
                             self._print(f"[{Colors.DIM}]Delivery: {total_arrived_amount:.1f} of {drug_name} arrived at Node {to_id} from Node {from_id} on Day {self.current_day+1}[/dim]")

                    # Update pipeline with remaining items
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
        """Process patient demand at hospitals, calculate metrics. (BC UPDATE REMOVED)"""
        for region_id in range(self.num_regions):
            hospital_id = self.num_regions + 1 + region_id

            for drug_id in range(self.num_drugs):
                 try:
                    # Get demand using scenario generator for the current day index
                    day_idx = min(self.current_day, self.scenario_length - 1)
                    demand = self.scenario.get_daily_drug_demand(day_idx, region_id, drug_id)
                    demand = max(0.0, float(demand)) # Ensure non-negative float

                    if demand <= 1e-6: continue # Skip if no demand

                    # Ensure inventory keys exist, default to 0
                    available = self.inventories.get(drug_id, {}).get(hospital_id, 0.0)

                    # Track total demand
                    self.total_demand.setdefault(drug_id, {}).setdefault(region_id, 0.0) # Ensure keys exist
                    self.total_demand[drug_id][region_id] += demand
                    self.demand_history.append({
                        "day": self.current_day, "drug_id": drug_id, "region_id": region_id,
                        "demand": demand, "available": available
                    })

                    fulfilled = min(demand, available)
                    unfulfilled = demand - fulfilled

                    # Update inventory
                    if fulfilled > 0.01: # Only deduct meaningful amounts
                         self.inventories[drug_id][hospital_id] -= fulfilled

                    # Track stockouts and impact
                    if unfulfilled > 1e-6: # If meaningful unfulfilled demand
                        self.stockouts.setdefault(drug_id, {}).setdefault(region_id, 0) # Ensure keys exist
                        self.unfulfilled_demand.setdefault(drug_id, {}).setdefault(region_id, 0.0) # Ensure keys exist
                        self.patient_impact.setdefault(region_id, 0.0) # Ensure keys exist

                        self.stockouts[drug_id][region_id] += 1 # Increment stockout day counter
                        self.unfulfilled_demand[drug_id][region_id] += unfulfilled
                        self.stockout_history.append({
                            "day": self.current_day, "drug_id": drug_id, "region_id": region_id,
                            "demand": demand, "unfulfilled": unfulfilled
                        })
                        # Calculate patient impact (weighted by criticality)
                        drug_criticality = self.scenario.drugs[drug_id].get("criticality_value", 1)
                        impact = unfulfilled * drug_criticality
                        self.patient_impact[region_id] += impact
                        if self.verbose:
                             drug_name = self.scenario.drugs[drug_id].get("name", f"Drug-{drug_id}")
                             self._print(f"[{Colors.STOCKOUT}]Stockout: Hosp-{region_id}, {drug_name}. Unfulfilled: {unfulfilled:.1f}/{demand:.1f} units.[/]")


                 except KeyError as e:
                      self._print(f"[red]Key error processing demand: {e}. Hospital {hospital_id}, Drug {drug_id}.[/]")
                      continue
                 except Exception as e:
                      self._print(f"[red]Unexpected error processing demand for hospital {region_id}, drug {drug_id}: {e}[/]")
                      continue


    def _calculate_rewards(self) -> Dict:
        """Calculate simple rewards (negative values indicating penalties)."""
        rewards = {
            "manufacturer": 0.0,
            "distributors": {r: 0.0 for r in range(self.num_regions)},
            "hospitals": {r: 0.0 for r in range(self.num_regions)}
        }
        # Penalty for overall system inefficiency (unfulfilled demand)
        total_unfulfilled_all = sum(sum(v.values()) for v in self.unfulfilled_demand.values())
        rewards["manufacturer"] -= 0.001 * total_unfulfilled_all # Smaller penalty factor

        # Penalties for regional performance
        for r in range(self.num_regions):
            region_unfulfilled = sum(self.unfulfilled_demand.get(d, {}).get(r, 0.0) for d in range(self.num_drugs))
            regional_patient_impact = self.patient_impact.get(r, 0.0)
            rewards["distributors"][r] -= 0.002 * region_unfulfilled # Penalize regional unfulfillment
            rewards["hospitals"][r] -= 0.01 * regional_patient_impact # Penalize patient impact directly

        return rewards

    def _calculate_fair_allocation(
            self,
            drug_id: int,
            requested_amounts_dict: Dict[int, float], # {region_id: requested_amount}
            available_inventory: float
        ) -> Optional[Dict[int, float]]:
        """
        Calculate fair allocation. Uses blockchain if enabled, otherwise local logic.
        Incorporates projection adjustment before calling blockchain.

        Args:
            drug_id: ID of the drug.
            requested_amounts_dict: Dict of {region_id: requested_amount}.
            available_inventory: Current manufacturer inventory.

        Returns:
            Dict {region_id: allocated_amount} or None if calculation fails.
        """
        if available_inventory <= 1e-6:
            return {r_id: 0.0 for r_id in requested_amounts_dict}

        # --- Try Blockchain Strategy First (if enabled) ---
        if self.use_blockchain and self.blockchain:
            try:
                region_ids = list(requested_amounts_dict.keys())
                original_requests_list = list(requested_amounts_dict.values())

                # --- Adjust requests based on projection before sending to BC ---
                adjusted_requests_list = []
                # Use current_day index for projection lookup
                current_day_idx = min(self.current_day, self.scenario_length - 1)
                projection_weight = 0.6 # TUNABLE: How much to weigh projection vs agent request
                max_boost_factor = 2.0 # TUNABLE: Max factor to boost request based on projection

                for i, region_id in enumerate(region_ids):
                    agent_request = max(0.0, original_requests_list[i]) # Ensure non-negative
                    projected_demand = max(0.0, self.scenario.get_daily_drug_demand(current_day_idx, region_id, drug_id))

                    # Blending logic: If projection is higher, blend towards it, cap boost
                    if projected_demand > agent_request and agent_request > 0:
                        boost_ratio = projected_demand / agent_request
                        effective_boost = min(max_boost_factor, 1.0 + (boost_ratio - 1.0) * projection_weight)
                        adjusted_request = agent_request * effective_boost
                    elif projected_demand > 0: # If agent requested 0 but projection > 0
                         adjusted_request = projected_demand * projection_weight # Start with a portion of projection
                    else: # If projection is lower or zero, stick closer to agent request
                         adjusted_request = (agent_request * (1 - projection_weight)) + (projected_demand * projection_weight)

                    adjusted_requests_list.append(max(0.0, adjusted_request)) # Ensure non-negative final

                if self.verbose:
                    print_orig = {r_id: f"{req:.1f}" for r_id, req in requested_amounts_dict.items()}
                    print_adj = {r_id: f"{adj:.1f}" for r_id, adj in zip(region_ids, adjusted_requests_list)}
                    if print_orig != print_adj: # Only print if adjustment happened
                         self._print(f"[{Colors.BLOCKCHAIN}]Adjusting requests for BC alloc (Drug {drug_id}). Original: {print_orig}, Adjusted: {print_adj}[/]")
                # ---------------------------------------------------------------------


                # Ensure lists are not empty before calling
                if not region_ids or not adjusted_requests_list:
                     self._print(f"[{Colors.BLOCKCHAIN}][yellow]Blockchain allocation for Drug {drug_id}: No valid requests provided after adjustment.[/]")
                     return {r_id: 0.0 for r_id in requested_amounts_dict}

                self._print(f"[{Colors.BLOCKCHAIN}]Using blockchain allocation strategy for Drug-{drug_id}...[/]")
                bc_allocations_dict = self.blockchain.execute_fair_allocation(
                    drug_id=int(drug_id),
                    region_ids=region_ids,
                    requested_amounts=adjusted_requests_list, # *** USE ADJUSTED REQUESTS ***
                    available_inventory=available_inventory
                )

                if bc_allocations_dict is not None:
                    self._print(f"[{Colors.BLOCKCHAIN}]Blockchain allocation result for Drug-{drug_id}: { {k: f'{v:.1f}' for k, v in bc_allocations_dict.items()} }[/]")
                    # Ensure all original requesting regions have an entry (even if 0)
                    final_alloc = {r_id: 0.0 for r_id in requested_amounts_dict}
                    total_allocated_bc = 0
                    for r_id_res, amount_res in bc_allocations_dict.items():
                         # Ensure result keys are handled correctly if BC returns different keys/types
                         try: r_id_int = int(r_id_res)
                         except (ValueError, TypeError): continue
                         if r_id_int in final_alloc: # Only update if it was an original requestor
                             final_alloc[r_id_int] = max(0.0, amount_res)
                             total_allocated_bc += max(0.0, amount_res)

                    # Sanity check total allocation from BC
                    if total_allocated_bc > available_inventory * 1.01: # Allow 1% tolerance for float/int issues
                        self._print(f"[{Colors.BLOCKCHAIN}][yellow]Warning: Blockchain allocation for Drug {drug_id} exceeded available ({total_allocated_bc:.1f} > {available_inventory:.1f}). Scaling down.[/]")
                        if total_allocated_bc > 1e-6:
                            scale_down = available_inventory / total_allocated_bc
                            final_alloc = {r_id: amount * scale_down for r_id, amount in final_alloc.items()}
                        else:
                            final_alloc = {r_id: 0.0 for r_id in final_alloc}

                    return final_alloc
                else:
                    self._print(f"[{Colors.BLOCKCHAIN}][yellow]Blockchain allocation strategy call failed for Drug-{drug_id}. Falling back to local logic.[/]")
                    # Fall through to local calculation

            except Exception as e:
                self._print(f"[{Colors.BLOCKCHAIN}][yellow]Blockchain allocation strategy error for Drug-{drug_id}: {e}. Falling back to local logic.[/]")
                # Fall through to local calculation

        # --- Local Fair Allocation Fallback ---
        self._print(f"[{Colors.FALLBACK}]Using local allocation logic for Drug-{drug_id}...[/]")
        drug_info = self.scenario.drugs[drug_id]
        # Get cases from simulation curve ONLY FOR LOCAL FALLBACK
        # Use the function to get current cases for the day index
        region_cases_fallback = self.get_current_simulation_cases()

        # Call the centralized tool logic (imported)
        # Pass original requests here, as projection adjustment was for BC call
        local_allocations = allocation_priority_tool(
            drug_info, requested_amounts_dict, region_cases_fallback, available_inventory
            )
        if self.verbose:
            self._print(f"[{Colors.FALLBACK}]Local allocation result for Drug-{drug_id}: { {k: f'{v:.1f}' for k, v in local_allocations.items()} }[/dim]")
        return local_allocations


    # --- Observation Methods ---

    def _get_manufacturer_observation(self) -> Dict:
        """Get observation for manufacturer agent, EXCLUDING current cases/trend."""
        # Use current_day index for lookups related to scenario state
        current_day_idx = min(self.current_day, self.scenario_length - 1)

        obs = {
            "day": self.current_day, # Still represents the start of the day (0-based index)
            "inventories": {str(drug_id): self.inventories.get(drug_id, {}).get(0, 0.0) for drug_id in range(self.num_drugs)},
            "warehouse_inventories": {str(drug_id): self.warehouse_inventories.get(drug_id, 0.0) for drug_id in range(self.num_drugs)},
            "production_capacity": {str(drug_id): self.scenario.get_manufacturing_capacity(current_day_idx, drug_id) for drug_id in range(self.num_drugs)},
            "pipeline": {}, # Outgoing pipeline (Manu -> Dists) - Simplified view
            "recent_orders": [], # Incoming orders from distributors (Dist -> Manu)
            "drug_info": {str(i): d for i, d in enumerate(self.scenario.drugs)},
            # Epidemiological data only contains projected demand
            "epidemiological_data": {},
            "disruptions": [d for d in self.scenario.disruptions if isinstance(d, dict) and d.get("type") == "manufacturing" and d.get("drug_id") is not None and d.get("start_day", -1) <= current_day_idx <= d.get("end_day", self.scenario_length)],
            "pending_releases": [], # From warehouse -> Manu
            "pending_allocations": getattr(self, 'pending_allocations', {}),
            "batch_allocation_frequency": getattr(self, 'allocation_batch_frequency', 1),
            "is_batch_processing_day": (self.current_day + 1) % max(1, getattr(self, 'allocation_batch_frequency', 1)) == 0 if self.current_day >=0 else False, # Check based on 1-based day number
            "days_to_next_batch_process": 0,
            "downstream_inventory_summary": {str(drug_id): {"total_distributor": 0.0, "total_hospital": 0.0, "total_downstream": 0.0} for drug_id in range(self.num_drugs)},
            "downstream_pipeline_summary": {str(drug_id): {"manu_to_dist": 0.0, "dist_to_hosp": 0.0, "total_inbound_downstream": 0.0} for drug_id in range(self.num_drugs)},
            "downstream_projected_demand_summary": {str(drug_id): 0.0 for drug_id in range(self.num_drugs)},
        }

        # Calculate days_to_next_batch_process (using 0-based current_day)
        freq = obs["batch_allocation_frequency"]
        if freq > 0:
            # How many days *including today* until the next batch day?
            # Batch happens at END of day `k*freq - 1` (0-based index)
            current_cycle_day_index = self.current_day % freq
            days_remaining_in_cycle = freq - current_cycle_day_index
            obs["days_to_next_batch_process"] = days_remaining_in_cycle
            # next batch processing day index is current_day + days_remaining_in_cycle - 1
            obs["next_batch_process_day"] = self.current_day + days_remaining_in_cycle -1
        else: # Continuous allocation
            obs["days_to_next_batch_process"] = 0
            obs["next_batch_process_day"] = self.current_day

        # Populate Downstream Summaries and Projected Demand
        for drug_id in range(self.num_drugs):
            drug_id_str = str(drug_id)
            total_dist_inv = 0.0; total_hosp_inv = 0.0
            total_dist_pipeline = 0.0; total_hosp_pipeline = 0.0
            total_proj_demand = 0.0

            for region_id in range(self.num_regions):
                region_id_str = str(region_id)
                dist_id = region_id + 1
                hosp_id = self.num_regions + 1 + region_id

                # Populate ONLY projected demand into epidemiological_data
                # Use current_day_idx for demand lookup
                projected_demand = self.scenario.get_daily_drug_demand(current_day_idx, region_id, drug_id)
                if region_id_str not in obs["epidemiological_data"]:
                    obs["epidemiological_data"][region_id_str] = {}
                # Store projected demand per drug for this region
                obs["epidemiological_data"][region_id_str].setdefault("projected_demand", {})[drug_id_str] = projected_demand

                # Summaries remain the same
                total_dist_inv += self.inventories.get(drug_id, {}).get(dist_id, 0.0)
                total_hosp_inv += self.inventories.get(drug_id, {}).get(hosp_id, 0.0)
                total_dist_pipeline += sum(amount for amount, _ in self.pipelines.get(0, {}).get(dist_id, {}).get(drug_id, []))
                total_hosp_pipeline += sum(amount for amount, _ in self.pipelines.get(dist_id, {}).get(hosp_id, {}).get(drug_id, []))
                total_proj_demand += projected_demand # Sum total projected demand

            # Store summarized info
            obs["downstream_inventory_summary"][drug_id_str]["total_distributor"] = total_dist_inv
            obs["downstream_inventory_summary"][drug_id_str]["total_hospital"] = total_hosp_inv
            obs["downstream_inventory_summary"][drug_id_str]["total_downstream"] = total_dist_inv + total_hosp_inv
            obs["downstream_pipeline_summary"][drug_id_str]["manu_to_dist"] = total_dist_pipeline
            obs["downstream_pipeline_summary"][drug_id_str]["dist_to_hosp"] = total_hosp_pipeline
            obs["downstream_pipeline_summary"][drug_id_str]["total_inbound_downstream"] = total_dist_pipeline + total_hosp_pipeline
            obs["downstream_projected_demand_summary"][drug_id_str] = total_proj_demand

        # Populate Outgoing Pipeline (Manu -> All Dists) - Simplified view
        obs["pipeline"]["total_to_distributors"] = {}
        for drug_id in range(self.num_drugs):
             total_outgoing = sum(amount for r_id in range(self.num_regions) for amount, _ in self.pipelines.get(0, {}).get(r_id + 1, {}).get(drug_id, []))
             obs["pipeline"]["total_to_distributors"][str(drug_id)] = total_outgoing

        # Populate Recent Orders (Dist -> Manu) - Look back from current_day
        lookback_days = 7
        obs["recent_orders"] = [o for o in self.order_history if o.get("to_id") == 0 and o.get("day", -1) >= self.current_day - lookback_days]

        # Populate Pending Releases (Warehouse -> Manu)
        release_delay = getattr(self, 'warehouse_release_delay', 1)
        obs["pending_releases"] = [{
            "drug_id": str(entry["drug_id"]), "amount": entry["amount"], "production_day": entry["day"],
            "days_in_warehouse": self.current_day - entry["day"],
            "expected_release_day": entry["day"] + release_delay # Arrival day index
        } for entry in self.production_history if not entry.get("released", False)]

        return obs

    def _get_distributor_observation(self, region_id: int) -> Dict:
        """Get observation for distributor agent, EXCLUDING current cases/trend."""
        distributor_id = region_id + 1
        hospital_id = self.num_regions + 1 + region_id
        current_day_idx = min(self.current_day, self.scenario_length - 1)

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
            # Epidemiological data only contains projected demand
            "epidemiological_data": {},
            "disruptions": [d for d in self.scenario.disruptions if isinstance(d, dict) and d.get("type") == "transportation" and d.get("region_id") == region_id and d.get("start_day", -1) <= current_day_idx <= d.get("end_day", self.scenario_length)]
        }

        # Populate Projected Demand for the hospital in this region
        projected_demand_this_region = {}
        for drug_id in range(self.num_drugs):
            projected_demand_this_region[str(drug_id)] = self.scenario.get_daily_drug_demand(current_day_idx, region_id, drug_id)
        # Ensure the nested structure is created correctly
        obs["epidemiological_data"].setdefault("projected_demand", {}).update(projected_demand_this_region)

        # Populate Inbound Pipeline (Manu -> This Dist)
        for drug_id in range(self.num_drugs):
             obs["inbound_pipeline"][str(drug_id)] = sum(amount for amount, _ in self.pipelines.get(0, {}).get(distributor_id, {}).get(drug_id, []))

        # Populate Outbound Pipeline (This Dist -> Hosp)
        for drug_id in range(self.num_drugs):
             obs["outbound_pipeline"][str(drug_id)] = sum(amount for amount, _ in self.pipelines.get(distributor_id, {}).get(hospital_id, {}).get(drug_id, []))

        # Populate Recent Orders (Hosp -> This Dist) - Look back from current_day
        lookback_days = 7
        obs["recent_orders"] = [o for o in self.order_history if o.get("to_id") == distributor_id and o.get("day", -1) >= self.current_day - lookback_days]

        # Populate Recent Allocations (Manu -> This Dist) - Look back from current_day
        obs["recent_allocations"] = [a for a in self.allocation_history if a.get("to_id") == distributor_id and a.get("day", -1) >= self.current_day - lookback_days]

        return obs

    def _get_hospital_observation(self, region_id: int) -> Dict:
        """Get observation for hospital agent, EXCLUDING current cases/trend."""
        hospital_id = self.num_regions + 1 + region_id
        distributor_id = region_id + 1
        current_day_idx = min(self.current_day, self.scenario_length - 1)

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
            # Epidemiological data only contains projected demand
            "epidemiological_data": {},
            "disruptions": [d for d in self.scenario.disruptions if isinstance(d, dict) and d.get("type") == "transportation" and d.get("region_id") == region_id and d.get("start_day", -1) <= current_day_idx <= d.get("end_day", self.scenario_length)]
        }

        # Populate Projected Demand for this hospital's region
        projected_demand_this_region = {}
        for drug_id in range(self.num_drugs):
            projected_demand_this_region[str(drug_id)] = self.scenario.get_daily_drug_demand(current_day_idx, region_id, drug_id)
        # Ensure the nested structure is created correctly
        obs["epidemiological_data"].setdefault("projected_demand", {}).update(projected_demand_this_region)

        # Populate Inbound Pipeline (Dist -> This Hosp)
        for drug_id in range(self.num_drugs):
            obs["inbound_pipeline"][str(drug_id)] = sum(amount for amount, _ in self.pipelines.get(distributor_id, {}).get(hospital_id, {}).get(drug_id, []))

        # Populate Recent Allocations (Dist -> This Hosp) - Look back from current_day
        lookback_days = 7
        obs["recent_allocations"] = [a for a in self.allocation_history if a.get("to_id") == hospital_id and a.get("day", -1) >= self.current_day - lookback_days]

        # Populate Demand History (This Hosp) - Look back from current_day
        obs["demand_history"] = [d for d in self.demand_history if d.get("region_id") == region_id and d.get("day", -1) >= self.current_day - lookback_days]

        # Populate Stockout History (This Hosp) - Look back from current_day
        obs["stockout_history"] = [s for s in self.stockout_history if s.get("region_id") == region_id and s.get("day", -1) >= self.current_day - lookback_days]

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
        # Day starts at self.current_day (0-based index)

        # 1. Production (to warehouse) - Based on decisions for current day
        self._process_production(actions.get("manufacturer_production", {}))

        # 2. Warehouse Release (to manufacturer inventory) - Based on past production + delay
        self._process_warehouse_release() # Uses self.warehouse_release_delay

        # 3. Record Orders (does not move inventory) - Based on decisions for current day
        self._process_distributor_orders(actions.get("distributor_orders", {}))
        self._process_hospital_orders(actions.get("hospital_orders", {}))

        # 4. Process Allocations (accumulates or triggers shipments)
        #    Manufacturer allocation requests are batched/processed based on current day & frequency
        self._process_batch_allocation(actions.get("manufacturer_allocation", {}))
        #    Distributor allocation happens daily based on current decision
        self._process_distributor_allocation(actions.get("distributor_allocation", {}))

        # 5. Process Deliveries (checks pipelines for items arriving *before start of next day*)
        self._process_deliveries()

        # 6. Process Patient Demand (consumes hospital inventory based on current day's demand)
        self._process_patient_demand()

        # 7. Increment Day and Record History AFTER all state changes for current_day index
        self._record_daily_history() # Record state at the *end* of current_day
        self.current_day += 1       # Advance to the next day index

        # 8. Calculate Rewards and Check Termination
        rewards = self._calculate_rewards()
        done = self.current_day >= self.scenario_length # Check if we have completed the last day
        observations = {}
        if not done:
            observations = self.get_observations() # Get observations for the START of the *next* day

        # Info dictionary for logging/analysis (reflects state *after* step actions)
        info = {
            "stockouts": {d: r.copy() for d, r in self.stockouts.items()},
            "unfulfilled_demand": {d: r.copy() for d, r in self.unfulfilled_demand.items()},
            "patient_impact": self.patient_impact.copy(),
            "current_day": self.current_day, # Day number we are about to start (1-based for logs)
            "warehouse_inventory": self.warehouse_inventories.copy(),
            "manufacturer_inventory": {str(drug_id): self.inventories.get(drug_id, {}).get(0, 0.0) for drug_id in range(self.num_drugs)},
            "pending_allocations": getattr(self, 'pending_allocations', {}) # Show pending for next batch
        }

        return observations, rewards, done, info

# --- END OF FILE src/environment/supply_chain.py ---